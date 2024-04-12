import copy
import os
from typing import List, Optional

import hydra
import lightning as L
import numpy as np
import polars as pl
import rootutils
import torch
from lightning import LightningDataModule, LightningModule
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from mtr.models.mtr_module import MTRLiDARModule

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from mtr.datasets.waymo.waymo_eval import (
    generate_final_pred_dicts,
    waymo_motion_metrics,
)
from mtr.datasets.waymo_module import WebWaymoModule
from mtr.utils.motion_utils import batch_nms, batch_nms_typed
from utils.pylogger import RankedLogger
from utils.utils import extras

# create resolver for wandb group
OmegaConf.register_new_resolver("join_tags", lambda x: "_".join(x))

log = RankedLogger(__name__, rank_zero_only=True)


def combine_input_dicts(input_dicts: List[dict]):
    new_input_dict = {}
    for key in [
        "scenario_id",
        "center_objects_world",
        "center_objects_id",
        "center_objects_type",
        "center_gt_trajs_src",
        "track_index_to_predict",
    ]:
        if isinstance(input_dicts[0][key], torch.Tensor):
            new_input_dict[key] = torch.concatenate([in_dict[key] for in_dict in input_dicts], dim=0)
        else:
            new_input_dict[key] = np.concatenate([in_dict[key] for in_dict in input_dicts], axis=0)

    return new_input_dict


def extract_keys_input(input_dict: dict):
    new_input_dict = {}

    for key in [
        "scenario_id",
        "center_objects_world",
        "center_objects_id",
        "center_objects_type",
        "center_gt_trajs_src",
        "track_index_to_predict",
    ]:
        if isinstance(input_dict[key], torch.Tensor):
            new_input_dict[key] = input_dict[key].detach().cpu()
        else:
            new_input_dict[key] = input_dict[key]

    return new_input_dict


def move_data_to_device(batch: dict, device: str):
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(device)
        if isinstance(val, dict):
            batch[key] = move_data_to_device(batch[key], device=device)

    return batch


def model_nms_search(model, test_loader, use_batches: int = 2000):
    # run model on test loader and get full predictions
    model.cuda()
    model.eval()

    pred_scores = []
    pred_trajs = []
    obj_types = []

    all_input_dicts = []
    for i, batch_dict in tqdm(enumerate(test_loader)):
        if use_batches > 0 and i > use_batches:
            break
        batch_dict = move_data_to_device(batch_dict, "cuda")
        with torch.no_grad():
            model(batch_dict)
            pred_list = model.motion_decoder.forward_ret_dict["pred_list"][-1]
            pred_scores.append(torch.softmax(pred_list[0].detach().cpu(), dim=-1))
            pred_trajs.append(pred_list[1].detach().cpu())
            obj_types.append(batch_dict["input_dict"]["center_objects_type"])

            # final_pred_dicts = add_predlist_preddict(model, final_pred_dicts)
            all_input_dicts.append(extract_keys_input(batch_dict["input_dict"]))

        torch.cuda.empty_cache()

    # combine predictions
    pred_scores = torch.concatenate(pred_scores, axis=0)
    pred_trajs = torch.concatenate(pred_trajs, axis=0)
    obj_types = np.concatenate(obj_types, axis=0)
    input_dicts_com = combine_input_dicts(all_input_dicts)

    torch.cuda.empty_cache()
    return pred_trajs, pred_scores, obj_types, input_dicts_com


def nms_search(pred_trajs, pred_scores, obj_types, input_dicts_com, output_dir: str):
    # run nms HP search on distances
    search = np.arange(0.0, 12.5, 0.25)
    results = []
    all_results = []

    log.info("Running the NMS calculation for ", search)
    for dist in tqdm(search):
        # run nms with different dist
        ret_trajs, ret_scores, ret_idxs = batch_nms(pred_trajs, pred_scores, dist_thresh=dist)
        # create pred dicts with new trajs and scores
        pred_dicts = {"input_dict": input_dicts_com, "pred_scores": ret_scores, "pred_trajs": ret_trajs}
        final_pred_dicts = generate_final_pred_dicts(pred_dicts)

        # gather statistics on performance
        metric_results, metric_names, object_type_cnt_dict = waymo_motion_metrics(
            pred_dicts=final_pred_dicts, num_modes_for_eval=6
        )
        all_results.append([dist, metric_results])
        res_map = metric_results["mean_average_precision"].numpy().reshape(3, 3).mean(axis=1)
        res_ade = metric_results["min_ade"].numpy().reshape(3, 3).mean(axis=1)
        for i, key in enumerate(["VEHICLE", "PEDESTRIAN", "CYCLIST"]):
            results.append([key, dist, res_map[i], res_ade[i]])

    # print results of best nms dist
    df = pl.DataFrame(results, ["obj_type", "dist_thres", "map", "min_ade"])
    df_best_map = df.group_by(by="obj_type").agg([pl.all().sort_by("map").last()])
    df_best_ade = df.group_by(by="obj_type").agg([pl.all().sort_by("min_ade").last()])

    df.write_csv(os.path.join(output_dir, "nms_search_results.csv"))
    df_best_map.write_csv(os.path.join(output_dir, "nms_search_best.csv"))
    plot_results(df, df_best_map, key="map", output_dir=output_dir)
    plot_results(df, df_best_ade, key="min_ade", output_dir=output_dir)


def plot_results(df: pl.DataFrame, df_best: pl.DataFrame, key: str, output_dir: str):
    fig, ax = plt.subplots()
    for obj_type in ["CYCLIST", "PEDESTRIAN", "VEHICLE"]:
        df_obj = df.filter(pl.col("obj_type") == obj_type)
        plt.plot(df_obj.select("dist_thres"), df_obj.select(key), label=obj_type)
    plt.title(f"NMS search dist threshold for {key}")
    plt.xlabel("Dist Threshold (m)")
    plt.ylabel(f"{key}")
    plt.legend()

    ymin, ymax = df[key].min(), df[key].max()
    colors = ["tab:blue", "tab:orange", "tab:green"]
    df_best = df_best.sort(by="obj_type")
    previous_dist_thres = []
    for row, c in zip(df_best.iter_rows(named=True), colors):
        small_offset = 0
        if row["dist_thres"] in previous_dist_thres:
            small_offset = 0.04
        previous_dist_thres.append(row["dist_thres"])
        plt.vlines(row["dist_thres"] + small_offset, ymin=ymin - 0.02, ymax=ymax + 0.02, colors=c)
    plt.grid()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))

    plt.savefig(os.path.join(output_dir, f"nms_search_{key}.png"))


@hydra.main(version_base="1.3", config_path="../configs", config_name="nms_search.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Entry point to run NMS search.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data.name}>")
    datamodule: LightningDataModule = WebWaymoModule(cfg.data)
    datamodule.setup()

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    if cfg.ckpt_path is not None:
        log.info(f"Loading checkpoint {cfg.ckpt_path}")
        model = MTRLiDARModule.load_from_checkpoint(cfg.ckpt_path)

    # collect output
    log.info("Running model on validation data to collect prediction")
    data_loader = datamodule.val_dataloader()
    if cfg.get("data_loader") == "train":
        data_loader = datamodule.train_dataloader()
    pred_trajs, pred_scores, obj_types, input_dicts_com = model_nms_search(
        model, data_loader, use_batches=cfg.num_batches
    )

    # run the nms search
    nms_search(
        pred_scores=pred_scores,
        pred_trajs=pred_trajs,
        obj_types=obj_types,
        input_dicts_com=input_dicts_com,
        output_dir=cfg.paths.output_dir,
    )


if __name__ == "__main__":
    main()
