import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig
from torch import nn
from torch.optim import AdamW
from torchmetrics import MeanMetric
from waymo_open_dataset.metrics.python import config_util_py

from mtr.datasets.waymo.waymo_eval import (
    _default_metrics_config,
    generate_final_pred_dicts,
    waymo_motion_metrics,
)
from mtr.models.context_encoder.mtr_encoder import MTREncoder
from mtr.models.motion_decoder.mtr_decoder import MTRDecoder
from mtr.models.utils.adams import AdamS
from mtr.utils.lr_scheduler import WarmupLinearLR
from tools.evaluation_utils.angled_dist_metrics import (
    evaluation_minFRDE,
    get_metric_keys,
)
from utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class MTRLiDARModule(L.LightningModule):
    def __init__(
        self,
        CONTEXT_ENCODER: dict,
        MOTION_DECODER: dict,
        optimizer: Union[torch.optim.Optimizer, dict],
        scheduler: Union[torch.optim.lr_scheduler.LRScheduler, dict],
        precision: str = "highest",
        val_output_path: Optional[str] = None,
    ):
        """MTR module with LiDAR capabilities.

        Args:
            CONTEXT_ENCODER (dict): config of context encoder
            MOTION_DECODER (dict): config of motion decoder
            optimizer (Union[torch.optim.Optimizer, dict]): optimizer settings
            scheduler (Union[torch.optim.lr_scheduler.LRScheduler, dict]): scheduler settings
            precision (str, optional): torch precision to use. Defaults to "highest".
            val_output_path (Optional[str], optional): path to save validation output if none does not save. Defaults to None.
        """
        super().__init__()
        # for learning rate finder
        self.lr = optimizer["lr"]

        self.val_output_path = val_output_path

        # set precision on matmul calculations. Gives potential speed up but might be slightly less accuracte
        torch.set_float32_matmul_precision(precision)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # build context encoder and decoder part
        self.context_encoder = MTREncoder(config=CONTEXT_ENCODER)
        self.motion_decoder = MTRDecoder(in_channels=self.context_encoder.num_out_channels, config=MOTION_DECODER)

        # set of all val metrics
        self.waymo_metric_keys = ["mean_average_precision", "overlap_rate", "miss_rate", "min_fde", "min_ade"]
        self.val_metrics = self.initialize_val_metrics()
        # collect output model during validation to run metrics on at the end
        self.validation_predictions = []
        # set to false to log scenario id on every process
        self.showed_first_batch = True

    def initialize_val_metrics(self):
        needed_keys = []
        eval_config = _default_metrics_config(eval_second=8, num_modes_for_eval=6)
        metric_names = config_util_py.get_breakdown_names_from_motion_config(eval_config)

        # get waymo metric keys
        for key in self.waymo_metric_keys:
            for name in metric_names:
                needed_keys.append(f"{key}/{name}")
                obj_type = name.rsplit("_", 1)[0]
                needed_keys.append(f"{key}/{obj_type}")

        # get lateral and longitudinal keys
        needed_keys.extend(get_metric_keys(obj_types=self.hparams.MOTION_DECODER["OBJECT_TYPE"]))

        val_metrics = nn.ModuleDict()
        for key in needed_keys:
            val_metrics.add_module(key, MeanMetric())

        return val_metrics

    def forward(self, batch):
        batch = self.context_encoder(batch)
        batch = self.motion_decoder(batch)

        return batch

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.validation_predictions.clear()

        for _, val in self.val_metrics.items():
            val.reset()

    def add_prefix_metrics(self, metrics_dict: dict, prefix: str) -> dict:
        new_dict = {}
        for key, val in metrics_dict.items():
            new_dict[f"{prefix}{key}"] = val

        return new_dict

    def training_step(self, batch, batch_idx):
        if not self.showed_first_batch:
            self.showed_first_batch = True
            logger.info(f"Scenario ids: {batch['input_dict']['scenario_id']}", all_ranks=True)

        batch = self.forward(batch)

        loss, metrics_dict = self.motion_decoder.get_loss()

        if "input_trans" in batch:
            # save loss before running
            metrics_dict["loss/motions_loss"] = loss.item()

            # calculate regularization loss
            pointnet_reg_loss = self.context_encoder.lidar_pointnet_encoder.get_reg_loss(batch)
            loss += pointnet_reg_loss * self.hparams.CONTEXT_ENCODER.get("REGULIZAR_WEIGHT_POINTNET")

            metrics_dict["loss/pointnet_reg_loss"] = pointnet_reg_loss

        # log loss metrics and ade
        self.log(
            "train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch["batch_size"]
        )
        train_metrics = self.add_prefix_metrics(metrics_dict, prefix="train/")
        self.log_dict(train_metrics, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.forward(batch)
        assert self.context_encoder.map_polyline_encoder.mlps[1].training is False, "Model must be in validation mode"
        final_pred_dicts = generate_final_pred_dicts(batch)

        self.validation_predictions.extend(final_pred_dicts)

    def calculate_validation_metrics(self):
        metric_results, metric_names, object_type_cnt_dict = waymo_motion_metrics(
            self.validation_predictions, num_modes_for_eval=6, gpu_id=self.trainer.local_rank
        )

        # track waymo motion metric results
        for key, val in metric_results.items():
            val = val.numpy()
            for i, name in enumerate(metric_names):
                if val[i] != -1:  # skip -1 if no object
                    metric_key = f"{key}/{name}"
                    obj_type = name.rsplit("_", 1)[0]
                    # metric of object on timestamp
                    self.update_val_metrics(key=metric_key, value=val[i], weight=object_type_cnt_dict[obj_type])
                    # mean metric of object over all timestamp
                    self.update_val_metrics(
                        key=f"{key}/{obj_type}", value=val[i], weight=object_type_cnt_dict[obj_type]
                    )

        # lateral and longitudinal metrics
        angled_errors = evaluation_minFRDE(pred_dicts=self.validation_predictions)
        for key, val in angled_errors.items():
            for obj_type in ["TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"]:
                if obj_type in key:
                    self.update_val_metrics(key=key, value=val, weight=object_type_cnt_dict[obj_type])

        # remove predictions again
        self.validation_predictions.clear()

    def update_val_metrics(self, key: str, value: float, weight: int):
        if key not in self.val_metrics:
            logger.warn(f"Trying to log unknown key: {key}")
            return

        self.val_metrics[key].update(value=value, weight=weight)

    def save_model_predictions(self, model_predictions: list):
        path = Path(self.val_output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, mode="wb") as fp:
            # highest protocol for faster saving of large numpy arrays
            pickle.dump(model_predictions, fp, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved validation results to: {path}")

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        if self.val_output_path:
            self.save_model_predictions(self.validation_predictions)

        # empty cash before calculating metrics as this is done with tensorflow using GPU
        torch.cuda.empty_cache()
        self.calculate_validation_metrics()

        # log results
        mean_metrics = defaultdict(list)
        for key, metric in self.val_metrics.items():
            if "TYPE" not in key:  # skip average for next iteration
                continue
            value = metric.compute()
            # add other metrics to average metrics
            metric_name = key.split("/")[0]
            if metric_name in ["along", "across"]:
                # make name of average metric without the type
                add_to_key = f'{metric_name}/MDE_{key.split("_")[-1]}'
                if metric.weight > 0:  # only add if any elements are added
                    mean_metrics[add_to_key].append(value.cpu().numpy())
            else:
                if not key.endswith("5") and not key.endswith("9"):  # skip metric of a time, 5, 9, 15
                    # waymo mean metrics over type, same weight
                    if metric.weight > 0:  # only add if any elements are added
                        mean_metrics[metric_name].append(value.cpu().numpy())

            self.log(
                f"val/{key}", value, on_epoch=True, on_step=False, batch_size=1, rank_zero_only=True, sync_dist=True
            )

        # log average metrics separate, because classes have equal weights
        # only rank zero will log this
        for key, values in mean_metrics.items():
            value = np.mean(values)
            self.log(
                f"val/{key}",
                value,
                on_epoch=True,
                on_step=False,
                batch_size=1,
                rank_zero_only=True,
                sync_dist=True,
            )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.hparams.optimizer["name"] == "AdamW":
            optimizer = AdamW(
                params=self.trainer.model.parameters(), lr=self.lr, weight_decay=self.hparams.optimizer["weight_decay"]
            )
        elif self.hparams.optimizer["name"] == "AdamS":
            optimizer = AdamS(
                params=self.trainer.model.parameters(), lr=self.lr, weight_decay=self.hparams.optimizer["weight_decay"]
            )
        else:
            raise ValueError(f"Unknown optimizer, {self.hparams.optimizer}")
            # optimizer = self.hparams.optimizer(params=self.trainer.model.parameters(), lr=self.lr)

        if self.hparams.scheduler is not None:
            if isinstance(self.hparams.scheduler, DictConfig) or isinstance(self.hparams.scheduler, dict):
                if self.hparams.scheduler["name"] == "warmup-lineardecay":
                    accumulate_gradient = 1
                    if self.trainer.accumulate_grad_batches is not None:
                        accumulate_gradient = self.trainer.accumulate_grad_batches
                    total_iters = (
                        self.trainer.max_epochs
                        * self.trainer.datamodule.nr_scenes_training
                        // (self.trainer.datamodule.batch_size * accumulate_gradient)
                    )
                    warmup_iters = int(total_iters * self.hparams.scheduler["warmup_percentage"])
                    logger.info(f"Using warmup LR, total iters: {total_iters} whereof {warmup_iters} warmup iters")
                    scheduler = WarmupLinearLR(
                        optimizer=optimizer,
                        warmup_steps=warmup_iters,
                        total_steps=total_iters,
                        **self.hparams.scheduler["kwargs"],
                    )
                    # scheduler needs to be stepped every batch
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": scheduler,
                            "interval": "step",
                            "frequency": 1,
                        },
                    }
                else:
                    raise ValueError(f"Unknown scheduler, {self.hparams.scheduler}")
            else:
                scheduler = self.hparams.scheduler(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
        return {"optimizer": optimizer}
