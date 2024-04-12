import os
from typing import Any, Dict, List, Optional

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig, OmegaConf

from mtr.datasets.tno.tno_module import TNOmodule
from mtr.models.mtr_module import MTRLiDARModule

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from mtr.datasets.waymo_module import WebWaymoModule
from utils.instantiators import instantiate_callbacks, instantiate_loggers
from utils.logging_utils import log_hyperparameters
from utils.pylogger import RankedLogger
from utils.utils import extras, get_metric_value, task_wrapper

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#   (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

# create resolver for wandb group
OmegaConf.register_new_resolver("join_tags", lambda x: "_".join(x))


log = RankedLogger(__name__, rank_zero_only=True)


# in case of failure log
@task_wrapper
def train(cfg: DictConfig) -> Dict[str, Any]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data.name}>")
    if cfg.data.name == "WaymoWebDataset":
        datamodule: LightningDataModule = WebWaymoModule(cfg.data)
    elif cfg.data.name == "TnoDataset":
        datamodule: LightningDataModule = TNOmodule(cfg.data)
    else:
        raise ValueError("Unknown dataset")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # if cfg.get("ckpt_path"):
    #     # load model without strict, to allow for using part of the model
    #     model_settings = OmegaConf.to_container(cfg.model, resolve=True)
    #     model = MTRLiDARModule.load_from_checkpoint(cfg.get("ckpt_path"), strict=False, **model_settings)

    #     if cfg.get("freeze_std"):
    #         # freeze the whole model, but unfreeze lidar point net encoder
    #         model.freeze()
    #         # unfreeze lidar pointnet encoder
    #         for param in model.context_encoder.lidar_pointnet_encoder.parameters():
    #             param.requires_grad = True

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("batch_size_finder"):
        # Create a tuner for the trainer
        tuner = Tuner(trainer)

        # Auto-scale batch size with binary search
        tuner.scale_batch_size(model=model, datamodule=datamodule, mode="binsearch", init_val=1)
        return {}

    if cfg.get("lr_finder"):
        run_lr_finder(trainer, model, datamodule, cfg)
        return {}

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test"):
        # run validation on trained model and save output
        log.info("Starting validation!")
        model = model.eval()
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    return trainer.callback_metrics


def run_lr_finder(trainer, model, datamodule, cfg):
    # Create a tuner for the trainer
    tuner = Tuner(trainer)
    # run the lightning learning rate finder, this runs the model with different learning rates
    lr_finder = tuner.lr_find(model=model, datamodule=datamodule)

    fig = lr_finder.plot(suggest=True)
    output_fig = os.path.join(cfg.paths.output_dir, "loss_lr.png")
    fig.savefig(output_fig)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    if torch.cuda.device_count() > 0:
        log.info(f"Available GPU: {torch.cuda.get_device_name(0)}")
    else:
        log.info("No GPUs found")

    # train the model
    metric_dict = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
