import os
from typing import Any, Optional

import polars as pl
import torch
import webdataset as wds
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from mtr.datasets.waymo import waymo_webdataset
from mtr.datasets.waymo.waymo_scenedata import WaymoSceneData
from utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class WebWaymoModule(LightningDataModule):
    """`LightningDataModule` for the Waymo dataset using webdataset."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize a `WebWaymoModule`.

        :param cfg: config
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size = cfg.batch_size_per_device
        self.previous_batch_size = cfg.batch_size_per_device

        self.world_size = 1

        self.nr_scenes_training = None
        self.nr_scenes_val = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        logger.info(f"Batch size per GPU: {self.cfg.batch_size_per_device}")

        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
        logger.info(f"Got world size: {self.world_size}, for length dataset")

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            self.data_train = self.create_webdataset_waymo(training=True)
            self.data_val = self.create_webdataset_waymo(training=False)

            self.nr_scenes_training = self.get_length_dataset(training=True)
            self.nr_scenes_val = self.get_length_dataset(training=False)

    def get_length_dataset(self, training: bool) -> int:
        """Returns the length of the dataset in number of scenes This is an estimate as we divide
        by the number of gpus and used shards. And distance filtering is not applied.

        Args:
            training (bool): if to get length training dataset or validation

        Returns:
            int: length dataset
        """
        mode = "train" if training else "test"
        info_path = os.path.join(self.cfg.DATA_ROOT, self.cfg.OBJECTS_FILE[mode])

        df_obj = pl.scan_parquet(info_path)
        nr_scenes_before = df_obj.select(pl.col("scenario_id").n_unique())

        # filter on number of file
        if self.cfg.NUM_FILES[mode] > 0:
            df_obj = df_obj.filter(pl.col("shard_id") < self.cfg.NUM_FILES[mode])

        # filter on objects
        df_obj = df_obj.filter(pl.col("object_type").is_in(self.cfg.OBJECT_TYPE))

        # filter on dist
        if self.cfg.get("FILTER_DIST_THRESHOLD") and self.cfg.get("FILTER_DIST_THRESHOLD") > 0:
            df_obj = df_obj.filter(pl.col("dist_sdc") < self.cfg.get("FILTER_DIST_THRESHOLD"))

        # count unique scenario ids over
        nr_scenes = df_obj.select(pl.col("scenario_id").n_unique()).collect().item()
        nr_scenes_before = nr_scenes_before.collect().item()

        logger.info(
            f"Nr scenes {mode} before: {nr_scenes_before} after filter: {nr_scenes}, percentage {nr_scenes/nr_scenes_before:.2%}"
        )

        # divide by world size so every process get his share, this will not be exact,
        # as one might get more due to imbalance in filtering
        nr_scenes_gpu = nr_scenes // self.world_size

        logger.info(
            f"Nr scenes {mode}: for {self.world_size} device(s): {nr_scenes_gpu}. Num expected iters: {nr_scenes_gpu//self.batch_size}"
        )

        return nr_scenes_gpu

    def create_webdataset_waymo(self, training: bool):
        def obj_filter_func(x):
            return waymo_webdataset.filter_info_by_object_types(x, self.cfg.OBJECT_TYPE)

        def dist_filter_func(x):
            return waymo_webdataset.filter_info_dist(x, dist_threshold=self.cfg.get("FILTER_DIST_THRESHOLD", -1))

        def random_dropout(x):
            if training:
                return waymo_webdataset.filter_info_dropout_vehil(
                    x, dropout_percentage=float(self.cfg.get("dropout_vehicle_percentage", 0))
                )
            return x

        scene_creator = WaymoSceneData(self.cfg, train=training)

        data_url = waymo_webdataset.get_data_url(self.cfg, training=training)
        dataset = (
            wds.WebDataset(data_url, nodesplitter=wds.split_by_node)
            .map(waymo_webdataset.unpickle)
            .map(dist_filter_func)  # dist filter must be first otherwise idx of target will not match
            .map(obj_filter_func)
            .map(random_dropout)
            .select(waymo_webdataset.filter_empty_scenes)
            .shuffle(400 if training else 0)  # do not shuffle during validation, 0 is no shuffle
            .map(scene_creator.create_scene_level_data)
            .batched(
                self.batch_size,
                collation_fn=waymo_webdataset.waymo_collate_batch,
                partial=not (training and self.cfg.DROP_LAST),  # allow partial in val or not drop last
            )
        )

        return dataset

    def make_dataloader(self, training: bool) -> DataLoader[Any]:
        if training:
            dataset = self.data_train
        else:
            dataset = self.data_val

        dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            # batch_size=self.batch_size,
            pin_memory=self.cfg.pin_memory,
            num_workers=self.cfg.NUM_WORKERS,
            shuffle=False,
            collate_fn=lambda x: x,  # default pytorch collate converts numpy to tensors
            # collate_fn=waymo_webdataset.waymo_collate_batch,
            # drop_last=drop_last,
            # timeout=0,
        )

        if training:
            # repeats the dataset twice and gives the nr scenes // batch size of samples
            # This is important to equalize the number of samples in DDP
            dataloader = dataloader.repeat(nepochs=2, nbatches=self.nr_scenes_training // self.batch_size)

        return dataloader

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.previous_batch_size != self.batch_size:
            # recreate dataset for batch size finder
            logger.info(f"Recreating datasets with new batch size {self.batch_size}")
            self.data_train = self.create_webdataset_waymo(training=True)
            self.data_val = self.create_webdataset_waymo(training=False)

            self.nr_scenes_training = self.get_length_dataset(training=True)
            self.nr_scenes_val = self.get_length_dataset(training=False)

            self.previous_batch_size = self.batch_size

        return self.make_dataloader(training=True)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        import time

        print(f"val loader, {self.batch_size}, {time.time()}")
        return self.make_dataloader(training=False)


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base="1.3", config_path="../../../configs", config_name="train.yaml")
    def test(cfg):
        dataset = WebWaymoModule(cfg.data)
        dataset.prepare_data()
        dataset.setup()

        counter = 0
        cum_batch_size = 0
        for batch in dataset.train_dataloader():
            counter += 1
            cum_batch_size += batch["batch_size"]

        print("Train iters:", counter, "total batch_size", cum_batch_size)

        counter = 0
        cum_batch_size = 0
        for batch in dataset.val_dataloader():
            counter += 1
            cum_batch_size += batch["batch_size"]

        print("Val iters:", counter, "total batch_size", cum_batch_size)

    test()
