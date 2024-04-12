import os
import pickle
import random

import numpy as np
import torch

from mtr.utils import common_utils
from utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def filter_empty_scenes(info: dict):
    """Filter a whole scene if it has objects we need to predict."""
    # tracks to predict could be reduced by filter info by type of dist
    num_interested_agents = info["tracks_to_predict"]["track_index"].__len__()
    if num_interested_agents == 0:
        return False

    return True


def filter_info_by_object_types(info: dict, valid_object_types: list):
    """Filter scene for desired object types, returns info with tracks to predict.

    Args:
        info (dict): sample
        valid_object_types (_type_, optional): list of desired object types. Defaults to None.
    """
    valid_mask = []
    for idx, _ in enumerate(info["tracks_to_predict"]["track_index"]):
        valid_mask.append(info["tracks_to_predict"]["object_type"][idx] in valid_object_types)

    assert len(info["tracks_to_predict"].keys()) == 3, f"{info['tracks_to_predict'].keys()}"
    # set track index to predict to
    info["tracks_to_predict"]["track_index"] = [
        track for valid, track in zip(valid_mask, info["tracks_to_predict"]["track_index"]) if valid
    ]
    info["tracks_to_predict"]["object_type"] = [
        track for valid, track in zip(valid_mask, info["tracks_to_predict"]["object_type"]) if valid
    ]
    info["tracks_to_predict"]["difficulty"] = [
        track for valid, track in zip(valid_mask, info["tracks_to_predict"]["difficulty"]) if valid
    ]

    return info


def filter_info_dist(info: dict, dist_threshold: int):
    """Filter objects based on distance to SDC. THIS MUST BE THE FIRST FILTER FUNCTION.

    Args:
        info (dict): sample
        dist_threshold (int): threshold distance to exclude targets in meters, -1 to not filter
    """
    info["initial_tracks_to_predict"] = info["tracks_to_predict"]["track_index"]

    if dist_threshold < 0:
        return info

    valid_mask = []
    for idx, _ in enumerate(info["tracks_to_predict"]["track_index"]):
        # get the distance to the target at last timestamp
        dist_target = np.linalg.norm(info["lidar_targets"][idx][-1, :2])
        valid_mask.append(dist_target < dist_threshold)

    assert len(info["tracks_to_predict"].keys()) == 3, f"{info['tracks_to_predict'].keys()}"
    # set track index to predict to
    info["tracks_to_predict"]["track_index"] = [
        track for valid, track in zip(valid_mask, info["tracks_to_predict"]["track_index"]) if valid
    ]
    info["tracks_to_predict"]["object_type"] = [
        track for valid, track in zip(valid_mask, info["tracks_to_predict"]["object_type"]) if valid
    ]
    info["tracks_to_predict"]["difficulty"] = [
        track for valid, track in zip(valid_mask, info["tracks_to_predict"]["difficulty"]) if valid
    ]

    return info


def filter_info_dropout_vehil(info: dict, dropout_percentage: float):
    """Randomly dropout vehicles with some percentage.

    Args:
        info (dict): sample
        dropout_percentage: percentage change for each vehicle object if to drop it
    """
    if dropout_percentage == 0:
        return info

    valid_mask = []
    for idx, _ in enumerate(info["tracks_to_predict"]["track_index"]):
        if info["tracks_to_predict"]["object_type"][idx] == "TYPE_VEHICLE" and random.random() < dropout_percentage:
            valid_mask.append(False)
        else:
            valid_mask.append(True)

    # set track index to predict to
    info["tracks_to_predict"]["track_index"] = [
        track for valid, track in zip(valid_mask, info["tracks_to_predict"]["track_index"]) if valid
    ]
    info["tracks_to_predict"]["object_type"] = [
        track for valid, track in zip(valid_mask, info["tracks_to_predict"]["object_type"]) if valid
    ]
    info["tracks_to_predict"]["difficulty"] = [
        track for valid, track in zip(valid_mask, info["tracks_to_predict"]["difficulty"]) if valid
    ]

    return info


def unpickle(data: dict):
    return pickle.loads(data["pkl"])


def get_data_url(dataset_cfg, training: bool):
    """Returns the pytorch webdataset url format.

    Args:
        dataset_cfg (dict): config settings of dataset used for path dir
        training (bool): if training
        logger (logger): logger object
    Returns:
        str: url of data shard in webdataset format
    """
    mode = "train" if training else "test"
    data_path = os.path.join(dataset_cfg.DATA_ROOT, dataset_cfg.SPLIT_DIR[mode])
    shards = os.listdir(data_path)

    def convert_int(x):
        try:
            return int(x)
        except ValueError:
            return 0

    max_shard = max(map(convert_int, [a.strip(".tar").strip("training_shard_") for a in shards]))
    if dataset_cfg.NUM_FILES[mode] > 0:
        # limit number of shards to use
        max_shard = min(max_shard, dataset_cfg.NUM_FILES[mode])
    to_shard_str = str(max_shard).rjust(5, "0")

    data_url = os.path.join(data_path, f"training_shard_{{00000..{to_shard_str}}}.tar")
    logger.info(f"Using data url {mode}: {data_url}")

    return data_url


def waymo_collate_batch(batch_list):
    """
    Args:
    batch_list:
        scenario_id: (num_center_objects)
        track_index_to_predict (num_center_objects):

        obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
        obj_trajs_mask (num_center_objects, num_objects, num_timestamps):
        map_polylines (num_center_objects, num_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
        map_polylines_mask (num_center_objects, num_polylines, num_points_each_polyline)

        obj_trajs_pos: (num_center_objects, num_objects, num_timestamps, 3)
        obj_trajs_last_pos: (num_center_objects, num_objects, 3)
        obj_types: (num_objects)
        obj_ids: (num_objects)

        center_objects_world: (num_center_objects, 10)  [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        center_objects_type: (num_center_objects)
        center_objects_id: (num_center_objects)

        obj_trajs_future_state (num_center_objects, num_objects, num_future_timestamps, 4): [x, y, vx, vy]
        obj_trajs_future_mask (num_center_objects, num_objects, num_future_timestamps):
        center_gt_trajs (num_center_objects, num_future_timestamps, 4): [x, y, vx, vy]
        center_gt_trajs_mask (num_center_objects, num_future_timestamps):
        center_gt_final_valid_idx (num_center_objects): the final valid timestamp in num_future_timestamps
    """
    batch_size = len(batch_list)
    key_to_list = {}
    for key in batch_list[0].keys():
        if key != "__key__":  # __key__ from webdataset gets added and should be ignored
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

    input_dict = {}
    for key, val_list in key_to_list.items():
        if key in ["obj_trajs_pos"]:
            # ignore as is not used
            continue

        if key in [
            "obj_trajs",
            "obj_trajs_mask",
            "map_polylines",
            "map_polylines_mask",
            "map_polylines_center",
            "obj_trajs_pos",
            "obj_trajs_last_pos",
            "obj_trajs_future_state",
            "obj_trajs_future_mask",
            "lidar_mask",
        ]:
            val_list = [torch.from_numpy(x) for x in val_list]
            input_dict[key] = common_utils.merge_batch_by_padding_2nd_dim(val_list)
        elif key in [
            "scenario_id",
            "obj_types",
            "obj_ids",
            "center_objects_type",
            "center_objects_id",
            "center_gt_trajs_src",
        ]:
            input_dict[key] = np.concatenate(val_list, axis=0)
        else:
            val_list = [torch.from_numpy(x) for x in val_list]
            input_dict[key] = torch.cat(val_list, dim=0)

    batch_sample_count = [len(x["track_index_to_predict"]) for x in batch_list]
    batch_dict = {"batch_size": batch_size, "input_dict": input_dict, "batch_sample_count": batch_sample_count}

    return batch_dict
