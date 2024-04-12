import logging
import os
import tarfile
from math import floor

import torch

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import glob
import multiprocessing
import pickle
import time
from typing import List, Tuple

import numpy as np
import polars as pl
import tensorflow as tf
import torch.multiprocessing as mp
import webdataset as wds
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import womd_lidar_utils

from mtr.datasets.waymo.lidar_decompression import decompress_lidar_pointcloud
from mtr.datasets.waymo.waymo_types import (
    lane_type,
    object_type,
    polyline_type,
    road_edge_type,
    road_line_type,
    signal_state,
)


def decode_tracks_from_proto(tracks):
    track_infos = {
        "object_id": [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
        "object_type": [],
        "trajs": [],
    }
    for cur_data in tracks:  # number of objects
        cur_traj = [
            np.array(
                [
                    x.center_x,
                    x.center_y,
                    x.center_z,
                    x.length,
                    x.width,
                    x.height,
                    x.heading,
                    x.velocity_x,
                    x.velocity_y,
                    x.valid,
                ],
                dtype=np.float32,
            )
            for x in cur_data.states
        ]
        cur_traj = np.stack(cur_traj, axis=0)  # (num_timestamp, 10)

        track_infos["object_id"].append(cur_data.id)
        track_infos["object_type"].append(object_type[cur_data.object_type])
        track_infos["trajs"].append(cur_traj)

    track_infos["trajs"] = np.stack(track_infos["trajs"], axis=0)  # (num_objects, num_timestamp, 9)
    return track_infos


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir


def decode_map_features_from_proto(map_features):
    map_infos = {"lane": [], "road_line": [], "road_edge": [], "stop_sign": [], "crosswalk": [], "speed_bump": []}
    polylines = []

    point_cnt = 0
    for cur_data in map_features:
        cur_info = {"id": cur_data.id}

        if cur_data.lane.ByteSize() > 0:
            cur_info["speed_limit_mph"] = cur_data.lane.speed_limit_mph
            cur_info["type"] = lane_type[
                cur_data.lane.type
            ]  # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane

            cur_info["interpolating"] = cur_data.lane.interpolating
            cur_info["entry_lanes"] = list(cur_data.lane.entry_lanes)
            cur_info["exit_lanes"] = list(cur_data.lane.exit_lanes)

            cur_info["left_boundary"] = [
                {
                    "start_index": x.lane_start_index,
                    "end_index": x.lane_end_index,
                    "feature_id": x.boundary_feature_id,
                    "boundary_type": x.boundary_type,  # roadline type
                }
                for x in cur_data.lane.left_boundaries
            ]
            cur_info["right_boundary"] = [
                {
                    "start_index": x.lane_start_index,
                    "end_index": x.lane_end_index,
                    "feature_id": x.boundary_feature_id,
                    "boundary_type": road_line_type[x.boundary_type],  # roadline type
                }
                for x in cur_data.lane.right_boundaries
            ]

            global_type = polyline_type[cur_info["type"]]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.lane.polyline], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["lane"].append(cur_info)

        elif cur_data.road_line.ByteSize() > 0:
            cur_info["type"] = road_line_type[cur_data.road_line.type]

            global_type = polyline_type[cur_info["type"]]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_line.polyline], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["road_line"].append(cur_info)

        elif cur_data.road_edge.ByteSize() > 0:
            cur_info["type"] = road_edge_type[cur_data.road_edge.type]

            global_type = polyline_type[cur_info["type"]]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_edge.polyline], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["road_edge"].append(cur_info)

        elif cur_data.stop_sign.ByteSize() > 0:
            cur_info["lane_ids"] = list(cur_data.stop_sign.lane)
            point = cur_data.stop_sign.position
            cur_info["position"] = np.array([point.x, point.y, point.z])

            global_type = polyline_type["TYPE_STOP_SIGN"]
            cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type]).reshape(1, 7)

            map_infos["stop_sign"].append(cur_info)
        elif cur_data.crosswalk.ByteSize() > 0:
            global_type = polyline_type["TYPE_CROSSWALK"]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.crosswalk.polygon], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["crosswalk"].append(cur_info)

        elif cur_data.speed_bump.ByteSize() > 0:
            global_type = polyline_type["TYPE_SPEED_BUMP"]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.speed_bump.polygon], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["speed_bump"].append(cur_info)
        elif cur_data.driveway.ByteSize() > 0:
            # skip for now
            pass
        else:
            print(cur_data)
            raise ValueError

        polylines.append(cur_polyline)
        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
        point_cnt += len(cur_polyline)

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except Exception:
        polylines = np.zeros((0, 7), dtype=np.float32)
        print("Empty polylines: ")
    map_infos["all_polylines"] = polylines
    return map_infos


def decode_dynamic_map_states_from_proto(dynamic_map_states):
    dynamic_map_infos = {"lane_id": [], "state": [], "stop_point": []}
    for cur_data in dynamic_map_states:  # (num_timestamp)
        lane_id, state, stop_point = [], [], []
        for cur_signal in cur_data.lane_states:  # (num_observed_signals)
            lane_id.append(cur_signal.lane)
            state.append(signal_state[cur_signal.state])
            stop_point.append([cur_signal.stop_point.x, cur_signal.stop_point.y, cur_signal.stop_point.z])

        dynamic_map_infos["lane_id"].append(np.array([lane_id]))
        dynamic_map_infos["state"].append(np.array([state]))
        dynamic_map_infos["stop_point"].append(np.array([stop_point]))

    return dynamic_map_infos


def split_numbers_tfrecord(path_file: str) -> Tuple[str, str]:
    """Return the numbers of the tfrecord.

    Args:
        path_file (str): path to tfrecord

    Returns:
        Tuple[str, str]: numbers of number and total
    """
    splits = path_file.split(".")[-1].split("-")
    number, total = splits[1], splits[-1]
    return number, total


def get_target_vehicle_frame(track_infos, tracks_to_predict, compressed_frame_laser_data):
    """Calculate the targets position in the vehicle frame of view.

    Args:
        track_infos (Dict): trajs info
        targets_ids (list): ids of targets to retrieve
        compressed_frame_laser_data (int): scenario lidar data to get frame transform matrices from

    Returns:
        list of np: targets relative position for first 1 frames
    """
    # get rotation and translation matrices
    transformation_matrices = []
    for frame in compressed_frame_laser_data:
        transformation_matrices.append(np.array(frame.pose.transform).reshape(4, 4))
    world_to_vehicle = np.linalg.inv(transformation_matrices)
    rotation_matrices = world_to_vehicle[:, :3, :3]
    translations = world_to_vehicle[:, :3, 3]

    # get targets scenario
    target_tracks_xyz = []

    for tp_track in tracks_to_predict:
        target_points = track_infos["trajs"][tp_track.track_index, :11].copy()
        # change points from world to vehicle frame by applying the rotation matrices and translations
        target_points[:, :3] = (
            np.matmul(rotation_matrices, target_points[:, :3, np.newaxis]).squeeze(axis=2) + translations
        )
        # calculate difference in heading and update targets
        roll = np.arctan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])
        target_points[:, 6] += roll

        target_tracks_xyz.append(target_points)

    return target_tracks_xyz


def get_subset_points_target(
    frame_points_xyz, frame_points_feature, targets_xyz, threshold: int = 5, max_points: int = 2000
):
    """Return the subset of points within threshold range of targets.

    Args:
        frame_points_xyz (dict of np): points xyz coords
        frame_points_feature (dict of np): points features
        targets (np): 2d np array of targets xyz
        threshold (int, optional): threshold to a target in meters. Defaults to 5.
    """
    # calculate distance lidar points and targets
    close_lidar_xyz = []
    close_lidar_feature = []
    for target in targets_xyz:
        targets_close_xyz = []
        targets_lidar_feature = []
        for i in range(len(frame_points_xyz)):
            # if target is invalid skip
            if target[i, -1] < 1:
                targets_close_xyz.append(np.zeros(0))
                targets_lidar_feature.append(np.zeros(0))
                continue
            # get points that are in the bounding box of a target
            # for each time frame
            dists = np.linalg.norm(frame_points_xyz[i] - target[i, :3], axis=-1)
            # get points within threshold of target
            idx = np.where(dists < threshold)[0]
            # get random subset of close points
            np.random.shuffle(idx)
            targets_close_xyz.append(frame_points_xyz[i][idx[:max_points]])
            targets_lidar_feature.append(frame_points_feature[i][idx[:max_points]])

        close_lidar_xyz.append(targets_close_xyz)
        close_lidar_feature.append(targets_lidar_feature)

    return close_lidar_xyz


def points_within_bb(points: np.ndarray, target: np.ndarray, error_margin: float):
    """Checks if points are inside the bounding box specified by its center, dimensions, and angle
    of rotation.

    Parameters:
        points (numpy.ndarray): Array of points, each row representing a point in the format [x, y, z].
        target (numpy.ndarray): info of target, position and size bounding box
        error_margin (float): small error margin to allow around target in meters

    Returns:
        numpy.ndarray: Boolean array indicating whether each point is inside the bounding box.
    """
    # Calculate the rotation matrix
    rotation_matrix = np.array(
        [[np.cos(target[6]), -np.sin(target[6]), 0], [np.sin(target[6]), np.cos(target[6]), 0], [0, 0, 1]],
        dtype=target.dtype,
    )

    # Translate points to be relative to the bounding box center
    translated_points = points.copy() - target[:3]

    # Rotate the points
    rotated_points = np.dot(translated_points, rotation_matrix)
    # rotated_points = np.hstack((rotated_points, translated_points[:, 2:]))

    # Check if points are within the half-length, half-width, and half-height of the bounding box
    in_x = np.abs(rotated_points[:, 0]) <= target[3] / 2 + error_margin
    in_y = np.abs(rotated_points[:, 1]) <= target[4] / 2 + error_margin
    in_z = np.abs(rotated_points[:, 2]) <= target[5] / 2 + error_margin

    # Check if the points are inside the bounding box in all dimensions
    inside_bbox = np.logical_and.reduce((in_x, in_y, in_z))

    return inside_bbox


def get_subset_points_target_boundingbox(frame_points_xyz, frame_points_feature, targets_xyz, max_points: int = 2000):
    """Return the subset of points within threshold range of targets.

    Args:
        frame_points_xyz (dict of np): points xyz coords
        frame_points_feature (dict of np): points features
        targets (np): 2d np array of targets xyz
    """
    # calculate distance lidar points and targets
    close_lidar_xyz = []
    close_lidar_feature = []
    for target in targets_xyz:
        targets_close_xyz = []
        targets_lidar_feature = []
        for i in range(len(frame_points_feature)):
            # if target is invalid skip
            if target[i, -1] < 1:
                targets_close_xyz.append(np.zeros(0))
                targets_lidar_feature.append(np.zeros(0))
                continue
            # get points that are in the bounding box of a target
            idx_points_bb = points_within_bb(frame_points_xyz[i], target[i], error_margin=0.15)
            # get random subset of close points
            idx = np.where(idx_points_bb)[0]
            np.random.shuffle(idx)
            targets_close_xyz.append(frame_points_xyz[i][idx[:max_points]])
            targets_lidar_feature.append(frame_points_feature[i][idx[:max_points]])

        close_lidar_xyz.append(targets_close_xyz)
        close_lidar_feature.append(targets_lidar_feature)

    return close_lidar_xyz, close_lidar_feature


def load_scenario_data(tfrecord_file: str) -> scenario_pb2.Scenario:
    """Load a scenario proto from a tfrecord dataset file."""
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type="")
    data = next(iter(dataset))
    # print(type(data))
    return scenario_pb2.Scenario.FromString(data.numpy())


def decode_lidar_point_cloud(data_file: str, scenario, track_infos):
    # add lidar info to scene
    dir_original = os.path.dirname(data_file)
    path_lidar = os.path.join(
        dir_original, "..", "..", "lidar", os.path.split(dir_original)[-1], f"{scenario.scenario_id}.tfrecord"
    )
    womd_lidar_scenario = load_scenario_data(path_lidar)
    scenario_augmented = womd_lidar_utils.augment_womd_scenario_with_lidar_points(scenario, womd_lidar_scenario)

    # decompress lidar to point cloud
    frame_points_xyz, frame_points_feature = decompress_lidar_pointcloud(scenario_augmented)

    # get relative targets
    targets_xyz = get_target_vehicle_frame(
        track_infos, scenario_augmented.tracks_to_predict, scenario_augmented.compressed_frame_laser_data
    )

    # get subset points
    # close_points = get_subset_points_target(frame_points_xyz, frame_points_feature, targets_xyz)
    # get subset points in bounding box of target
    lidar_points_bb, lidar_features_bb = get_subset_points_target_boundingbox(
        frame_points_xyz, frame_points_feature, targets_xyz, max_points=2048
    )

    return targets_xyz, lidar_points_bb, lidar_features_bb


def process_single_waymo_scenario(data, data_file: str, only_do_info: bool = False):
    """Process a single waymo scenario to the data we need.

    Args:
        data (bytes): scenario in bytes
        data_file (str): path of tfrecord
        only_do_info (bool, optional): if to only to make info and not the safe. Defaults to False.

    Returns:
        dict: infos and save_info
    """
    info = {}
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(bytearray(data.numpy()))

    info["scenario_id"] = scenario.scenario_id
    info["timestamps_seconds"] = list(scenario.timestamps_seconds)  # list of int of shape (91)
    info["current_time_index"] = scenario.current_time_index  # int, 10
    info["sdc_track_index"] = scenario.sdc_track_index  # int
    info["objects_of_interest"] = list(scenario.objects_of_interest)  # list, could be empty list

    info["tracks_to_predict"] = {
        "track_index": [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
        "difficulty": [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict],
    }  # for training: suggestion of objects to train on, for val/test: need to be predicted

    track_infos = decode_tracks_from_proto(scenario.tracks)
    info["tracks_to_predict"]["object_type"] = [
        track_infos["object_type"][cur_idx] for cur_idx in info["tracks_to_predict"]["track_index"]
    ]
    # option to only make pkl info file
    if only_do_info:
        return info

    # get lidar information
    lidar_targets, lidar_points_bb, lidar_feature_bb = decode_lidar_point_cloud(data_file, scenario, track_infos)

    # decode map related data
    map_infos = decode_map_features_from_proto(scenario.map_features)
    dynamic_map_infos = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)

    save_infos = {
        "track_infos": track_infos,
        "dynamic_map_infos": dynamic_map_infos,
        "map_infos": map_infos,
        "lidar_features_bb": lidar_feature_bb,
        "lidar_points_bb": lidar_points_bb,
        "lidar_targets": lidar_targets,  # save the targets to be able to change to view later
    }
    save_infos.update(info)

    return info, save_infos


def process_waymo_scenario_multi(data_path: str, output_path: str, pool: multiprocessing.Pool, limit: int, redo: bool):
    """Do preprocessing on Waymo TFRecord files to create a tar archive for each.

    Args:
        data_path (str): input path to look for tfrecord files
        output_path (str): output path to put shard
        num_workers (int): num of workers in multiprocessing
        limit (int): limit number of files to preprocess
        redo (bool): if to redo existing output files

    Returns:
        list: output basic info on samples to safe separately
    """
    from functools import partial

    os.makedirs(output_path, exist_ok=True)

    src_files = glob.glob(os.path.join(data_path, "*.tfrecord*"))
    src_files.sort()

    data_infos = []
    if limit > 0:
        src_files = src_files[:limit]
    for data_file in tqdm(src_files):
        number = split_numbers_tfrecord(data_file)[0]
        shard_path = os.path.join(output_path, f"training_shard_{number}.tar")

        # check if file is already process if so skip
        if not redo and os.path.exists(shard_path):
            print(f"Shard {shard_path} already exists, skipping")
            continue

        # preprocess .tfrecord into shard
        dataset = tf.data.TFRecordDataset(data_file, compression_type="")
        tar_sink = wds.TarWriter(shard_path)

        func = partial(process_single_waymo_scenario, data_file=data_file)
        # run processes and continouously write results to shard
        for info, save_infos in tqdm(pool.imap_unordered(func, dataset)):
            data_infos.append(info)
            tar_sink.write({"__key__": info["scenario_id"], "pkl": pickle.dumps(save_infos)})

    return data_infos


def make_only_infos_file(data_path: str, output_path: str, name_outputfile: str, limit: int):
    src_files = glob.glob(os.path.join(data_path, "*.tfrecord*"))
    src_files.sort()

    data_infos = []
    if limit > 0:
        src_files = src_files[:limit]
    for data_file in tqdm(src_files):
        dataset = tf.data.TFRecordDataset(data_file, compression_type="")

        # gather infos from file
        data_infos.extend([process_single_waymo_scenario(scene, data_file, only_do_info=True) for scene in dataset])

    # write output to pickle
    output_file = os.path.join(output_path, name_outputfile)
    with open(output_file, "wb") as f:
        pickle.dump(data_infos, f)
    print(f"Done with {output_file}")


def create_only_info_files(raw_data_path, output_path, limit: int = -1, subset: str = "all"):
    print("Making the infos.pkl files")
    if subset != "val":
        make_only_infos_file(
            data_path=os.path.join(raw_data_path, "training"),
            output_path=output_path,
            name_outputfile="processed_scenarios_training_infos.pkl",
            limit=limit,
        )
    if subset != "train":
        make_only_infos_file(
            data_path=os.path.join(raw_data_path, "validation"),
            output_path=output_path,
            name_outputfile="processed_scenarios_val_infos.pkl",
            limit=limit,
        )


def get_stats_shard(shard_file: str) -> List[list]:
    shard = tarfile.open(shard_file, "r")

    # get shard number
    shard_text = shard_file.split("/")[-1].rstrip(".tar").lstrip("training_shard_")
    try:
        shard_id = int(shard_text)
    except ValueError:
        shard_id = -1
        print(f"Could not get shard id {shard_file}")

    objects = []

    for file in shard:
        # unpickle
        info = pickle.load(shard.extractfile(file))

        # gather infos objects from file
        for i, track_to_predict in enumerate(info["tracks_to_predict"]["track_index"]):
            num_lidar_points = 0
            if len(info["lidar_points_bb"][i]) > 0:
                num_lidar_points = info["lidar_points_bb"][i][-1].shape[0]
            new_object = [
                shard_id,
                info["scenario_id"],
                info["track_infos"]["object_id"][track_to_predict],
                info["track_infos"]["object_type"][track_to_predict],
                np.linalg.norm(info["lidar_targets"][i][-1, :2]),  # distance to SDC at t=0
                num_lidar_points,  # nr of lidar points at t=0
            ]

            objects.append(new_object)

    return objects


def create_sampling_file(data_path: str, output_file: str, pool: multiprocessing.Pool):
    """Create a dataset of all intererested agents need to be predicted, with information used for filtering
    This includes: file nr, scenario id, object type and dist to SDC

    Args:
        data_path (str): path to shards to process
        output_file (str): path to write output to
    """
    # go over every shard and pickle file to
    src_files = glob.glob(os.path.join(data_path, "*.tar"))
    src_files.sort()

    objects = []
    for output in tqdm(pool.imap_unordered(get_stats_shard, src_files)):
        objects.extend(output)

    # write output to parquet
    column_names = ["shard_id", "scenario_id", "object_id", "object_type", "dist_sdc", "nr_points"]
    df = pl.DataFrame(objects, schema=column_names)
    df.write_parquet(output_file, compression="lz4")

    print(f"Done with {output_file}")


def create_stats_files(output_path: str, num_workers: int, subset: str = "all"):
    """Create statistics file for get nr samples in dataset.

    Args:
        output_path (str): output path of previous run to take shards from
        subset (str, optional): if to run subset of training for validation. Defaults to "all".
    """
    print("Making statistics files")
    pool = mp.Pool(processes=num_workers)

    if subset != "val":
        create_sampling_file(
            data_path=os.path.join(output_path, "processed_scenarios_training_web"),
            output_file=os.path.join(output_path, "training_samples.parquet"),
            pool=pool,
        )
    if subset != "train":
        create_sampling_file(
            data_path=os.path.join(output_path, "processed_scenarios_validation_web"),
            output_file=os.path.join(output_path, "validation_samples.parquet"),
            pool=pool,
        )


def set_num_threads(num_threads: int):
    """Define the num threads used in current sub-processes.

    Args:
        num_threads (int): number of threads to use
    """
    torch.set_num_threads(num_threads)


def create_infos_from_protos(
    raw_data_path, output_path, num_workers: int, limit: int = -1, redo: bool = False, subset: str = "all"
):
    """Run the preprocessing on the training and validation sets.

    Args:
        raw_data_path (str): path to raw data
        output_path (str): path to save output
        num_workers (int): num workers
        limit (int, optional): How many files to do,  -1 to run on all. Defaults to -1.
        redo (bool, optional): If to redo files. Defaults to False.
        subset (str, optional): which subset to do. Defaults to "all".
    """
    sub_processes = floor(os.cpu_count() / num_workers)
    print(f"Subprocesses per workers: {sub_processes}")
    pool = mp.Pool(processes=num_workers, initializer=set_num_threads, initargs=(sub_processes,))

    if subset != "val":
        train_infos = process_waymo_scenario_multi(
            data_path=os.path.join(raw_data_path, "training"),
            output_path=os.path.join(output_path, "processed_scenarios_training_web"),
            pool=pool,
            limit=limit,
            redo=redo,
        )
        train_filename = os.path.join(output_path, "processed_scenarios_training_infos.pkl")
        with open(train_filename, "wb") as f:
            pickle.dump(train_infos, f)
        print("----------------Waymo info train file is saved to %s----------------" % train_filename)

    if subset != "train":
        val_infos = process_waymo_scenario_multi(
            data_path=os.path.join(raw_data_path, "validation"),
            output_path=os.path.join(output_path, "processed_scenarios_validation_web"),
            pool=pool,
            limit=limit,
            redo=redo,
        )
        pool.close()
        val_filename = os.path.join(output_path, "processed_scenarios_val_infos.pkl")
        with open(val_filename, "wb") as f:
            pickle.dump(val_infos, f)
        print("----------------Waymo info val file is saved to %s----------------" % val_filename)


def test(data_path: str):
    start_time = time.perf_counter()
    src_files = glob.glob(os.path.join(data_path, "training", "*.tfrecord*"))
    src_files.sort()

    test_dir = os.path.join(data_path, "..", "test_output")
    os.makedirs(test_dir, exist_ok=True)

    dataset = tf.data.TFRecordDataset(src_files[0], compression_type="")
    infos = []
    for i, data in tqdm(enumerate(dataset)):
        if i > 10:
            break
        info, saving_info = process_single_waymo_scenario(data, src_files[0])

        # save output
        infos.append(info)
        info_path = os.path.join(test_dir, info["scenario_id"] + ".pkl")
        with open(info_path, "wb") as fp:
            pickle.dump(saving_info, fp)

    with open(os.path.join(test_dir, "training_infos.pkl"), "wb") as fp:
        pickle.dump(infos, fp)

    print("Time run", time.perf_counter() - start_time)


def count_scenarios_file(path_file):
    dataset = tf.data.TFRecordDataset(path_file, compression_type="")
    count = 0
    for item in dataset:
        count += 1

    print("Number of scenarios files", count)


def main():
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("-n", "--num_workers", type=int, help="number of workers to use, default is cpu_count")
    parser.add_argument("-l", "--limit", type=int, help="Limit number of files to process")
    parser.add_argument(
        "--redo",
        action="store_true",
        default=False,
        help="Redo preprocessing if file exists, default is False meaining skipping existing output files",
    )
    parser.add_argument(
        "--info_pickle", required=False, action="store_true", default=False, help="To run the create only infos pickle"
    )
    parser.add_argument(
        "--stats", required=False, action="store_true", default=False, help="Run create of statistics file from pickle"
    )
    parser.add_argument(
        "--subset", required=False, default="all", choices=["all", "train", "val"], help="Specify a subset to "
    )

    parser.add_argument(
        "--test", required=False, action="store_true", default=False, help="To run a test of a single scenario"
    )

    args = parser.parse_args()

    if args.num_workers is None:
        num_workers = os.cpu_count()
    else:
        num_workers = int(args.num_workers)
    print("Num of workers ", num_workers)

    if args.test:
        test(args.input_path)
    elif args.info_pickle:
        create_only_info_files(
            raw_data_path=args.input_path, output_path=args.output_path, limit=args.limit, subset=args.subset
        )
    elif args.stats:
        create_stats_files(output_path=args.output_path, num_workers=num_workers, subset=args.subset)
    else:
        create_infos_from_protos(
            raw_data_path=args.input_path,
            output_path=args.output_path,
            num_workers=num_workers,
            limit=args.limit,
            redo=args.redo,
            subset=args.subset,
        )


if __name__ == "__main__":
    main()

    # count_scenarios_file('data/waymo/scenario/training/training.tfrecord-00000-of-01000')
