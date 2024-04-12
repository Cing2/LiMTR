import copy
import math
import os
import pickle
import tarfile
from typing import Literal, Optional

import numpy as np
from PIL import Image, ImageDraw

type_color_map = {
    "TYPE_BROKEN_SINGLE_WHITE": "lightblue",
    "TYPE_SOLID_SINGLE_WHITE": "mediumslateblue",
    "TYPE_SOLID_SINGLE_YELLOW": "yellow",
    "TYPE_BROKEN_SINGLE_YELLOW": "lightyellow",
    "TYPE_PASSING_DOUBLE_YELLOW": "orange",
    "TYPE_SOLID_DOUBLE_YELLOW": "orange",
    "TYPE_BROKEN_DOUBLE_YELLOW": "yellow",
}


def draw_bound_box_car(image, state, color="grey"):
    # calculate bounding coordinates
    x, y, _, length, width, _, angle_in_radians, _, _, _ = state
    half_length = length / 2
    half_width = width / 2
    # print(half_length, half_width)
    cos_angle = math.cos(angle_in_radians)
    sin_angle = math.sin(angle_in_radians)
    # Calculate the coordinates of the four corners of the box
    x1 = x + half_length * cos_angle - half_width * sin_angle
    y1 = y + half_length * sin_angle + half_width * cos_angle
    x2 = x + half_length * cos_angle + half_width * sin_angle
    y2 = y + half_length * sin_angle - half_width * cos_angle
    x3 = x - half_length * cos_angle + half_width * sin_angle
    y3 = y - half_length * sin_angle - half_width * cos_angle
    x4 = x - half_length * cos_angle - half_width * sin_angle
    y4 = y - half_length * sin_angle + half_width * cos_angle

    # draw bounding box
    image.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=color)


def draw_lines(image, lines, color: str = "grey", include_end=False):
    # print(lines)
    points = [(point[0], point[1]) for point in lines]
    if include_end:
        points.append(points[0])
    image.line(points, fill=color, joint="curve", width=2)


def draw_road(image, map_infos):
    #  draw road edges
    for road_edge in map_infos["road_edge"]:
        idx_poly = road_edge["polyline_index"]
        if road_edge["type"] == "TYPE_ROAD_EDGE_BOUNDARY":
            color = "black"
        else:
            color = "green"
        draw_lines(image, map_infos["all_polylines"][idx_poly[0] : idx_poly[1]], color=color)

    # draw road lines
    for road_line in map_infos["road_line"]:
        idx_poly = road_line["polyline_index"]
        draw_lines(
            image, map_infos["all_polylines"][idx_poly[0] : idx_poly[1]], color=type_color_map[road_line["type"]]
        )

    for crosswalk in map_infos["crosswalk"]:
        idx_poly = crosswalk["polyline_index"]

        # draw_lines(image, map_infos['all_polylines'][idx_poly[0]:idx_poly[1]], color='purple', include_end=True)

    return image


def get_scale(scenario: dict):
    # scale the data along the size of the cars
    car_traj = scenario["track_infos"]["trajs"]
    min_traj = car_traj[car_traj[:, :, -1] > 0].min(0)
    max_traj = car_traj[car_traj[:, :, -1] > 0].max(0)
    scale = 1000 / (max_traj[0] - min_traj[0])

    return min_traj.copy(), scale


def scale_scene_data(scene_data, min_traj, scale):
    # scale obj trajs
    car_traj = scene_data["track_infos"]["trajs"]
    car_traj[car_traj[:, :, -1] > 0, :2] -= min_traj[:2]
    car_traj[:, :, :6] *= scale

    # scale map data
    map_infos = scene_data["map_infos"]
    map_infos["all_polylines"][:, :2] -= min_traj[:2]
    map_infos["all_polylines"][:, :3] *= scale

    return scene_data


def scale_trajs(trajs: np.ndarray, min_traj: np.ndarray, scale: float) -> np.ndarray:
    """Scale the trajs by subtracting min trajection and multiply by scale.

    Args:
        trajs (np.ndarray): trajectories to scale
        min_traj (np.ndarray): min traj to subtract
        scale (float): float to scale by

    Returns:
        np.ndarray: scaled trajectories
    """
    trajs[..., :2] -= min_traj[:2]
    trajs[..., :3] *= scale

    return trajs


def scale_scenario(scenario):
    min_traj, scale = get_scale(scenario)
    return scale_scene_data(scenario, min_traj, scale)


def draw_scenario(scenario: dict, time: int = 10) -> Image:
    # get size box
    size = scenario["track_infos"]["trajs"][scenario["track_infos"]["trajs"][:, :, -1] > 0].max(0)[:2]
    # print(size)
    image = Image.new("RGB", (int(size[0]), int(size[1])), "white")
    draw = ImageDraw.Draw(image)

    # draw road items
    draw_road(draw, scenario["map_infos"])

    # draw bounding box of cars
    for i, traj in enumerate(scenario["track_infos"]["trajs"]):
        if traj[0][-1] > 0:
            color = "grey"
            if i in scenario["tracks_to_predict"]["track_index"]:
                color = "blue"
            if i == scenario["sdc_track_index"]:
                color = "red"
            draw_bound_box_car(draw, traj[time], color=color)

    return image


def show_scenario(scenario, time: int = 11):
    scenario = scale_scenario(copy.deepcopy(scenario))
    image = draw_scenario(scenario, time)
    image.show()


def draw_multiple_scenarios(path_tar: str, num: int):
    training_shard = tarfile.open(path_tar)
    files = training_shard.getmembers()

    output_dir = os.path.join("..", "exploration", "plots", "scenarios")
    os.makedirs(output_dir, exist_ok=True)

    for file in files[:num]:
        data = training_shard.extractfile(file)
        scenario = pickle.load(data)
        scenario = scale_scenario(scenario)
        image = draw_scenario(scenario)
        output_file = os.path.join(output_dir, scenario["scenario_id"] + ".png")
        image.save(output_file)


def draw_timeline_scenario(tar_file: str, draw_member: int = 0):
    """Draw the timeline of the scenario for the draw_member nr in the tarfile This draws the
    scenario for 10 timestamps.

    Args:
        tar_file (str): path to tar file
        draw_member (int, optional): number in tarfile to draw. Defaults to 0.
    """
    training_shard = tarfile.open(tar_file)
    files = training_shard.getmembers()

    output_dir = os.path.join("..", "plots", "scenarios", "timelines")
    os.makedirs(output_dir, exist_ok=True)

    data = training_shard.extractfile(files[draw_member])
    scenario = pickle.load(data)
    scenario = scale_scenario(scenario)

    for t in range(10):
        image = draw_scenario(scenario, time=t * 10)
        output_file = os.path.join(output_dir, f'{scenario["scenario_id"]}_t{t}.png')
        image.save(output_file)


def make_animation(scenario):
    # resize all coordinates
    scenario = scale_scenario(copy.deepcopy(scenario))
    frames = [draw_scenario(scenario, time=t) for t in range(len(scenario["timestamps_seconds"]))]
    output_path = os.path.join(
        "..", "exploration", "plots", "scenarios", "animations", f"scene_{scenario['scenario_id']}.gif"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frames[0].save(
        output_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=scenario["timestamps_seconds"][-1],
        loop=0,
    )


def get_scenario_file(shard: tarfile.TarFile, scenario_id: str) -> Optional[dict]:
    """Check a given shard if the scenario is in it.

    Args:
        training_shard (tarfile.TarFile): tarfile object
        scenario_id (str): id of scenario to retrieve

    Returns:
        Optional[dict]: scenario data
    """
    for file in shard.getmembers():
        if scenario_id in file.name:
            data = shard.extractfile(file)
            return pickle.load(data)
    else:
        # print("Scenario not in files")
        return None


def get_scenario_shards(dir_shards: str, scenario_id: str) -> Optional[dict]:
    """Check all shards in dir to check scenario and if found return it.

    Args:
        dir_shards (str): dir to all shard
        scenario_id (str): scenario to find

    Returns:
        Optional[dict]: dict of scenario info
    """
    scene_data = None
    for name_shard in os.listdir(dir_shards):
        if name_shard.endswith("tar"):
            shard = tarfile.open(os.path.join(dir_shards, name_shard))
            scene_data = get_scenario_file(shard, scenario_id)
            if scene_data:
                break

    return scene_data


def get_nearest_traj(pred_trajs: np.ndarray, gt_traj: np.ndarray, gt_valid: np.ndarray) -> np.ndarray:
    """Get the closest trajectory to given gt traj.

    Args:
        pred_trajs (np.ndarray): pred trajs (6, T, 3)
        gt_traj (np.ndarray): gt traj (T, 10)
        gt_valid (np.ndarray): (T)

    Returns:
        np.ndarray: _description_
    """
    # get nearest trajectories of the 6 predicted, based on average dist gt
    distance = np.linalg.norm(pred_trajs - gt_traj[None, :, :2], axis=-1)
    distance = (distance * gt_valid[None]).sum(-1)
    nearest_mode_idx = np.argmin(distance, axis=-1)

    nearest_trajs = pred_trajs[nearest_mode_idx]
    return nearest_trajs


def draw_paths_in_scene(image: ImageDraw.Draw, paths: np.ndarray, colors: list):
    """Draw paths in a scene.

    Args:
        image (ImageDraw.Draw): pillow image to draw on
        paths (np.ndarray): paths to draw of size (B, T, 2)
        colors (list): color for each path (B)
    """
    point_size = 2
    for path, color in zip(paths, colors):
        points = [
            (point[0] - point_size, point[1] - point_size, point[0] + point_size, point[1] + point_size)
            for point in path
        ]
        # draw a ellipse per path
        for point in points:
            image.ellipse(point, fill=color)


def scale_scene_trajs(pred_dict, scene_data, draw_path: Literal["nearest", "all"]):
    # scale data to fit in size of image
    min_traj, scale = get_scale(scene_data)
    scene_data = scale_scene_data(scene_data, min_traj, scale)
    draw_trajs, colors = get_draw_trajs(pred_dict, draw_path)
    draw_trajs = scale_trajs(draw_trajs, min_traj, scale)

    return scene_data, draw_trajs, colors


def draw_scenario_with_preds(pred_dict: dict, draw_path: Literal["nearest", "all"], timestamp=10):
    # check if scenario in tarfile
    dir_shards = os.path.join("..", "data", "waymo", "processed_scenarios_validation_web")
    scene_data = get_scenario_shards(dir_shards, scenario_id=pred_dict["scenario_id"])
    if not scene_data:
        # print("could not find scenario of prediction")
        return False

    scene_data, draw_trajs, colors = scale_scene_trajs(pred_dict, scene_data, draw_path=draw_path)

    # draw scenario and paths
    image = draw_scenario(scene_data, time=timestamp)
    draw = ImageDraw.Draw(image)
    draw_paths_in_scene(draw, draw_trajs, colors)

    return image


def get_draw_trajs(pred_dict, draw_path: Literal["nearest", "all"]):
    # get values pred
    pred_trajs = pred_dict["pred_trajs"]
    gt_trajs = pred_dict["gt_trajs"]
    gt_trajs = gt_trajs[11:]  # only need future
    # object_types = pred_dict['object_type']
    gt_valid = gt_trajs[..., -1] == 1

    if draw_path == "nearest":
        nearest_traj = get_nearest_traj(pred_trajs, gt_trajs, gt_valid)
        draw_trajs = np.stack([nearest_traj, gt_trajs[:, :2]])
        colors = ["orange", "green"]
    else:
        draw_trajs = np.concatenate([gt_trajs.reshape(1, *gt_trajs.shape)[..., :2], pred_trajs])
        colors = ["green"] + ["#0D3B66", "#0290fc", "#fc02ef", "#02c9d3", "#fc0213", "#fc7b02"]

    return draw_trajs, colors


def make_animation_with_preds(pred_dict: dict, draw_path: Literal["nearest", "all"] = "nearest"):
    # check if scenario in tarfile
    dir_shards = os.path.join("..", "data", "waymo", "processed_scenarios_validation_web")
    scene_data = get_scenario_shards(dir_shards, scenario_id=pred_dict["scenario_id"])
    if not scene_data:
        print("could not find scenario of prediction")
        return

    scene_data, draw_trajs, traj_colors = scale_scene_trajs(pred_dict, scene_data, draw_path=draw_path)

    # draw scenario at multiple timestamps
    frames = []
    for time in range(11):
        # draw scenario and paths
        image = draw_scenario(scene_data, time=time)
        draw = ImageDraw.Draw(image)
        draw_paths_in_scene(draw, draw_trajs, traj_colors)
        frames.append(image)

    output_path = os.path.join(
        "..", "exploration", "plots", "scenarios", "animations", "preds", f"scene_{pred_dict['scenario_id']}.gif"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frames[0].save(
        output_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=100,
        loop=0,
    )
    return frames


if __name__ == "__main__":
    training_shard = tarfile.open("../data/waymo/processed_scenario_validation_web/training_shard_00000.tar")
    files = training_shard.getmembers()
    data = training_shard.extractfile(files[0])
    scene_1 = pickle.load(data)

    scenario_data = get_scenario_file(training_shard, "7a4dc4eb40f93323")
    draw_timeline_scenario("../data/waymo/processed_scenarios_training_web/training_shard_00000.tar", draw_member=1)
