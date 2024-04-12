from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt


def heading_to_rotation(angles: np.ndarray) -> np.ndarray:
    """Create rotation matrixes based on the given angles.

    Args:
        angles (np.ndarrayd): 1d array of angles in radia

    Returns:
        np.ndarray: 3d array of N - 2D rotation matrixes
    """

    cosa = np.cos(angles)
    sina = np.sin(angles)
    rot_matrix = np.stack((cosa, sina, -sina, cosa), axis=1).reshape(-1, 2, 2)

    return rot_matrix


def evaluation_minFRDE(pred_dicts):
    # extract preds as numpy arrays
    pred_trajs = np.stack([preds["pred_trajs"] for preds in pred_dicts], axis=0)
    gt_trajs = np.stack([preds["gt_trajs"] for preds in pred_dicts], axis=0)
    gt_trajs = gt_trajs[:, 11:]  # only need future
    object_types = np.stack([preds["object_type"] for preds in pred_dicts], axis=0)
    gt_valid = gt_trajs[..., -1] == 1

    return calculate_minFRDE(pred_trajs, gt_trajs, object_types, gt_valid)


def calculate_minFRDE(pred_trajs: np.ndarray, gt_trajs: np.ndarray, object_types: np.ndarray, gt_valid: np.ndarray):
    # get nearest trajectories of the 6 predicted, based on average dist gt
    distance = np.linalg.norm(pred_trajs - gt_trajs[:, None, :, :2], axis=-1)
    distance = (distance * gt_valid[:, None]).sum(-1)
    nearest_mode_idx = np.argmin(distance, axis=-1)

    nearest_mode_bs_idxs = np.arange(pred_trajs.shape[0])
    nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idx]

    # rotate error vector on GT heading of final position
    res_trajs = gt_trajs[..., :2] - nearest_trajs  # (batch_size, num_timestamps, 2)
    evaluate_times = np.array([29, 49, 79])  # evaluate on times 3, 5, and 8 seconds
    res_trajs = res_trajs[:, evaluate_times]
    # set invalid gt to 0
    res_trajs[~gt_valid[:, evaluate_times]] = 0
    # rotate the distance vector to align with the gt car
    rot_matrix = heading_to_rotation(-gt_trajs[:, evaluate_times, 6].reshape(-1))
    res_rotated = np.matmul(res_trajs.reshape(-1, 1, 2), rot_matrix).reshape(res_trajs.shape[0], res_trajs.shape[1], 2)

    # get lateral and longitudinal part of vector, on time stamps and types
    res_abs = np.abs(res_rotated)
    results = make_results_dictionary(res_abs, object_types)

    return results


def get_metric_keys(obj_types: List[str]):
    metric_keys = []
    # for time in [3, 5, 8]:
    #     metric_keys.append(f"along/MDE_{time}")
    #     metric_keys.append(f"across/MDE_{time}")

    # mean over obj and time
    for obj_type in obj_types:
        for time in [3, 5, 8]:
            metric_keys.append(f"along/MDE_{obj_type}_{time}")
            metric_keys.append(f"across/MDE_{obj_type}_{time}")

    return metric_keys


def make_results_dictionary(angled_dist, object_types) -> Dict[str, float]:
    results = {}
    # mean error at time
    res_mean = angled_dist.mean(0)
    for i, time in zip([0, 1, 2], [3, 5, 8]):
        results[f"along/MDE_{time}"] = res_mean[i, 0]
        results[f"across/MDE_{time}"] = res_mean[i, 1]

    # mean over obj and time
    for obj_type in np.unique(object_types):
        res_obj_mean = angled_dist[object_types == obj_type].mean(0)
        for i, time in zip([0, 1, 2], [3, 5, 8]):
            results[f"along/MDE_{obj_type}_{time}"] = res_obj_mean[i, 0]
            results[f"across/MDE_{obj_type}_{time}"] = res_obj_mean[i, 1]

    return results


def plot_trajs(gt_traj, pred_traj, gt_valid, plt_id: int):
    plt.scatter(
        x=gt_traj[plt_id, gt_valid[plt_id], 0], y=gt_traj[plt_id, gt_valid[plt_id], 1], c="green", s=1, label="GT"
    )
    plt.scatter(
        x=pred_traj[plt_id, gt_valid[plt_id], 0],
        y=pred_traj[plt_id, gt_valid[plt_id], 1],
        c="orange",
        s=1,
        label="Pred",
    )
    plt.legend()
    plt.show()


def plot_trajectory(traj, cmap="hsv"):
    plt.scatter(x=traj[:, 0], y=traj[:, 1], c=np.arange(traj.shape[0]), s=1, cmap=cmap)
    plt.colorbar(label="Time")
    plt.title("Trajectory over time")
    plt.axis("equal")
    plt.show()


def plot_headings(headings):
    plt.plot(headings)
    plt.show()
