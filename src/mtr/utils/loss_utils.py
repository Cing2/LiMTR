# Adaptation to: 
# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved


import torch
from torch import nn


def nll_loss_gmm_direct(
    pred_scores,
    pred_trajs,
    gt_trajs,
    gt_valid_mask,
    pre_nearest_mode_idxs=None,
    timestamp_loss_weight=None,
    use_square_gmm=False,
    log_std_range=(-1.609, 5.0),
    rho_limit=0.5,
):
    """
    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi

    Args:
        pred_scores (batch_size, num_modes):
        pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        timestamp_loss_weight (num_timestamps):
    """
    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3
    else:
        assert pred_trajs.shape[-1] == 5

    batch_size = pred_scores.shape[0]

    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1)
        distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)

        nearest_mode_idxs = distance.argmin(dim=-1)
    nearest_mode_bs_idxs = torch.arange(batch_size).to(nearest_mode_idxs.dtype).to(pred_trajs.device)  # (batch_size, 2)

    nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
    res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
    dx = res_trajs[:, :, 0]
    dy = res_trajs[:, :, 1]

    if use_square_gmm:
        log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
        rho = torch.zeros_like(log_std1)
    else:
        log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)  # (0.2m to 150m)
        std2 = torch.exp(log_std2)  # (0.2m to 150m)
        rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

    gt_valid_mask = gt_valid_mask.to(pred_scores.dtype)
    if timestamp_loss_weight is not None:
        gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

    # -log(a^-1 * e^b) = log(a) - b
    reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
    reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * (
        (dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2)
    )  # (batch_size, num_timestamps)

    reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

    return reg_loss, nearest_mode_idxs


def nll_loss_gmm_pytorch(
    pred_scores,
    pred_trajs,
    gt_trajs,
    gt_valid_mask,
    pre_nearest_mode_idxs=None,
    log_std_range=(-1.609, 5.0),
    rho_limit=0.5,
):
    """
    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi

    Args:
        pred_scores (batch_size, num_modes):
        pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        timestamp_loss_weight (num_timestamps):
    """
    batch_size = pred_scores.shape[0]

    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1)
        distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)

        nearest_mode_idxs = distance.argmin(dim=-1)
    nearest_mode_bs_idxs = torch.arange(batch_size).to(nearest_mode_idxs.dtype).to(pred_trajs.device)  # (batch_size, 2)

    nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)

    log_std = torch.clip(nearest_trajs[:, :, 2:4], min=log_std_range[0], max=log_std_range[1])
    std = torch.exp(log_std)  # (0.2m to 150m)
    gt_valid_mask = gt_valid_mask.to(pred_scores.dtype)

    loss_fn = nn.GaussianNLLLoss(reduction="none")

    loss = loss_fn(nearest_trajs[..., :2], gt_trajs, std)
    loss = (loss[..., 0] + loss[..., 1]) * gt_valid_mask
    loss = loss.sum(dim=-1)

    return loss, nearest_mode_idxs


def rotate_trajs(trajs, headings):
    """Rotate trajectories on given headings.

    Args:
        trajs (tensor): (B, N, 2)
        headings (tensor): (B, N)

    Returns:
        tensor: trajs rotated on heading
    """
    headings = headings.reshape(-1)
    cosa = torch.cos(headings)
    sina = torch.sin(headings)
    rot_matrix = torch.stack((cosa, sina, -sina, cosa), axis=1).reshape(-1, 2, 2)

    traj_rotated = torch.matmul(trajs[..., :2].reshape(-1, 1, 2), rot_matrix).reshape(trajs.shape[0], trajs.shape[1], 2)

    assert trajs.shape == traj_rotated.shape

    return traj_rotated


def nll_loss_gmm_direct_angled(
    pred_scores: torch.Tensor,
    pred_trajs: torch.Tensor,
    gt_trajs: torch.Tensor,
    gt_valid_mask: torch.Tensor,
    pre_nearest_mode_idxs=None,
    timestamp_loss_weight=None,
    use_square_gmm=False,
    log_std_range=(-1.609, 5.0),
    rho_limit=0.5,
    align_weights: float = None,  # along, across
    use_torch: bool = False,
):
    """
    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi

    Args:
        pred_scores (batch_size, num_modes):
        pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        timestamp_loss_weight (num_timestamps):
        align_weights (float): weights to prefer direction of along the car or across, default is not
    """
    if use_torch:
        return nll_loss_gmm_pytorch(
            pred_scores=pred_scores,
            pred_trajs=pred_trajs,
            gt_trajs=gt_trajs[..., :2],
            gt_valid_mask=gt_valid_mask,
            pre_nearest_mode_idxs=pre_nearest_mode_idxs,
            log_std_range=log_std_range,
            rho_limit=rho_limit,
        )
    elif align_weights is None:
        return nll_loss_gmm_direct(
            pred_scores=pred_scores,
            pred_trajs=pred_trajs,
            gt_trajs=gt_trajs[..., :2],
            gt_valid_mask=gt_valid_mask,
            pre_nearest_mode_idxs=pre_nearest_mode_idxs,
            timestamp_loss_weight=timestamp_loss_weight,
            use_square_gmm=use_square_gmm,
            log_std_range=log_std_range,
            rho_limit=rho_limit,
        )

    else:
        return _nll_loss_gmm_direct_angled(
            pred_scores=pred_scores,
            pred_trajs=pred_trajs,
            gt_trajs=gt_trajs,
            gt_valid_mask=gt_valid_mask,
            pre_nearest_mode_idxs=pre_nearest_mode_idxs,
            timestamp_loss_weight=timestamp_loss_weight,
            use_square_gmm=use_square_gmm,
            log_std_range=log_std_range,
            rho_limit=rho_limit,
            align_weights=align_weights,
        )


def _nll_loss_gmm_direct_angled(
    pred_scores: torch.Tensor,
    pred_trajs: torch.Tensor,
    gt_trajs: torch.Tensor,
    gt_valid_mask: torch.Tensor,
    pre_nearest_mode_idxs=None,
    timestamp_loss_weight=None,
    use_square_gmm=False,
    log_std_range=(-1.609, 5.0),
    rho_limit=0.5,
    align_weights: float = None,  # along, across
):
    if align_weights is None:
        # make sure defaults are set
        align_weights = (1.0, 1.0)

    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3
    else:
        assert pred_trajs.shape[-1] == 5

    batch_size = pred_scores.shape[0]

    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        # get average closest trajectory
        distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :2]).norm(dim=-1)
        distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)

        nearest_mode_idxs = distance.argmin(dim=-1)
    nearest_mode_bs_idxs = torch.arange(batch_size).to(nearest_mode_idxs.dtype).to(pred_trajs.device)  # (batch_size, 2)

    nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
    res_trajs = gt_trajs[:, :, :2] - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)

    # rotate res trajectories to get the vector aligned with the heading of the vehicles
    # the x is then along the vehicle and y is across the target
    # only rotate if align weights are not default
    res_trajs = rotate_trajs(res_trajs, -gt_trajs[..., 4])

    dx = res_trajs[:, :, 0]
    dy = res_trajs[:, :, 1]

    if use_square_gmm:
        log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
        rho = torch.zeros_like(log_std1)
    else:
        # also rotate the standard deviation to align with the original prediction
        stds_rot = rotate_trajs(nearest_trajs[..., 2:4], -gt_trajs[..., 4])

        log_std1 = torch.clip(stds_rot[..., 0], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clip(stds_rot[..., 1], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)  # (0.2m to 150m)
        std2 = torch.exp(log_std2)  # (0.2m to 150m)
        rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

    gt_valid_mask = gt_valid_mask.to(pred_scores.dtype)
    if timestamp_loss_weight is not None:
        gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

    # -log(a^-1 * e^b) = log(a) - b
    reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
    reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * (
        (align_weights[0] * dx**2) / (std1**2)
        + (align_weights[1] * dy**2) / (std2**2)
        - 2 * rho * dx * dy / (std1 * std2)
    )  # (batch_size, num_timestamps)

    reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

    return reg_loss, nearest_mode_idxs

