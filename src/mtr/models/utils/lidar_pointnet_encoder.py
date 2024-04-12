from typing import Dict, List

import torch
import torch.nn as nn

from ..utils import common_layers


class AdeptPolylineEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        pooling_type: str,
        pre_layers: List[int],
        add_global: bool,
        bet_layers: List[int],
        out_channels: int = None,
        without_bnorm: bool = False,
    ):
        """Polyline encoder with more flexibility.

        Args:
            in_channels (int): dimension of input
            pooling_type (str): which pooling function to use, can be either max or average
            pre_layers (List[int]): hidden dim of layers before adding global
            add_global (bool): if to add global feature
            bet_layers (List[int]): hidden dims of layers after global
            out_channels (int, optional): dimension of output. Defaults to None.
        """
        super().__init__()
        # TODO: add input transform option
        assert pooling_type in ["max", "average"], f"unknown pooling type {pooling_type}"
        self.add_global = add_global
        self.pooling_type = pooling_type
        self.in_channels = in_channels

        self.pre_mlps = common_layers.build_mlps(
            c_in=in_channels, mlp_channels=pre_layers, ret_before_act=False, without_norm=without_bnorm
        )

        dim_in_between = pre_layers[-1]
        if add_global:
            dim_in_between *= 2
        self.mlps = common_layers.build_mlps(
            c_in=dim_in_between, mlp_channels=bet_layers, ret_before_act=False, without_norm=without_bnorm
        )

        if out_channels is not None:
            self.out_mlps = common_layers.build_mlps(
                c_in=bet_layers[-1], mlp_channels=[bet_layers[-1], out_channels], ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None

    def forward(self, lidar_points, lidar_mask, output_max_idx: bool = False):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):
            output_max_idx (bool): if to also output idx of max pooling

        Returns:
        """
        if output_max_idx:
            assert self.pooling_type == "max", "cannot output idx of max pool for average"

        list_idx_max_pool = []

        batch_size, num_timestamps, num_points_each_polylines, C = lidar_points.shape

        polylines_feature_valid = self.pre_mlps(lidar_points[lidar_mask].reshape(-1, C))  # (N, C)
        polylines_feature = lidar_points.new_zeros(
            batch_size, num_timestamps, num_points_each_polylines, polylines_feature_valid.shape[-1]
        )
        polylines_feature[lidar_mask] = polylines_feature_valid.reshape(polylines_feature[lidar_mask].shape)

        # get global feature
        if self.add_global:
            if self.pooling_type == "max":
                pooled_feature, idx_max_pool_global = polylines_feature.max(dim=2)
                if output_max_idx:
                    list_idx_max_pool.append(idx_max_pool_global.cpu().numpy())
            elif self.pooling_type == "average":
                pooled_feature = polylines_feature.mean(dim=2)

            polylines_feature = torch.cat(
                (polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1
            )

        # mlp
        polylines_feature_valid = self.mlps(polylines_feature[lidar_mask].reshape(-1, polylines_feature.shape[-1]))
        feature_buffers = polylines_feature.new_zeros(
            batch_size, num_timestamps, num_points_each_polylines, polylines_feature_valid.shape[-1]
        )
        feature_buffers[lidar_mask] = polylines_feature_valid.reshape(feature_buffers[lidar_mask].shape)

        # max-pooling
        if self.pooling_type == "max":
            feature_buffers, idx_max_pool_feature = feature_buffers.max(dim=2)  # (batch_size, num_polylines, C)
            if output_max_idx:
                list_idx_max_pool.append(idx_max_pool_feature.cpu().numpy())
        elif self.pooling_type == "average":
            feature_buffers = feature_buffers.mean(dim=2)  # (batch_size, num_polylines, C)

        # out-mlp
        if self.out_mlps is not None:
            valid_mask = lidar_mask.sum(dim=-1) > 0
            feature_buffers_valid = self.out_mlps(
                feature_buffers[valid_mask].reshape(-1, feature_buffers.shape[-1])
            )  # (N, C)
            feature_buffers = feature_buffers.new_zeros(batch_size, num_timestamps, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid.reshape(feature_buffers[valid_mask].shape)

        return feature_buffers, list_idx_max_pool


class LidarPointNetEncoder(nn.Module):
    def __init__(self, input_channels: int, cfg: Dict, output_channels: int):
        """Lidar Polyline encoder based on pointnet. Has two polyline encoder to compress for a
        single object the number of points and then the time.

        Args:
            input_channels (int): dimension of input
            cfg (Dict): configuration
            output_channels (int): output channels
        """
        super().__init__()
        if cfg.get("layers_per_block") is not None:
            # set nr layers per block for each
            cfg["part1"]["num_pre_layers"] = cfg.get("layers_per_block")
            cfg["part1"]["num_bet_layers"] = cfg.get("layers_per_block")
            cfg["part2"]["num_pre_layers"] = cfg.get("layers_per_block")
            cfg["part2"]["num_bet_layers"] = cfg.get("layers_per_block")

        self.point_compression = AdeptPolylineEncoder(
            in_channels=input_channels,
            pooling_type=cfg["pooling"],
            pre_layers=[cfg["part1"]["dim_pre_layers"]] * cfg["part1"]["num_pre_layers"],
            add_global=cfg["part1"]["add_global"],
            bet_layers=[cfg["part1"]["dim_bet_layers"]] * cfg["part1"]["num_bet_layers"],
            out_channels=None,
            input_transform=cfg.get("input_transform"),
            without_bnorm=cfg.get("without_bnorm", False),
        )
        self.time_compression = AdeptPolylineEncoder(
            in_channels=cfg["part1"]["dim_bet_layers"],
            pooling_type=cfg["pooling"],
            pre_layers=[cfg["part2"]["dim_pre_layers"]] * cfg["part2"]["num_pre_layers"],
            add_global=cfg["part2"]["add_global"],
            bet_layers=[cfg["part2"]["dim_bet_layers"]] * cfg["part2"]["num_bet_layers"],
            out_channels=output_channels,
            without_bnorm=cfg.get("without_bnorm", False),
        )

    def forward(self, lidar_points, lidar_mask, output_max_idx: bool = False):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):
            output_max_idx (bool): if to retrieve the lidar max output

        Returns:
        """
        lidar_features, idx_max_pool = self.point_compression(lidar_points, lidar_mask, output_max_idx=output_max_idx)

        # second polyline encoder to compress time aspect, add dimension to use same code
        lidar_mask = torch.unsqueeze(lidar_mask.sum(dim=-1) > 0, 1)
        lidar_features, idx_max_pool2 = self.time_compression(
            torch.unsqueeze(lidar_features, 1), lidar_mask, output_max_idx=output_max_idx
        )
        idx_max_pool.extend(idx_max_pool2)

        if output_max_idx:
            return lidar_features, idx_max_pool

        return lidar_features


class LidarPointNetEncoderLin(nn.Module):
    def __init__(self, input_channels: int, cfg: Dict, output_channels: int):
        """Lidar Polyline encoder based on pointnet. Has two polyline encoder to compress for a
        single object the number of points and then the time.

        Args:
            input_channels (int): dimension of input
            cfg (Dict): configuration
            output_channels (int): output channels
        """
        super().__init__()
        if cfg.get("layers_per_block") is not None:
            # set nr layers per block for each
            cfg["part1"]["num_pre_layers"] = cfg.get("layers_per_block")
            cfg["part1"]["num_bet_layers"] = cfg.get("layers_per_block")
            cfg["part2"]["num_pre_layers"] = cfg.get("layers_per_block")
            cfg["part2"]["num_bet_layers"] = cfg.get("layers_per_block")

        self.point_compression = AdeptPolylineEncoder(
            in_channels=input_channels,
            pooling_type=cfg["pooling"],
            pre_layers=[cfg["part1"]["dim_pre_layers"]] * cfg["part1"]["num_pre_layers"],
            add_global=cfg["part1"]["add_global"],
            bet_layers=[cfg["part1"]["dim_bet_layers"]] * cfg["part1"]["num_bet_layers"],
            out_channels=None,
            input_transform=cfg.get("input_transform"),
            without_bnorm=cfg.get("without_bnorm", False),
        )
        time_layers = [cfg["part2"]["dim_bet_layers"]] * cfg["part2"]["num_bet_layers"]
        time_layers.append(output_channels)
        time_in = cfg["part1"]["dim_bet_layers"] * cfg["nr_timestamps"]

        self.time_compression = common_layers.build_mlps(
            time_in,
            mlp_channels=time_layers,
            without_norm=cfg.get("without_bnorm", False),
        )

    def forward(self, lidar_points, lidar_mask, output_max_idx: bool = False):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):
            output_max_idx (bool): if to retrieve the lidar max output

        Returns:
        """
        lidar_features, idx_max_pool = self.point_compression(lidar_points, lidar_mask, output_max_idx=output_max_idx)

        batch_size, num_timeframes, C = lidar_features.shape
        # second polyline encoder to compress time aspect
        # concat timestamps together and pass through linear layers
        lidar_features = self.time_compression(lidar_features.reshape(-1, num_timeframes * C))  # (N, C)
        lidar_features = torch.unsqueeze(lidar_features, dim=1)

        if output_max_idx:
            return lidar_features, idx_max_pool

        return lidar_features
