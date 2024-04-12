# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved


import torch
import torch.nn as nn

from mtr.models.utils import polyline_encoder
from mtr.models.utils.lidar_pointnet_encoder import (
    LidarPointNetEncoder,
    LidarPointNetEncoderLin,
)
from mtr.models.utils.transformer import (
    position_encoding_utils,
    transformer_encoder_layer,
)
from mtr.ops.knn import knn_utils
from mtr.utils import common_utils


class MTREncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        # build polyline encoders
        self.agent_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg["NUM_INPUT_ATTR_AGENT"] + 1,
            hidden_dim=self.model_cfg["NUM_CHANNEL_IN_MLP_AGENT"],
            num_layers=self.model_cfg["NUM_LAYER_IN_MLP_AGENT"],
            out_channels=self.model_cfg["D_MODEL"],
        )
        self.map_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg["NUM_INPUT_ATTR_MAP"],
            hidden_dim=self.model_cfg["NUM_CHANNEL_IN_MLP_MAP"],
            num_layers=self.model_cfg["NUM_LAYER_IN_MLP_MAP"],
            num_pre_layers=self.model_cfg["NUM_LAYER_IN_PRE_MLP_MAP"],
            out_channels=self.model_cfg["D_MODEL"],
        )

        # build pointnet encoder for lidar
        if self.model_cfg.get("USE_LIDAR_POINTS", False):
            self.build_lidar_encoder()

        # build transformer encoder layers
        self.use_local_attn = self.model_cfg.get("USE_LOCAL_ATTN", False)
        self_attn_layers = []
        for _ in range(self.model_cfg["NUM_ATTN_LAYERS"]):
            self_attn_layers.append(
                self.build_transformer_encoder_layer(
                    d_model=self.model_cfg["D_MODEL"],
                    nhead=self.model_cfg["NUM_ATTN_HEAD"],
                    dropout=self.model_cfg.get("DROPOUT_OF_ATTN", 0.1),
                    normalize_before=False,
                    use_local_attn=self.use_local_attn,
                )
            )

        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.num_out_channels = self.model_cfg["D_MODEL"]

    def build_lidar_encoder(self):
        # determine the feature dimension shape of points, 3 for x,y,z, + features + class one hot
        self.model_cfg["lidar_in_k"] = 3 + len(self.model_cfg["INCLUDE_LIDAR_FEATURES"])
        if self.model_cfg.get("LP_ADD_ONEHOT_CLASS"):
            self.model_cfg["lidar_in_k"] += 3

        if self.model_cfg.get("lidar_encoder") == "double-pointnet":
            self.lidar_pointnet_encoder = LidarPointNetEncoder(
                input_channels=self.model_cfg["lidar_in_k"],
                cfg=self.model_cfg["double_poly"],
                output_channels=self.model_cfg["D_MODEL"],
            )
        elif self.model_cfg.get("lidar_encoder") == "pointnet-linear":
            self.lidar_pointnet_encoder = LidarPointNetEncoderLin(
                input_channels=self.model_cfg["lidar_in_k"],
                cfg=self.model_cfg["double_poly"],
                output_channels=self.model_cfg["D_MODEL"],
            )
        else:
            raise ValueError(f'Unknown lidar encoder {self.model_cfg.get("lidar_encoder")}')

    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels,
        )
        return ret_polyline_encoder

    def build_transformer_encoder_layer(
        self, d_model, nhead, dropout=0.1, normalize_before=False, use_local_attn=False
    ):
        single_encoder_layer = transformer_encoder_layer.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            normalize_before=normalize_before,
            use_local_attn=use_local_attn,
        )
        return single_encoder_layer

    def apply_global_attn(self, x, x_mask, x_pos):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)

        batch_size, N, d_model = x.shape
        x_t = x.permute(1, 0, 2)
        x_mask_t = x_mask.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)

        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_t, hidden_dim=d_model)

        for k in range(len(self.self_attn_layers)):
            x_t = self.self_attn_layers[k](src=x_t, src_key_padding_mask=~x_mask_t, pos=pos_embedding)
        x_out = x_t.permute(1, 0, 2)  # (batch_size, N, d_model)
        return x_out

    def apply_local_attn(self, x, x_mask, x_pos, num_of_neighbors):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)
        batch_size, N, d_model = x.shape

        x_stack_full = x.view(-1, d_model)  # (batch_size * N, d_model)
        x_mask_stack = x_mask.view(-1)
        x_pos_stack_full = x_pos.view(-1, 3)
        batch_idxs_full = torch.arange(batch_size).type_as(x)[:, None].repeat(1, N).view(-1).int()  # (batch_size * N)

        # filter invalid elements
        x_stack = x_stack_full[x_mask_stack]
        x_pos_stack = x_pos_stack_full[x_mask_stack]
        batch_idxs = batch_idxs_full[x_mask_stack]

        # knn
        batch_offsets = common_utils.get_batch_offsets(batch_idxs=batch_idxs, bs=batch_size).to(
            device=x.device
        )  # (batch_size + 1)
        batch_cnt = batch_offsets[1:] - batch_offsets[:-1]

        index_pair = knn_utils.knn_batch_mlogk(
            x_pos_stack, x_pos_stack, batch_idxs, batch_offsets, num_of_neighbors
        )  # (num_valid_elems, K)

        # positional encoding
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(
            x_pos_stack[None, :, 0:2], hidden_dim=d_model
        )[0]

        # local attn
        output = x_stack
        for k in range(len(self.self_attn_layers)):
            output = self.self_attn_layers[k](
                src=output,
                pos=pos_embedding,
                index_pair=index_pair,
                query_batch_cnt=batch_cnt,
                key_batch_cnt=batch_cnt,
                index_pair_batch=batch_idxs,
            )

        ret_full_feature = torch.zeros_like(x_stack_full)  # (batch_size * N, d_model)
        ret_full_feature[x_mask_stack] = output

        ret_full_feature = ret_full_feature.view(batch_size, N, d_model)
        return ret_full_feature

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
              input_dict:
        """
        input_dict = batch_dict["input_dict"]
        obj_trajs, obj_trajs_mask = input_dict["obj_trajs"], input_dict["obj_trajs_mask"]
        map_polylines, map_polylines_mask = input_dict["map_polylines"], input_dict["map_polylines_mask"]
        if self.model_cfg.get("USE_LIDAR_POINTS", False):
            center_lidar_points = input_dict["center_lidar_points"]
            lidar_mask = input_dict["lidar_mask"]
            num_lidar_timestamps = center_lidar_points.shape[1]

        obj_trajs_last_pos = input_dict["obj_trajs_last_pos"]
        map_polylines_center = input_dict["map_polylines_center"]
        track_index_to_predict = input_dict["track_index_to_predict"]

        assert obj_trajs_mask.dtype == torch.bool and map_polylines_mask.dtype == torch.bool

        num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
        num_polylines = map_polylines.shape[1]

        # apply polyline encoder
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        obj_polylines_feature = self.agent_polyline_encoder(
            obj_trajs_in, obj_trajs_mask
        )  # (num_center_objects, num_objects, C)
        map_polylines_feature = self.map_polyline_encoder(
            map_polylines, map_polylines_mask
        )  # (num_center_objects, num_polylines, C)

        obj_valid_mask = obj_trajs_mask.sum(dim=-1) > 0  # (num_center_objects, num_objects)
        map_valid_mask = map_polylines_mask.sum(dim=-1) > 0  # (num_center_objects, num_polylines)

        if self.model_cfg.get("USE_LIDAR_POINTS", False):
            if (
                self.model_cfg.get("lidar_encoder") == "double-pointnet"
                or self.model_cfg.get("lidar_encoder") == "pointnet-linear"
            ):
                # apply the double polyline
                if self.model_cfg.get("output_max_idx", False):
                    lidar_features, idx_max_pool = self.lidar_pointnet_encoder(
                        center_lidar_points, lidar_mask, output_max_idx=True
                    )
                    batch_dict["idx_max_pool"] = idx_max_pool
                else:
                    lidar_features = self.lidar_pointnet_encoder(
                        center_lidar_points, lidar_mask, output_max_idx=False
                    )
            # mask of lidar features, is true if objects has any lidar points
            lidar_mask = torch.unsqueeze(lidar_mask.sum(dim=-1).sum(dim=-1) > 0, dim=1)

            # position of lidar is the average position of the car over the last second
            lidar_pos = obj_trajs[torch.arange(num_center_objects), track_index_to_predict, :, :3].mean(
                dim=1, keepdim=True
            )

            # mask input for self attention
            global_token_feature = torch.cat((obj_polylines_feature, map_polylines_feature, lidar_features), dim=1)
            global_token_mask = torch.cat((obj_valid_mask, map_valid_mask, lidar_mask), dim=1)
            global_token_pos = torch.cat((obj_trajs_last_pos, map_polylines_center, lidar_pos), dim=1)
        else:
            # mask input for self attention
            global_token_feature = torch.cat((obj_polylines_feature, map_polylines_feature), dim=1)
            global_token_mask = torch.cat((obj_valid_mask, map_valid_mask), dim=1)
            global_token_pos = torch.cat((obj_trajs_last_pos, map_polylines_center), dim=1)

        # apply self-attn
        if self.use_local_attn:
            global_token_feature = self.apply_local_attn(
                x=global_token_feature,
                x_mask=global_token_mask,
                x_pos=global_token_pos,
                num_of_neighbors=self.model_cfg["NUM_OF_ATTN_NEIGHBORS"],
            )
        else:
            global_token_feature = self.apply_global_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos
            )

        obj_polylines_feature = global_token_feature[:, :num_objects]
        map_polylines_feature = global_token_feature[:, num_objects : (num_polylines + num_objects)]
        assert map_polylines_feature.shape[1] == num_polylines

        if self.model_cfg.get("USE_LIDAR_POINTS", False):
            lidar_feature = global_token_feature[:, (num_polylines + num_objects) :]
            batch_dict["lidar_feature"] = lidar_feature
            batch_dict["lidar_mask"] = lidar_mask
            batch_dict["lidar_pos"] = lidar_pos

        # organize return features
        center_objects_feature = obj_polylines_feature[torch.arange(num_center_objects), track_index_to_predict]

        batch_dict["center_objects_feature"] = center_objects_feature
        batch_dict["obj_feature"] = obj_polylines_feature
        batch_dict["map_feature"] = map_polylines_feature
        batch_dict["obj_mask"] = obj_valid_mask
        batch_dict["map_mask"] = map_valid_mask
        batch_dict["obj_pos"] = obj_trajs_last_pos
        batch_dict["map_pos"] = map_polylines_center

        return batch_dict
