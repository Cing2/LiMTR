# Code is copy of Waymo opendataset for decompressing LiDAR images to point cloud rewritten with pytorch

import os
import time
import zlib
from typing import Tuple

import numpy as np
import torch
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import compressed_lidar_pb2, scenario_pb2

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"


def decompress_lidar_pointcloud(scenario_augmented):
    frame_points_xyz = []  # map from frame indices to point clouds
    frame_points_feature = []
    frame_i = 0

    # Extract point cloud xyz and features from each LiDAR and merge them for each
    # laser frame in the scenario proto.
    for frame_lasers in scenario_augmented.compressed_frame_laser_data:
        points_xyz_list = []
        points_feature_list = []
        frame_pose = (
            torch.tensor(scenario_augmented.compressed_frame_laser_data[frame_i].pose.transform)
            .reshape((4, 4))
            .to(DEVICE)
        )
        for laser in frame_lasers.lasers:
            if laser.name == dataset_pb2.LaserName.TOP:
                c = _get_laser_calib(frame_lasers, laser.name)
                (
                    points_xyz,
                    points_feature,
                    points_xyz_return2,
                    points_feature_return2,
                ) = extract_top_lidar_points(laser, frame_pose, c)
                # print(points_xyz.shape, points_xyz_return2.shape)
            else:
                c = _get_laser_calib(frame_lasers, laser.name)
                (
                    points_xyz,
                    points_feature,
                    points_xyz_return2,
                    points_feature_return2,
                ) = extract_side_lidar_points(laser, c)
            points_xyz_list.append(points_xyz.cpu().numpy())
            points_xyz_list.append(points_xyz_return2.cpu().numpy())
            points_feature_list.append(points_feature.cpu().numpy())
            points_feature_list.append(points_feature_return2.cpu().numpy())
        frame_points_xyz.append(np.concatenate(points_xyz_list, axis=0))
        frame_points_feature.append(np.concatenate(points_feature_list, axis=0))
        frame_i += 1

    return frame_points_xyz, frame_points_feature


def _get_laser_calib(
    frame_lasers: compressed_lidar_pb2.CompressedFrameLaserData,
    laser_name: dataset_pb2.LaserName.Name,
):
    for laser_calib in frame_lasers.laser_calibrations:
        if laser_calib.name == laser_name:
            return laser_calib
    return None


def extract_top_lidar_points(
    laser: compressed_lidar_pb2.CompressedLaser,
    frame_pose: torch.Tensor,
    laser_calib: dataset_pb2.LaserCalibration,
) -> Tuple[torch.Tensor, ...]:
    """Extract point clouds from the top laser proto.

    Args:
      laser: the top laser proto.
      frame_pose: a (4, 4) array which decides the vehicle frame at which the
        cartesian points are computed.
      laser_calib: calib proto of the top lidar.

    Returns:
      points_xyz_ri1: a torch.Tensor of shape [#points, 3] for xyz coordinates from
        the 1st range image.
      points_feature_ri1: a torch.Tensor of shape [#points, 3] for point cloud
        features (range, intensity and elongation) from the 1st range image.
      points_xyz_ri2: a torch.Tensor of shape [#points, 3] for xyz coordinates from
        the 2nd range image.
      points_feature_ri2: a torch.Tensor of shape [#points, 3] for point cloud
        features (range, intensity and elongation) from the 2nd range image.
    """
    # Get top pose info and lidar calibrations.
    # -------------------------------------------
    range_image_pose_decompressed = decompress(laser.ri_return1.range_image_pose_delta_compressed)
    range_image_top_pose_arr = range_image_pose_decompressed.to(DEVICE)
    range_image_top_pose_rotation = get_rotation_matrix(
        range_image_top_pose_arr[..., 0],
        range_image_top_pose_arr[..., 1],
        range_image_top_pose_arr[..., 2],
    )
    range_image_top_pose_translation = range_image_top_pose_arr[..., 3:]
    range_image_top_pose_arr = get_transform(range_image_top_pose_rotation, range_image_top_pose_translation)

    pixel_pose_local = range_image_top_pose_arr
    pixel_pose_local = torch.unsqueeze(pixel_pose_local, dim=0)
    frame_pose_local = torch.unsqueeze(frame_pose, dim=0)

    # Extract point xyz and features.
    # -------------------------------------------
    points_xyz_list = []
    points_feature_list = []
    for ri in [laser.ri_return1, laser.ri_return2]:
        range_image = decompress(ri.range_image_delta_compressed)
        c = laser_calib
        if not c.beam_inclinations:
            beam_inclinations = _get_beam_inclinations(
                c.beam_inclination_min,
                c.beam_inclination_max,
                range_image.dims[0],
            )
        else:
            beam_inclinations = torch.tensor(c.beam_inclinations, device=DEVICE)
        beam_inclinations = torch.flip(beam_inclinations, [0])
        extrinsic = torch.tensor(c.extrinsic.transform, device=DEVICE).reshape([4, 4])
        range_image_mask = range_image[..., 0] > 0
        range_image_cartesian = extract_point_cloud_from_range_image(
            torch.unsqueeze(range_image[..., 0], dim=0),
            torch.unsqueeze(extrinsic, dim=0),
            torch.unsqueeze(beam_inclinations, dim=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local,
        )
        range_image_cartesian = torch.squeeze(range_image_cartesian, dim=0)
        # The points XYZ in the vehicle coordinate.
        # points_xyz = torch.gather_nd(range_image_cartesian, torch.where(range_image_mask)[0].reshape(range_image_mask.shape))
        points_xyz = range_image_cartesian[range_image_mask]
        points_xyz_list.append(points_xyz)
        # The range, intensity and elongation features.
        # points_feature = torch.gather_nd(range_image[..., 0:3], torch.where(range_image_mask)[0].reshape(range_image_mask.shape))
        points_feature = range_image[..., 0:3][range_image_mask]
        points_feature_list.append(points_feature)

    return (
        points_xyz_list[0],
        points_feature_list[0],
        points_xyz_list[1],
        points_feature_list[1],
    )


def extract_side_lidar_points(
    laser: compressed_lidar_pb2.CompressedLaser,
    laser_calib: dataset_pb2.LaserCalibration,
) -> Tuple[torch.Tensor, ...]:
    """Extract side lidar points.

    Args:
      laser: the side laser proto.
      laser_calib: calib proto of the side lidar.

    Returns:
      points_xyz_ri1: a torch.Tensor of shape [#points, 3] for xyz coordinates from
        the 1st range image.
      points_feature_ri1: a torch.Tensor of shape [#points, 3] for point cloud
        features (range, intensity and elongation) from the 1st range image.
      points_xyz_ri2: a torch.Tensor of shape [#points, 3] for xyz coordinates from
        the 2nd range image.
      points_feature_ri2: a torch.Tensor of shape [#points, 3] for point cloud
        features (range, intensity and elongation) from the 2nd range image.
    """
    points_xyz_list = []
    points_feature_list = []
    for ri in [laser.ri_return1, laser.ri_return2]:
        range_image = decompress(ri.range_image_delta_compressed)
        # Lidar calibration info.
        c = laser_calib
        if not c.beam_inclinations:
            beam_inclinations = _get_beam_inclinations(
                c.beam_inclination_min, c.beam_inclination_max, range_image.shape[0]
            )
        else:
            beam_inclinations = torch.tensor(c.beam_inclinations, device=DEVICE)
        beam_inclinations = torch.flip(beam_inclinations, [0])
        extrinsic = torch.tensor(c.extrinsic.transform, device=DEVICE).reshape([4, 4])

        range_image_mask = range_image[..., 0] > 0
        range_image_cartesian = extract_point_cloud_from_range_image(
            torch.unsqueeze(range_image[..., 0], dim=0),
            torch.unsqueeze(extrinsic, dim=0),
            torch.unsqueeze(beam_inclinations, dim=0),
            pixel_pose=None,
            frame_pose=None,
        )
        range_image_cartesian = torch.squeeze(range_image_cartesian, dim=0)
        # The points XYZ in the vehicle coordinate.
        # points_xyz = torch.gather_nd(range_image_cartesian, torch.where(range_image_mask))
        points_xyz = range_image_cartesian[range_image_mask]
        points_xyz_list.append(points_xyz)
        # The range, intensity and elongation features.
        points_feature = range_image[..., :3][range_image_mask]
        # points_feature = torch.gather_nd(range_image[..., 0:3], torch.where(range_image_mask))
        points_feature_list.append(points_feature)

    return (
        points_xyz_list[0],
        points_feature_list[0],
        points_xyz_list[1],
        points_feature_list[1],
    )


def decompress(compressed_range_image: bytes) -> torch.Tensor:
    """Decodes a delta encoded range image.

    Args:
      compressed_range_image: Compressed bytes of range image.

    Returns:
      decoded_range_image: A decoded range image.
    """
    lidar_proto = compressed_lidar_pb2.DeltaEncodedData.FromString(zlib.decompress(compressed_range_image))

    shape = torch.tensor(lidar_proto.metadata.shape, dtype=int)
    precision = torch.tensor(lidar_proto.metadata.quant_precision, dtype=float)
    decoded_mask = rldecode(torch.tensor(lidar_proto.mask, dtype=int))
    residuals = torch.tensor(np.fromiter(lidar_proto.residual, count=len(lidar_proto.residual), dtype=np.int64))
    decoded_residuals = torch.cumsum(residuals, dim=0)
    decoded_mask[decoded_mask > 0] = decoded_residuals
    decoded_range_image = torch.permute(decoded_mask.reshape(shape[2], shape[0], shape[1]), dims=(1, 2, 0)).to(
        torch.float
    )

    for c in range(shape[2]):
        decoded_range_image[..., c] = decoded_range_image[..., c] * precision[c]
    return decoded_range_image.to(DEVICE)


def rldecode(lengths: torch.Tensor) -> torch.Tensor:
    """Decodes an array using run-length decoding.

    Args:
        lengths (torch.Tensor): An array to decode.

    Returns:
        torch.Tensor: A run length decoded array.
    """
    # Create an array of alternating 0s and 1s
    alternating_pattern = (np.arange(lengths.shape[0]) + 1) % 2

    decoded = np.repeat(alternating_pattern, lengths)
    return torch.from_numpy(decoded)


def _get_beam_inclinations(beam_inclination_min: float, beam_inclination_max: float, height: int) -> torch.Tensor:
    """Gets the beam inclination information."""
    return compute_inclination(torch.tensor([beam_inclination_min, beam_inclination_max], device=DEVICE), height=height)


def compute_inclination(inclination_range: torch.Tensor, height: int, scope=None):
    """Computes uniform inclination range based the given range and height.

    Args:
      inclination_range: [..., 2] tensor. Inner dims are [min inclination, max
        inclination].
      height: an integer indicates height of the range image.
      scope: the name scope.

    Returns:
      inclination: [..., height] tensor. Inclinations computed.
    """
    diff = inclination_range[..., 1] - inclination_range[..., 0]
    inclination = (
        0.5 + torch.arange(0, height).to(inclination_range.dtype).to(inclination_range.device)
    ) / torch.tensor(height, dtype=inclination_range.dtype, device=inclination_range.device) * torch.unsqueeze(
        diff, axis=-1
    ) + inclination_range[
        ..., 0:1
    ]
    return inclination


def extract_point_cloud_from_range_image(
    range_image, extrinsic, inclination, pixel_pose=None, frame_pose=None, dtype=torch.float32
):
    """Extracts point cloud from range image.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
        image pixel.
      frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
        decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] with {x, y, z} as inner dims in vehicle
      frame.
    """
    range_image_polar = compute_range_image_polar(range_image, extrinsic, inclination, dtype=dtype)
    range_image_cartesian = compute_range_image_cartesian(
        range_image_polar, extrinsic, pixel_pose=pixel_pose, frame_pose=frame_pose, dtype=dtype
    )
    return range_image_cartesian


def get_transform(rotation, translation):
    """Combines NxN rotation and Nx1 translation to (N+1)x(N+1) transform.

    Args:
      rotation: [..., N, N] rotation tensor.
      translation: [..., N] translation tensor. This must have the same type as
        rotation.

    Returns:
      transform: [..., (N+1), (N+1)] transform tensor. This has the same type as
        rotation.
    """
    # [..., N, 1]
    translation_n_1 = translation[..., None]
    # [..., N, N+1]
    transform = torch.concat([rotation, translation_n_1], axis=-1)
    # [..., N]
    last_row = torch.zeros_like(translation)
    # [..., N+1]
    last_row = torch.concat([last_row, torch.ones_like(last_row[..., 0:1])], dim=-1)
    # [..., N+1, N+1]
    transform = torch.concat([transform, last_row[..., None, :]], dim=-2)
    return transform


def compute_range_image_polar(range_image, extrinsic, inclination, dtype=torch.float32):
    """Computes range image polar coordinates.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_polar: [B, H, W, 3] polar coordinates.
    """
    # pylint: disable=unbalanced-tuple-unpacking
    _, height, width = _combined_static_and_dynamic_shape(range_image)
    range_image_dtype = range_image.dtype
    range_image = range_image.to(dtype)
    extrinsic = extrinsic.to(dtype)
    inclination = inclination.to(dtype)

    # [B].
    az_correction = torch.atan2(extrinsic[..., 1, 0], extrinsic[..., 0, 0])
    # [W].
    ratios = ((torch.arange(width, 0, -1).to(dtype) - 0.5) / width).to(dtype).to(DEVICE)
    # [B, W].
    azimuth = (ratios * 2.0 - 1.0) * np.pi - torch.unsqueeze(az_correction, -1)

    # [B, H, W]
    azimuth_tile = torch.tile(azimuth[:, None, :], [1, height, 1])
    # [B, H, W]
    inclination_tile = torch.tile(inclination[:, :, None], [1, 1, width])
    range_image_polar = torch.stack([azimuth_tile, inclination_tile, range_image], dim=-1)
    return range_image_polar.to(range_image_dtype)


def compute_range_image_cartesian(range_image_polar, extrinsic, pixel_pose=None, frame_pose=None, dtype=torch.float32):
    """Computes range image cartesian coordinates from polar ones.

    Args:
      range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
        coordinate in sensor frame.
      extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
      pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
        range image pixel.
      frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
        It decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = range_image_polar.to(dtype)
    extrinsic = extrinsic.to(dtype)
    if pixel_pose is not None:
        pixel_pose = pixel_pose.to(dtype)
    if frame_pose is not None:
        frame_pose = frame_pose.to(dtype)

    azimuth, inclination, range_image_range = torch.unbind(range_image_polar, dim=-1)

    cos_azimuth = torch.cos(azimuth)
    sin_azimuth = torch.sin(azimuth)
    cos_incl = torch.cos(inclination)
    sin_incl = torch.sin(inclination)

    # [B, H, W].
    x = cos_azimuth * cos_incl * range_image_range
    y = sin_azimuth * cos_incl * range_image_range
    z = sin_incl * range_image_range

    # [B, H, W, 3]
    range_image_points = torch.stack([x, y, z], -1)
    # [B, 3, 3]
    rotation = extrinsic[..., 0:3, 0:3]
    # translation [B, 1, 3]
    translation = torch.unsqueeze(torch.unsqueeze(extrinsic[..., 0:3, 3], 1), 1)

    # To vehicle frame.
    # [B, H, W, 3]
    range_image_points = torch.einsum("bkr,bijr->bijk", rotation, range_image_points) + translation
    if pixel_pose is not None:
        # To global frame.
        # [B, H, W, 3, 3]
        pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
        # [B, H, W, 3]
        pixel_pose_translation = pixel_pose[..., 0:3, 3]
        # [B, H, W, 3]
        range_image_points = (
            torch.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_points) + pixel_pose_translation
        )
        if frame_pose is None:
            raise ValueError("frame_pose must be set when pixel_pose is set.")
        # To vehicle frame corresponding to the given frame_pose
        # [B, 4, 4]
        world_to_vehicle = torch.linalg.inv(frame_pose)
        world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
        world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
        # [B, H, W, 3]
        range_image_points = (
            torch.einsum("bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_points)
            + world_to_vehicle_translation[:, None, None, :]
        )

    range_image_points = range_image_points.to(range_image_polar_dtype)
    return range_image_points


def _combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      tensor: A tensor of any type.

    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = list(tensor.shape)
    dynamic_tensor_shape = tensor.shape
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape


def get_rotation_matrix(roll, pitch, yaw):
    """Gets a rotation matrix given roll, pitch, yaw.

    roll-pitch-yaw is z-y'-x'' intrinsic rotation which means we need to apply
    x(roll) rotation first, then y(pitch) rotation, then z(yaw) rotation.

    https://en.wikipedia.org/wiki/Euler_angles
    http://planning.cs.uiuc.edu/node102.html

    Args:
      roll : x-rotation in radians.
      pitch: y-rotation in radians. The shape must be the same as roll.
      yaw: z-rotation in radians. The shape must be the same as roll.
      name: the op name.

    Returns:
      A rotation tensor with the same data type of the input. Its shape is
        [input_shape_of_yaw, 3 ,3].
    """
    cos_roll = torch.cos(roll)
    sin_roll = torch.sin(roll)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    cos_pitch = torch.cos(pitch)
    sin_pitch = torch.sin(pitch)

    ones = torch.ones_like(yaw)
    zeros = torch.zeros_like(yaw)

    r_roll = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos_roll, -1.0 * sin_roll], dim=-1),
            torch.stack([zeros, sin_roll, cos_roll], dim=-1),
        ],
        dim=-2,
    )
    r_pitch = torch.stack(
        [
            torch.stack([cos_pitch, zeros, sin_pitch], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-1.0 * sin_pitch, zeros, cos_pitch], dim=-1),
        ],
        dim=-2,
    )
    r_yaw = torch.stack(
        [
            torch.stack([cos_yaw, -1.0 * sin_yaw, zeros], dim=-1),
            torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    return torch.matmul(r_yaw, torch.matmul(r_pitch, r_roll))


def load_scenario_data(tfrecord_file: str, nr: int = 0, scenario_name: str = None) -> scenario_pb2.Scenario:
    """Load a scenario proto from a tfrecord dataset file."""
    import tensorflow as tf

    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type="")
    for i, data in enumerate(dataset):
        if scenario_name is not None:
            scene = scenario_pb2.Scenario.FromString(data.numpy())
            if scene.scenario_id == scenario_name:
                print("Found scenario")
                break
            continue
        if i == nr:
            break
    else:
        if scenario_name is not None:
            print(f"Could not find {scenario_name}")
        else:
            print(f"nr {nr} is to high, max {i}")

    return scenario_pb2.Scenario.FromString(data.numpy())


def old_speed_test(scenario_augmented):
    start_time = time.perf_counter()
    points_xyz, points_feature = old_decompress_lidar_pointcloud(scenario_augmented)
    print(f"Time taken old: {time.perf_counter() - start_time}")


def old_decompress_lidar_pointcloud(scenario_augmented):
    from waymo_open_dataset.utils import womd_lidar_utils

    frame_points_xyz = []  # map from frame indices to point clouds
    frame_points_feature = []
    frame_i = 0

    def _get_laser_calib(
        frame_lasers: compressed_lidar_pb2.CompressedFrameLaserData,
        laser_name: dataset_pb2.LaserName.Name,
    ):
        for laser_calib in frame_lasers.laser_calibrations:
            if laser_calib.name == laser_name:
                return laser_calib
        return None

    # Extract point cloud xyz and features from each LiDAR and merge them for each
    # laser frame in the scenario proto.
    for frame_lasers in scenario_augmented.compressed_frame_laser_data:
        points_xyz_list = []
        points_feature_list = []
        frame_pose = np.reshape(
            np.array(scenario_augmented.compressed_frame_laser_data[frame_i].pose.transform),
            (4, 4),
        )
        for laser in frame_lasers.lasers:
            if laser.name == dataset_pb2.LaserName.TOP:
                c = _get_laser_calib(frame_lasers, laser.name)
                (
                    points_xyz,
                    points_feature,
                    points_xyz_return2,
                    points_feature_return2,
                ) = womd_lidar_utils.extract_top_lidar_points(laser, frame_pose, c)
                # print(points_xyz.shape, points_xyz_return2.shape)
            else:
                c = _get_laser_calib(frame_lasers, laser.name)
                (
                    points_xyz,
                    points_feature,
                    points_xyz_return2,
                    points_feature_return2,
                ) = womd_lidar_utils.extract_side_lidar_points(laser, c)
            points_xyz_list.append(points_xyz.numpy())
            points_xyz_list.append(points_xyz_return2.numpy())
            points_feature_list.append(points_feature.numpy())
            points_feature_list.append(points_feature_return2.numpy())
        frame_points_xyz.append(np.concatenate(points_xyz_list, axis=0))
        frame_points_feature.append(np.concatenate(points_feature_list, axis=0))
        frame_i += 1

    return frame_points_xyz, frame_points_feature


def test():
    from waymo_open_dataset.utils import womd_lidar_utils

    data_file = "../data/waymo/scenario/training/training.tfrecord-00000-of-01000"
    scenario = load_scenario_data(data_file, 0)  # , scenario_name='27dfda743246259d')
    # add lidar info to scene
    dir_original = os.path.dirname(data_file)
    path_lidar = os.path.join(
        dir_original, "..", "..", "lidar", os.path.split(dir_original)[-1], f"{scenario.scenario_id}.tfrecord"
    )
    print(f"Scenario: {scenario.scenario_id}")
    womd_lidar_scenario = load_scenario_data(path_lidar)
    scenario_augmented = womd_lidar_utils.augment_womd_scenario_with_lidar_points(scenario, womd_lidar_scenario)

    # run decompression
    # old_speed_test(scenario_augmented)

    start_time = time.perf_counter()
    points_xyz, points_features = decompress_lidar_pointcloud(scenario_augmented)
    print(f"Time taken: {time.perf_counter() - start_time}")
    print(points_xyz[0].shape, points_features[0].shape)

    # visualize


def visualize(points_xyz):
    import open3d as o3d

    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(points_xyz[0])
    o3d.visualization.draw_geometries([geom])


if __name__ == "__main__":
    test()
