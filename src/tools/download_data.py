import argparse
import os
import subprocess
from functools import partial

import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

bucket_name = "waymo_open_dataset_motion_v_1_2_0"
BASE_URI = f"gs://{bucket_name}/uncompressed/"
LOCAL_ROOT_DIR = "gpfs/work5/0/tesr0492"


def check_file_downloaded(local_dir: str, cloud_file: str):
    """Check if a google cloud docstring has been downloaded.

    Args:
        root_dir (str): local dir where data is downloaded
        cloud_file (str): path to cloud file
    """
    name_file = os.path.basename(cloud_file)
    # print(os.path.join(local_dir, name_file), os.path.exists(os.path.join(local_dir, name_file)))
    return not os.path.exists(os.path.join(local_dir, name_file))


def download_dataset(data_dir: str, nr_files: int, dry: bool = False):
    """Download the waymo dataset to local directory, needs gsutil installed.

    Args:
        data_dir (str): path to download to
        nr_files (int): number of files to download, -1 for all
        dry (bool, optional): if to do dry run of commands. Defaults to False.
    """
    # download the whole dataset to directory
    data_parts = [
        "scenario/training",
        "scenario/validation",
        "scenario/testing",
    ]

    # other wise limit number of files to download
    for folder in data_parts:
        local_dir = os.path.join(data_dir, folder)
        os.makedirs(local_dir, exist_ok=True)
        gs_dir = BASE_URI + folder
        if nr_files == -1:
            # do full rsync with directories
            cmd = ["gsutil", "-m", "rsync", "-r", gs_dir, local_dir]
            if dry:
                cmd.insert(3, "-n")

            print("Running command: ", " ".join(cmd))
            subprocess.run(cmd)

        else:
            # list directory files
            output_ls = subprocess.run(["gsutil", "ls", gs_dir], stdout=subprocess.PIPE)
            dir_files = output_ls.stdout.decode().split("\n")
            dir_files = list(filter(lambda x: x != "", dir_files))

            # limit number of files
            files = sorted(dir_files)[:nr_files]
            # print("Files to download: ", files)
            # filter files on already being download
            files = list(filter(partial(check_file_downloaded, local_dir), files))

            if len(files) == 0:
                print("All files already downloaded, skipping.")
                continue
            cmd = ["gsutil", "-m", "cp"] + files + [local_dir]
            if dry:
                print("cmd: ", " ".join(cmd))
            else:
                subprocess.run(cmd)


def download_lidar(data_dir: str, dry: bool = False, all: bool = False):
    """Download the lidar data that is needed for the downloaded training samples.

    Args:
        data_dir (str): path to data dir
        dry (bool, optional): if to run gsuitl command. Defaults to False.
    """
    data_parts = {
        "scenario/training": "lidar/training",
        "scenario/validation": "lidar/validation",
        "scenario/testing": "lidar/testing",
    }

    if all:
        local_dir = os.path.join(data_dir, "lidar")
        gs_dir = os.path.join(BASE_URI, "lidar")
        # gsutil rsync whole lidar directory
        cmd = ["gsutil", "-m", "rsync", "-r", gs_dir, local_dir]
        if dry:
            cmd.insert(3, "-n")

        print("Running command: ", " ".join(cmd))
        subprocess.run(cmd)
    else:
        for folder, lidar_folder in data_parts.items():
            print(f"Downloading lidar {folder}")
            # go over local files and get scenario ids
            local_dir = os.path.join(data_dir, folder)
            ids_scenario = []
            for file in tqdm(os.listdir(local_dir)):
                path_file = os.path.join(local_dir, file)
                dataset = tf.data.TFRecordDataset(path_file, compression_type="")

                for cnt, data in enumerate(dataset):
                    scenario = scenario_pb2.Scenario()
                    scenario.ParseFromString(bytearray(data.numpy()))
                    ids_scenario.append(scenario.scenario_id)

            print("Number Scenarios: ", len(ids_scenario))

            # download lidar files
            lidar_dir = os.path.join(data_dir, lidar_folder)
            os.makedirs(lidar_dir, exist_ok=True)

            files = [os.path.join(BASE_URI, lidar_folder, f"{id_sc}.tfrecord") for id_sc in ids_scenario]
            files = list(filter(partial(check_file_downloaded, lidar_dir), files))
            print("number of files to download:", len(files))

            if len(files) == 0:
                print("All files already downloaded, skipping.")
                continue

            max_download_at_once = 5000
            for i in tqdm(range(len(files) // max_download_at_once + 1)):
                part_files = files[(i * max_download_at_once) : ((i + 1) * max_download_at_once)]
                cmd = ["gsutil", "-m", "cp"] + part_files + [lidar_dir]
                print(cmd[:5])

                if dry:
                    print("cmd:", cmd[:5])
                else:
                    subprocess.run(cmd)


def count_number_files():
    gs_dir = os.path.join(BASE_URI, "lidar", "testing")
    print("counter gloud folder", gs_dir)
    output_ls = subprocess.run(["gsutil", "ls", gs_dir], stdout=subprocess.PIPE)
    dir_files = output_ls.stdout.decode().split("\n")
    print("number of files", len(dir_files))


def main():
    parser = argparse.ArgumentParser(description="To download data from the bucket")
    parser.add_argument(
        "dir",
        type=str,
        default=None,
        help="Directory the data should be downloaded to, will be created if does not exist (data/waymo)",
    )
    parser.add_argument("--dry", default=False, action="store_true", help="Dry run option")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Number of files to download, use -1 for all files")
    parser.add_argument("--lidar", action="store_true", help="Download the lidar data")
    args = parser.parse_args()

    if args.lidar:
        download_lidar(args.dir, dry=args.dry, all=args.limit == -1)
    else:
        download_dataset(args.dir, nr_files=args.limit, dry=args.dry)


if __name__ == "__main__":
    # count_number_files()
    main()
