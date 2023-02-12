import argparse
import os
import pickle
import random
import time

import numpy as np
import torch
import sys

sys.path.append(".")
from src import config
from src.NICE_SLAM import NICE_SLAM


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--times', type=int, default=100)
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    args = parser.parse_args()

    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')

    file_num = len([f for f in os.listdir(f"GPUoffload_test/saved_obs/")])
    assert file_num > 0
    n_steps = args.times
    model = config.get_model(cfg, nice=args.nice)
    decoders = model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    slam = NICE_SLAM(cfg, args)
    renderer = slam.renderer
    total_input_size = total_running_time = total_inf_time = 0

    for i in range(n_steps):
        time_ckp_0 = time.time()
        with open(f"GPUoffload_test/saved_obs/{i - i % file_num}/c.pkl", 'rb') as file:
            c = pickle.load(file)
        rays_d = torch.load(f"GPUoffload_test/saved_obs/{i - i % file_num}/rays_d.pth")
        rays_o = torch.load(f"GPUoffload_test/saved_obs/{i - i % file_num}/rays_o.pth")
        with open(f"GPUoffload_test/saved_obs/{i - i % file_num}/stage.pkl", 'rb') as file:
            stage = pickle.load(file)
        total_input_size += (sys.getsizeof(c) + sys.getsizeof(rays_d) + sys.getsizeof(rays_o) + sys.getsizeof(stage))
        time_ckp_1 = time.time()
        renderer.render_batch_ray(c, decoders, rays_d, rays_o, device, stage, None)
        time_ckp_2 = time.time()
        total_inf_time += (time_ckp_2 - time_ckp_1)
        total_running_time += (time_ckp_2 - time_ckp_0)
        print(f"processing {i + 1} / {n_steps}")

    print(f"Average input size: {total_input_size / n_steps} byte,",
          f"Average running time: {total_running_time / n_steps}, "
          f"Average inference time: {total_inf_time / n_steps}")


if __name__ == '__main__':
    main()
