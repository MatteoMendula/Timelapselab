from .paths import *

import argparse


def gather_settings():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    # parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--version', type=str, default='8', help='yolo version')
    parser.add_argument('--size', type=str, default='s', help='yolo size')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='w,h image sizes')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    opt = parser.parse_args()
    return opt
