from .paths import *
from .frameworks import *

import argparse
from pathlib import Path


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

    parser.add_argument('--inference_mode', type=str, default='torch',
                        help='deeplearning framework to use for inference')

    parser.add_argument('--tracking_method', type=str, default='bytetrack',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking_source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--conf_thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max_det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--show_vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save_crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save_trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save_vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes_to_track', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/track', help='save results to project/name')
    parser.add_argument('--exist_ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line_thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide_labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide_class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid_stride', type=int, default=1, help='video frame-rate stride')

    parser.add_argument('--timer_mode', type=str, default='minutes',
                        help='track every minute, hour, day or week')

    opt = parser.parse_args()
    return opt
