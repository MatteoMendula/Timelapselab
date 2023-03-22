import copy
import os
import platform
from pathlib import Path

import cv2
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from ultralytics.yolo.data.utils import VID_FORMATS
from ultralytics.yolo.utils import LOGGER, colorstr, yaml_load
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import select_device

from src.trackers.multi_tracker_zoo import create_tracker
from .base import BaseBin
from .detect import Detector

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class Tracker(BaseBin):
    def __init__(self,
                 version: str = '8',
                 size: str = 's',
                 dataset_name: str = 'timelapse',
                 tracking_method: str = 'strongsort',
                 reid_weights: str = 'reid_models/pretrained/osnet_x0_25_msmt17.pt',
                 device: str = '0',
                 half: bool = False,  # use FP16 half-precision inference
                 dnn: bool = False,  # use OpenCV DNN for ONNX inference
                 imgsz: tuple[int] = (640, 640),
                 classes_to_track: list[int] = None,
                 conf_thres: float = 0.2,  # confidence threshold
                 iou_thres: float = 0.6,  # NMS IOU threshold
                 max_det: int = 1000,  # maximum detections per image
                 agnostic_nms: bool = False,  # class-agnostic NMS
                 show_vid: bool = False,
                 nosave: bool = False,
                 save_txt: bool = False):
        super().__init__(version=version, size=size, dataset_name=dataset_name)

        self.yolo_model = None
        self.save_dir = None
        self.detector = Detector(version=version, size=size, dataset_name=dataset_name)

        self.device = select_device(device)
        self.dnn = dnn
        self.half = half
        self.pt = None
        self.class_names = None
        self.stride = None
        self.load_detection_model()
        self.imgsz = check_imgsz(imgsz, stride=self.stride)

        self.txt_path = None
        self.vid_writer = None
        self.vid_path = None
        self.dataset = None
        self.bs = None
        self.show_vid = show_vid
        self.vid_stride = None

        self.tracking_method = tracking_method
        self.outputs = None
        self.tracker_list = None
        self.tracking_config = None
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms

        # Tracking parameters
        self.annotator = None
        self.nosave = nosave
        self.save_txt = save_txt
        self.source = None
        self.webcam = None
        self.is_url = None
        self.is_file = None
        self.save_img = None

        # Variables for running tracking
        self.seen = None
        self.windows = None
        self.dt = None
        self.curr_frames = None
        self.prev_frames = None

        self.this_file_path = Path(__file__).resolve()
        self.root = self.this_file_path.parents[2]
        print('\n\nRoot: {}\n\n'.format(self.root))
        self.reid_weights = Path(os.path.join(self.root, reid_weights))
        if not os.path.exists(self.reid_weights.parents[0]):
            os.makedirs(self.reid_weights.parents[0])
        # self.reid_weights = reid_weights
        if classes_to_track is not None:
            self.classes_to_track = classes_to_track
        else:
            data_cfg_file = self.root / 'data' / '{}.yaml'.format(self.dataset_name)
            print(yaml_load(data_cfg_file))
            self.classes_to_track = [i for i in range(yaml_load(data_cfg_file)['nc'])]
        self.tracked_objects_dict = None
        self.define_tracked_dict()

    def define_tracked_dict(self):
        self.tracked_objects_dict = {}
        for class_to_track in self.classes_to_track:
            self.tracked_objects_dict[class_to_track] = []

    def reset_tracked_dict(self):
        for class_to_track in self.classes_to_track:
            self.tracked_objects_dict[class_to_track] = []

    def set_source(self, source):
        self.source = str(source)
        self.save_img = not self.nosave and not self.source.endswith('.txt')  # save inference images
        self.is_file = Path(self.source).suffix[1:] in (VID_FORMATS)
        self.is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (self.is_url and not self.is_file)
        if self.is_url and self.is_file:
            self.source = check_file(self.source)  # download

    def load_detection_model(self):
        # Load model
        self.yolo_model = AutoBackend(self.detector.yolo_weights_path, device=self.device, dnn=self.dnn, fp16=self.half)
        # self.yolo_model = AutoBackend('yolo_models/pretrained/yolov8s.pt', device=self.device, dnn=self.dnn,
        #                               fp16=self.half)
        self.stride, self.class_names, self.pt = self.yolo_model.stride, self.yolo_model.names, self.yolo_model.pt
        # model = self.detector.model
        # stride, class_names, pt = model.model.stride, model.model.class_names, model.model.model.pt

    def setup_stream(self, vid_stride: int = 1):
        # Dataloader
        self.vid_stride = vid_stride
        self.bs = 1
        if self.webcam:
            self.show_vid = True if check_imshow(warn=True) and self.show_vid else False
            self.dataset = LoadStreams(
                self.source,
                imgsz=self.imgsz,
                stride=self.stride,
                auto=self.pt,
                transforms=getattr(self.yolo_model.model, 'transforms', None),
                vid_stride=self.vid_stride
            )
            self.bs = len(self.dataset)
        else:
            self.dataset = LoadImages(
                self.source,
                imgsz=self.imgsz,
                stride=self.stride,
                auto=self.pt,
                transforms=getattr(self.yolo_model.model, 'transforms', None),
                vid_stride=self.vid_stride
            )
        self.vid_path, self.vid_writer, self.txt_path = [None] * self.bs, [None] * self.bs, [None] * self.bs

    def warmup_detection_model(self):
        self.yolo_model.warmup(imgsz=(1 if self.pt or self.yolo_model.triton else self.bs, 3, *self.imgsz))  # warmup

    def setup_tracker(self):
        # Create as many strong sort instances as there are video sources
        self.tracking_config = 'src/trackers/{}/configs/{}.yaml'.format(self.tracking_method, self.tracking_method)
        self.tracker_list = []
        for pred_index in range(self.bs):
            tracker = create_tracker(self.tracking_method, self.tracking_config, self.reid_weights, self.device,
                                     self.half)
            self.tracker_list.append(tracker, )
            if hasattr(self.tracker_list[pred_index], 'model'):
                if hasattr(self.tracker_list[pred_index].model, 'warmup'):
                    self.tracker_list[pred_index].model.warmup()
        self.outputs = [None] * self.bs

    @torch.no_grad()
    def run(self,
            source='0',
            save_crop=False,  # save cropped prediction boxes
            save_trajectories=False,  # save trajectories for each track
            save_vid=False,  # save confidences in --save-txt labels
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            project='runs/track',  # save results to project/name
            line_thickness=2,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            hide_class=False,  # hide IDs
            vid_stride=1,  # video frame-rate stride
            ):
        # Define source
        self.set_source(source)
        # Define experiments folder
        exp_name = '{}/{}'.format(self.detector.name, self.tracking_method)
        self.save_dir = increment_path(Path(project) / exp_name, exist_ok=False)  # increment run
        (self.save_dir / 'tracks' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # Setup detection model if not already setup
        if self.yolo_model is None:
            self.load_detection_model()

        self.setup_stream(vid_stride=vid_stride)
        self.warmup_detection_model()

        self.setup_tracker()

        self._track(save_vid=save_vid,
                    save_crop=save_crop,
                    save_trajectories=save_trajectories,
                    augment=augment,
                    visualize=visualize,
                    line_thickness=line_thickness,
                    hide_labels=hide_labels,
                    hide_conf=hide_conf,
                    hide_class=hide_class, )

        self.print_results(save_vid=save_vid)

    def _track(self,
               save_crop: bool = False,  # save cropped prediction boxes
               save_vid: bool = False,  # save confidences in --save-txt labels
               save_trajectories: bool = False,
               augment: bool = False,  # augmented inference
               visualize: bool = False,  # visualize features
               line_thickness: int = 2,  # bounding box thickness (pixels)
               hide_labels: bool = False,  # hide labels
               hide_conf: bool = False,  # hide confidences
               hide_class: bool = False,  # hide IDs
               ):
        # Run tracking
        self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile(), Profile())
        self.curr_frames, self.prev_frames = [None] * self.bs, [None] * self.bs
        for frame_idx, batch in enumerate(self.dataset):
            path, im, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
            with self.dt[0]:
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.half else im.float()  # uint8 to fp16/32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
                preds = self.yolo_model(im, augment=augment, visualize=visualize)

            # Apply NMS
            with self.dt[2]:
                pred = non_max_suppression(preds, self.conf_thres, self.iou_thres,
                                           self.classes_to_track, self.agnostic_nms, max_det=self.max_det)
            print(f'pred: {pred}')

            # Process detections
            for pred_index, det in enumerate(pred):  # detections per image
                self.seen += 1
                if self.webcam:  # bs >= 1
                    p, im0, _ = path[pred_index], im0s[pred_index].copy(), self.dataset.count
                    p = Path(p)  # to Path
                    s += f'{pred_index}: '
                    txt_file_name = p.name
                    save_path = str(self.save_dir / p.name)  # im.jpg, vid.mp4, ...
                else:
                    p, im0, _ = path, im0s.copy(), getattr(self.dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    # video file
                    if self.source.endswith(VID_FORMATS):
                        txt_file_name = p.stem
                        save_path = str(self.save_dir / p.name)  # im.jpg, vid.mp4, ...
                    # folder with imgs
                    else:
                        txt_file_name = p.parent.name  # get folder name containing current img
                        save_path = str(self.save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
                self.curr_frames[pred_index] = im0

                txt_path = str(self.save_dir / 'tracks' / txt_file_name)  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                imc = im0.copy() if save_crop else im0  # for save_crop

                # annotator = Annotator(im0, line_width=line_thickness, example=str(class_names))

                if hasattr(self.tracker_list[pred_index], 'tracker') and \
                        hasattr(self.tracker_list[pred_index].tracker, 'camera_update'):
                    if self.prev_frames[pred_index] is not None and \
                            self.curr_frames[pred_index] is not None:  # camera motion compensation
                        self.tracker_list[pred_index].tracker.camera_update(self.prev_frames[pred_index],
                                                                            self.curr_frames[pred_index])

                if det is not None and len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                    # Print results
                    for pred_class in det[:, 5].unique():
                        n = (det[:, 5] == pred_class).sum()  # detections per class
                        s += f"{n} {self.class_names[int(pred_class)]}{'s' * (n > 1)}, "  # add to string

                blurred_image = Tracker.blur_faces(imc, det)
                self.annotator = Annotator(blurred_image, line_width=line_thickness, example=str(self.class_names))

                if det is not None and len(det):
                    # pass detections to strongsort
                    with self.dt[3]:
                        self.outputs[pred_index] = self.tracker_list[pred_index].update(det.cpu(), im0)

                    print(f'\n\noutputs[pred_index]: {self.outputs[pred_index]}\n\n')

                    self.draw_tracked_objects_on_frame(pred_index=pred_index,
                                                       frame=blurred_image,
                                                       frame_idx=frame_idx,
                                                       path=path,
                                                       txt_file_name=txt_file_name,
                                                       txt_path=txt_path,
                                                       p=p,
                                                       save_vid=save_vid,
                                                       save_crop=save_crop,
                                                       save_trajectories=save_trajectories,
                                                       hide_labels=hide_labels,
                                                       hide_conf=hide_conf,
                                                       hide_class=hide_class, )

                else:
                    pass
                    # tracker_list[pred_index].tracker.pred_n_update_all_tracks()

                # Stream results
                im0 = self.annotator.result()
                print(f'self.show_vid: {self.show_vid}')
                if self.show_vid:
                    self.show_video(frame=im0, path=p)

                # Save results (image with detections)
                if save_vid:
                    self.save_video(frame=im0,
                                    index=pred_index,
                                    vid_cap=vid_cap,
                                    save_path=save_path)

                self.prev_frames[pred_index] = self.curr_frames[pred_index]

            # Print total time (preprocessing + inference + NMS + tracking)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}"
                        f"{sum([dt.dt for dt in self.dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

    def print_results(self, save_vid: bool = False):
        # Print results
        t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {self.tracking_method} update '
            f'per image at shape {(1, 3, *self.imgsz)}' % t)
        if self.save_txt or save_vid:
            s = f"\n{len(list((self.save_dir / 'tracks').glob('*.txt')))} tracks" \
                f" saved to {self.save_dir / 'tracks'}" if self.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

    def draw_tracked_objects_on_frame(self,
                                      pred_index,
                                      frame,
                                      frame_idx,
                                      path,
                                      txt_file_name,
                                      txt_path,
                                      p,
                                      save_vid: bool = False,
                                      save_crop: bool = False,
                                      save_trajectories: bool = False,
                                      hide_labels: bool = False,
                                      hide_conf: bool = False,
                                      hide_class: bool = False, ):
        # draw boxes for visualization
        if len(self.outputs[pred_index]) > 0:

            for j, (output) in enumerate(self.outputs[pred_index]):

                bbox, object_id, cls, conf = output[0:4], output[4], output[5], output[6]

                self.tracked_objects_dict[int(cls)].append(object_id)

                if self.save_txt:
                    Tracker.save_mot_to_txt(output=output,
                                            frame_idx=frame_idx,
                                            object_id=object_id,
                                            pred_index=pred_index,
                                            txt_path=txt_path)

                if save_vid or save_crop or self.show_vid:  # Add bbox/seg to image
                    pred_class, color = self.annotate_frame(bbox,
                                                            cls,
                                                            object_id,
                                                            conf,
                                                            hide_labels,
                                                            hide_class,
                                                            hide_conf)

                    if save_trajectories and self.tracking_method == 'strongsort':
                        q = output[7]
                        self.tracker_list[pred_index].trajectory(frame, q, color=color)
                    if save_crop:
                        Tracker.save_cropped_objects_to_files(image=frame,
                                                              bbox=bbox,
                                                              pred_class=pred_class,
                                                              object_id=object_id,
                                                              class_names=self.class_names,
                                                              txt_file_name=txt_file_name,
                                                              path=path,
                                                              save_dir=self.save_dir,
                                                              p=p)

    @staticmethod
    def blur_faces(initial_image, predictions):
        boxes = predictions[:, :4]
        classes = predictions[:, -1]
        image = copy.deepcopy(initial_image)
        for index, box in enumerate(boxes):
            cls = int(classes[index])
            if cls == 2:
                image = Detector.blur_face(image, box.squeeze().cpu().detach().numpy())
        return image

    @staticmethod
    def save_mot_to_txt(output, frame_idx, object_id, pred_index, txt_path):
        # to MOT format
        bbox_left = output[0]
        bbox_top = output[1]
        bbox_w = output[2] - output[0]
        bbox_h = output[3] - output[1]
        # Write MOT compliant results to file
        print(f'Logging text file to {txt_path}...')
        with open(txt_path + '.txt', 'a') as f:
            f.write(('%g ' * 10 + '\n') % (frame_idx + 1, object_id, bbox_left,  # MOT format
                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, pred_index))

    def annotate_frame(self, bbox, cls, object_id, conf, hide_labels, hide_class, hide_conf):
        pred_class = int(cls)  # integer class
        object_id = int(object_id)  # integer object_id
        if hide_labels:
            label = None
        elif hide_conf and not hide_class:
            label = f'{object_id} {self.class_names[pred_class]}'
        elif hide_class and not hide_conf:
            label = f'{object_id} {conf:.2f}'
        else:
            label = f'{object_id} {self.class_names[pred_class]} {conf:.2f}'
        color = colors(pred_class, True)
        self.annotator.box_label(bbox, label, color=color)
        return pred_class, color

    def show_video(self, frame, path):
        if platform.system() == 'Linux' and path not in self.windows:
            self.windows.append(path)
            cv2.namedWindow(str(path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(path), frame.shape[1], frame.shape[0])
        cv2.imshow(str(path), frame)
        if cv2.waitKey(1) == ord('q'):  # 1 millisecond
            exit()

    def save_video(self, frame, index, vid_cap, save_path):
        if self.vid_path[index] != save_path:  # new video
            self.vid_path[index] = save_path
            if isinstance(self.vid_writer[index], cv2.VideoWriter):
                self.vid_writer[index].release()  # release previous video writer
            if vid_cap:  # video
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # stream
                fps, w, h = 30, frame.shape[1], frame.shape[0]
            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            self.vid_writer[index] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        self.vid_writer[index].write(frame)

    @staticmethod
    def save_cropped_objects_to_files(image, bbox, pred_class, object_id, class_names, txt_file_name, path, save_dir,
                                      p):
        txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
        print(bbox)
        # print(torch.tensor(bbox))
        try:
            save_one_box(torch.tensor(bbox), image,
                         file=save_dir / 'crops' / txt_file_name / class_names[
                             pred_class] / f'{object_id}' / f'{p.stem}.jpg', BGR=True)
        except TypeError:
            save_one_box(torch.tensor([int(b) for b in bbox]), image,
                         file=save_dir / 'crops' / txt_file_name / class_names[
                             pred_class] / f'{object_id}' / f'{p.stem}.jpg', BGR=True)
