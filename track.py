from src.bin import Tracker
from src.utils import gather_settings


def main():
    settings = gather_settings()
    Tracker(version=settings.version,
            size=settings.size,
            dataset_name='timelapse',
            reid_weights='reid_models/pretrained/osnet_x0_25_msmt17.pt',
            tracking_method=settings.tracking_method,
            classes_to_track=settings.classes,
            imgsz=settings.img_size,
            conf_thres=settings.conf_thres,
            iou_thres=settings.iou_thres,
            max_det=settings.max_det,
            device=settings.device,
            nosave=settings.nosave,
            agnostic_nms=settings.agnostic_nms,
            half=settings.half,
            dnn=settings.dnn,
            show_vid=settings.show_vid,
            save_txt=settings.save_txt, ).run(source=settings.tracking_source,
                                              save_crop=settings.save_crop,
                                              save_trajectories=settings.save_trajectories,
                                              save_vid=settings.save_vid,
                                              augment=settings.augment,
                                              visualize=settings.visualize,
                                              project=settings.project,
                                              line_thickness=settings.line_thickness,
                                              hide_labels=settings.hide_labels,
                                              hide_conf=settings.hide_conf,
                                              hide_class=settings.hide_class,
                                              vid_stride=settings.vid_stride)


if __name__ == '__main__':
    main()
