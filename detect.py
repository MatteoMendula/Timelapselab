from src.bin import Detector
from src.utils import gather_settings


def main():
    settings = gather_settings()
    Detector(version=settings.version,
             size=settings.size,
             dataset_name='timelapse').detect(
        source='datasets/timelapse/test/images',
        imgsz=settings.img_size,
        device=settings.device,
        save=True)


if __name__ == '__main__':
    main()
