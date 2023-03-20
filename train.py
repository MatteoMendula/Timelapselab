from src.bin import Trainer
from src.utils import gather_settings


def main():
    settings = gather_settings()
    Trainer(version=settings.version,
            size=settings.size,
            dataset=settings.data).run(batch_size=settings.batch_size,
                                       img_size=settings.img_size,
                                       epochs=settings.epochs,
                                       device=settings.device)


if __name__ == '__main__':
    main()
