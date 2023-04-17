from src.bin import Trainer
from src.utils import gather_settings
import torch

import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:200'


def main():
    settings = gather_settings()
    Trainer(version=settings.version,
            size=settings.size,
            dataset_path=settings.data).run(batch_size=settings.batch_size,
                                            img_size=settings.img_size,
                                            epochs=settings.epochs,
                                            device=settings.device)


if __name__ == '__main__':
    print("matte", torch.cuda.is_available())
    print("matte 2", torch.cuda.device_count())
    main()
