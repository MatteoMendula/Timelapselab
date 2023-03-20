from .base import BaseBin


class Detector(BaseBin):
    def __init__(self,
                 version: str = '8',
                 size: str = 's',
                 dataset_name: str = 'timelapse'):
        super().__init__(version=version, size=size, dataset_name=dataset_name)
        self.load_best_model()

    def run(self):
        raise NotImplementedError()
