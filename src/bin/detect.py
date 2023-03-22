import copy
import cv2
from .base import BaseBin


class Detector(BaseBin):
    def __init__(self,
                 version: str = '8',
                 size: str = 's',
                 dataset_name: str = 'timelapse'):
        super().__init__(version=version, size=size, dataset_name=dataset_name)
        self.name = '{}/yolov{}{}'.format(self.dataset_name, self.version, self.size)
        self.load_best_model()
        self.model.add_callback("on_predict_postprocess_end", Detector.on_predict_postprocess_end)

    def detect(self, **args):
        return self.model(**args)

    @staticmethod
    def on_predict_postprocess_end(predictor):
        _, _, initial_images, _, _ = predictor.batch
        initial_images = initial_images if isinstance(initial_images, list) else [initial_images]
        for index, initial_image in enumerate(initial_images):
            blurred_image = Detector.blur_faces(initial_image,
                                                predictor.results[index])
            # print(predictor.results[index].orig_img)
            # print(blurred_image)
            predictor.results[index].orig_img = blurred_image
            predictor.results[index].blurred_image = blurred_image
            # print(predictor.results[index])

    @staticmethod
    def blur_faces(initial_image, yolo_results):
        boxes = yolo_results.boxes
        probs = yolo_results.probs
        image = copy.deepcopy(initial_image)
        for d in reversed(boxes):
            cls = d.cls.squeeze()
            c = int(cls)
            label = f'{yolo_results.names[c]}' if yolo_results.names else f'{c}'
            if label == 'Face':
                image = Detector.blur_face(image, d.xywh.squeeze().cpu().detach().numpy())
        return image

    @staticmethod
    def blur_face(image, face_box):
        h, w = image.shape[:2]
        kernel_width = (w // 7) | 1
        kernel_height = (h // 7) | 1
        x, y, w, h = int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3])
        face_roi = image[y:y + h, x:x + w]
        blurred_face = cv2.GaussianBlur(face_roi, (kernel_width, kernel_height), 0)
        image[y:y + h, x:x + w] = blurred_face
        return image
