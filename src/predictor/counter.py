import sys
sys.path.append('../..')
from utils.common.image_utils import Image as im
import torch
from ultralytics import YOLO, RTDETR
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    DeformableDetrForObjectDetection,
    DeformableDetrImageProcessor
)
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ModelsPaths():
    yolov8 = {
        'yolov8n': r'../../data/models/yolov8/yolov8n.pt',
        'yolov8s': r'../../data/models/yolov8/yolov8s.pt',
        'yolov8m': r'../../data/models/yolov8/yolov8m.pt',
        'yolov8l': r'../../data/models/yolov8/yolov8l.pt',
        'yolov8x': r'../../data/models/yolov8/yolov8s.pt',    
    }
    detr = {
        'detr-resnet-50': r'../../data/models/detr/detr-resnet-50',
    }
    deformable_detr = {
        'deformable-detr': r'../../data/models/deformable-detr/deformable-detr',
    }
    rtdetr = {
        'rtdetr-l': r'../../data/models/rtdetr/rtdetr-l.pt',
        'rtdetr-x': r'../../data/models/rtdetr/rtdetr-x.pt',
    }

@dataclass
class Yolov8Model():
    yolov8_name: str
    model: YOLO = None
    device: str = '0' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.yolov8_name not in ModelsPaths.yolov8.keys():
            raise Exception(f"Model '{self.yolov8_name}' not found.")
        self.yolo_path = ModelsPaths.yolov8[self.yolov8_name]
        assert Path(self.yolo_path).exists()
        self.model = YOLO(self.yolo_path)

    def predict(self, image, confiance):
        results = self.model.predict(
            source=image, 
            conf=confiance,
            show=False, 
            verbose=False,
            device=self.device)
        return results
    
    def get_xyxy_boxes(self, predicted):
        boxes = predicted[0].boxes
        xyxy_boxes = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            xyxy_boxes.append(list(map(int, xyxy)))
        return xyxy_boxes

@dataclass
class RTDetrModel():
    rtdetr_name: str
    model: RTDETR = None
    device: str = '0' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        if self.rtdetr_name not in ModelsPaths.rtdetr.keys():
            raise Exception(f"Model '{self.rtdetr_name}' not found.")
        self.rtdetr_path = ModelsPaths.rtdetr[self.rtdetr_name]
        assert Path(self.rtdetr_path).exists()
        self.model = RTDETR(self.rtdetr_path)

    def predict(self, image, confiance):
        results = self.model.predict(
            source=image, 
            conf=confiance,
            show=False, 
            verbose=False,
            device=self.device)
        return results
    
    def get_xyxy_boxes(self, predicted):
        boxes = predicted[0].boxes
        xyxy_boxes = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            xyxy_boxes.append(list(map(int, xyxy)))
        return xyxy_boxes

@dataclass
class DetrModel:
    detr_name: str
    model: DetrForObjectDetection = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        if self.detr_name not in ModelsPaths.detr.keys():
            raise Exception(f"Model '{self.detr_name}' not found.")
        self.detr_path = ModelsPaths.detr[self.detr_name]
        assert Path(self.detr_path).exists()
        self.model = DetrForObjectDetection.from_pretrained(
            self.detr_path).to(self.device)
    
    def predict(self, image, confiance):
        pretrained_model_name = "facebook/detr-resnet-50"
        image_processor = DetrImageProcessor.from_pretrained(pretrained_model_name)
        with torch.no_grad():
            inputs = image_processor(images=image, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            results = image_processor.post_process_object_detection(
                outputs=outputs, 
                threshold=confiance, 
                target_sizes=torch.tensor([image.shape[:2]]).to(self.device))
            return results
    
    def get_xyxy_boxes(self, predicted):
        xyxy_boxes = predicted[0]['boxes'].to('cpu').detach().numpy()
        xyxy_boxes = list(map(lambda x: list(map(int, x)), xyxy_boxes))
        return xyxy_boxes

@dataclass
class DeformableDetrModel:
    deformable_detr_name: str
    model: DeformableDetrForObjectDetection = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        if self.deformable_detr_name not in ModelsPaths.deformable_detr.keys():
            raise Exception(f"Model '{self.deformable_detr_name}' not found.")
        self.deformable_detr_path = ModelsPaths.deformable_detr[self.deformable_detr_name]
        assert Path(self.deformable_detr_path).exists()
        self.model = DeformableDetrForObjectDetection.from_pretrained(
            self.deformable_detr_path).to(self.device)
    
    def predict(self, image, confiance):
        pretrained_model_name = "SenseTime/deformable-detr"
        image_processor = DeformableDetrImageProcessor.from_pretrained(pretrained_model_name)
        with torch.no_grad():
            inputs = image_processor(images=image, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            results = image_processor.post_process_object_detection(
                outputs=outputs, 
                threshold=confiance, 
                target_sizes=torch.tensor([image.shape[:2]]).to(self.device))
            return results
        
    def get_xyxy_boxes(self, predicted):
        xyxy_boxes = predicted[0]['boxes'].to('cpu').detach().numpy()
        xyxy_boxes = list(map(lambda x: list(map(int, x)), xyxy_boxes))
        return xyxy_boxes

@dataclass
class Model():
    name: str

    def __post_init__(self):
        if self.name in ModelsPaths.yolov8.keys():
            self.model = Yolov8Model(self.name)
        elif self.name in ModelsPaths.rtdetr.keys():
            self.model = RTDetrModel(self.name)
        elif self.name in ModelsPaths.detr.keys():
            self.model = DetrModel(self.name)
        elif self.name in ModelsPaths.deformable_detr.keys():
            self.model = DeformableDetrModel(self.name)
        else:
            raise Exception(f"Model '{self.name}' not found.")

    def predict(self, image, confiance):
        return self.model.predict(image, confiance)

    def get_xyxy_boxes(self, predicted):
        return self.model.get_xyxy_boxes(predicted)

@dataclass
class CounterModel():
    model_name: str

    def __post_init__(self):
        self.model = Model(self.model_name.lower())
    
    def grid_split(self, image, grid_size):
        H = int(np.ceil(image.shape[0] / grid_size))
        W = int(np.ceil(image.shape[1] / grid_size))
        grid = []
        for y in range(H):
            for x in range(W):
                x1, y1 = x * grid_size, y * grid_size
                x2, y2 = x1 + grid_size, y1 + grid_size
                grid.append({
                    'grid_xyxy': [x1, y1, x2, y2],
                    'grid_index': [x, y]
                })
        return grid
    
    def expand_image_to_grid(self, image, grid_size):
        H_grid = int(np.ceil(image.shape[0] / grid_size))
        W_grid = int(np.ceil(image.shape[1] / grid_size))
        full_grid_image = np.zeros((H_grid * grid_size, W_grid * grid_size, 3), dtype=np.uint8)
        full_grid_image[:image.shape[0], :image.shape[1]] = image
        return full_grid_image

    def annotate_grid(self, grid_results, image):
        for grid in grid_results:
            if len(grid['grid_boxes_xyxy']) == 0:
                continue
            x0_img, y0_img = grid['grid_xyxy'][:2]
            xyxy = np.array(grid['grid_boxes_xyxy']) + np.array([x0_img, y0_img, x0_img, y0_img])
            for x1, y1, x2, y2 in xyxy:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        return im.numpy_to_base64(image)

    def count(self, _id, image, grid_scale, confiance, return_image):
        image = im.base64_to_numpy(image)
        original_shape = image.shape
        grid_size = int(min(image.shape[:2]) * grid_scale)

        image = self.expand_image_to_grid(image, grid_size)
        grid_results = self.grid_split(image, grid_size)

        for grid in grid_results:
            gx1, gy1, gx2, gy2 = grid['grid_xyxy']
            results = self.model.predict(
                image=image[gy1:gy2, gx1:gx2], 
                confiance=confiance,
            )
            grid['grid_boxes_xyxy'] = self.model.get_xyxy_boxes(results)

        if return_image:
            image = image[:original_shape[0], :original_shape[1]]
            image = self.annotate_grid(grid_results, image)
        else:
            image = None
        
        response = {
            '_id': _id,
            'grid_scale': grid_scale,
            'confiance': confiance,
            'total_count': sum([len(grid['grid_boxes_xyxy']) for grid in grid_results]),
            'grid_results': grid_results,
            'annotated_image': image
        }
        return response

@dataclass
class VideoCounterModel(CounterModel):
    def frames_count(self, _id, video, grid_scale, confiance, skip_frames, return_frames):
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []
        for frame_index in (range(total_frames)):
            ret, frame = video.read()
            if not ret:
                break
            if frame_index % skip_frames != 0:
                continue
            response = self.count(
                _id=_id,
                image=im.numpy_to_base64(frame),
                grid_scale=grid_scale,
                confiance=confiance,
                return_image=return_frames
            )
            results.append(response)
        video.release()
        response = {
            '_id': _id,
            'total_frames': total_frames,
            'skip_frames': skip_frames,
            'results': results
        }
        return results
