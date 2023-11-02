import os
from pathlib import Path
from random import random
import sys
sys.path.append('../..')
from utils.common.image_utils import Image as im
from src.predictor.counter import VideoCounterModel
import cv2
from glob import glob
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class VideoMetrics:
    model_name: str
    grid_scale: float
    resize_scale: float
    confiance: float
    data_augmentation: bool
    skip_frames: int
    show_frames: bool
    video_path: list
    save_path: str | None

    def __post_init__(self):
        self.model = VideoCounterModel(model_name=self.model_name)
        self.video = cv2.VideoCapture(self.video_path)
        self.real_count = int(Path(self.video_path).stem.split('_')[0])
        self.total_frames = cv2.VideoCapture(self.video_path).get(cv2.CAP_PROP_FRAME_COUNT)
        self.results = []
    
    def generate_metrics(self):
        results = self.model.frames_count(
            _id=self.video_path,
            video=self.video,
            grid_scale=self.grid_scale,
            confiance=self.confiance,
            skip_frames=self.skip_frames,
            return_frames=True,
        )
        results_count = [result['total_count'] for result in results]
        MEAN = np.mean(np.array(results_count))
        STD = np.std(np.array(results_count))
        MAE = np.mean(np.abs(np.array(results_count) - self.real_count))
        MAPE = np.mean(np.abs(np.array(results_count) - self.real_count) / self.real_count * 100)
        MSE = np.mean(np.square(np.array(results_count) - self.real_count))

        if self.show_frames:
            for frame_result in results:
                image = im.base64_to_numpy(frame_result['annotated_image'])
                print(f"count: {frame_result['total_count']}")
                im.show(image)

        if self.save_path:
            self.save_metrics(MEAN, STD, MAE, MAPE, MSE)

    
    def save_metrics(self, MEAN, STD, MAE, MAPE, MSE):
        filename_parameters = [
            self.model_name,
            self.grid_scale,
            self.resize_scale,
            self.confiance,
            self.skip_frames,
        ]
        filename_info = ['', 'g', 'r', 'c', 's']
        filename = '_'.join([f"{param}{name}" for name, param in zip(
            filename_info, filename_parameters)])
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
        with open(f"{self.save_path}/{filename}.txt", 'w') as f:
            f.write(f"MAE: {MAE}\n")
            f.write(f"MAPE: {MAPE}\n")
            f.write(f"MSE: {MSE}\n")
            f.write(f"MEAN: {MEAN}\n")
            f.write(f"STD: {STD}\n\n")
            f.write(f"real_count: {self.real_count}\n")
            f.write(f"grid_scale: {self.grid_scale}\n")
            f.write(f"resize_scale: {self.resize_scale}\n")
            f.write(f"confiance: {self.confiance}\n")
            f.write(f"data_augmentation: {self.data_augmentation}\n")
            f.write(f"total_frames: {self.total_frames}\n")
            f.write(f"skip_frames: {self.skip_frames}\n\n")
            f.write(f"video_path: {self.video_path}\n")


video_met = VideoMetrics(
    model_name='rtdetr-l',
    grid_scale=0.35,
    resize_scale=0.5,
    confiance=0.3,
    data_augmentation=False,
    skip_frames=10,
    show_frames=True,
    video_path=r'..\..\data\datasets\videos\dourado\238_20221213_072235.mp4',
    save_path=r'..\..\results\metrics\videos\yolov8n_gside_rscale_conf'
).generate_metrics()

