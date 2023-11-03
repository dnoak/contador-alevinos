import sys
sys.path.append('../..')
from utils.common.image_utils import Image as im
from utils.common.yolo_utils import YoloAnnotation
from src.predictor.counter import CounterModel
from dataclasses import dataclass
import datetime
import itertools
import os
import random
import contextlib
import time
from timeit import default_timer
import cv2
from glob import glob
import numpy as np
from tqdm import tqdm
import threading

args = {
    'model_name': ['detr-resnet-50', 'deformable-detr', 'rtdetr-l', 'rtdetr-x'],
    'grid_scale': [0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    'resize_scale': [0.5],
    'confiance': [0.2, 0.3, 0.5, 0.75, 0.8, 0.85, 0.9],
    'data_augmentation': [False],
    'random_seed': [1011],
    'samples': [13],
    'images_folder': [r'..\..\data\datasets\train\yolov8_originalres_train=130_val=0\train\images'],
    'annotations_folder': [r'..\..\data\datasets\train\yolov8_originalres_train=130_val=0\train\labels'],
    'show_image': [False],
    'save_path':[r'../../results/metrics/images/detr-ddetr-rtdetrl-rtdetrx-(samples=13)'],
    'verbose': [False],
}

@contextlib.contextmanager
def timer(message="Time"):
    t0 = default_timer()
    yield
    t1 = default_timer()
    print(f"{message}: {t1 - t0:.2f}s\n")

@dataclass
class MetricsComparison:
    model_name: str
    grid_scale: float
    resize_scale: float
    confiance: float
    data_augmentation: bool
    random_seed: int
    samples: int | str
    images_folder: str
    annotations_folder: str
    show_image: bool
    save_path: str
    verbose: bool

    def __post_init__(self):
        self.model = CounterModel(model_name=self.model_name)
        self.images_path = glob(f"{self.images_folder}/*")
        self.annotations_path = glob(f"{self.annotations_folder}/*")
        self.images_and_annotations = [i for i in zip(self.images_path, self.annotations_path)]
        if self.samples != 'all':
            self.images_and_annotations = random.sample(self.images_and_annotations, self.samples)
        
    def set(self, args):
        for key, value in args.items():
            if value == getattr(self, key):
                continue
            setattr(self, key, value)
    
    def save_metrics(self, MAE, MAPE, MSE, total_pred, total_real):
        filename_parameters = [
            self.model_name, 
            self.grid_scale, 
            self.resize_scale, 
            self.confiance, 
            self.samples
        ]
        filename_info = ['', 'g', 'r', 'c', 's']
        filename = '_'.join([f"{param}{name}" for name, param in zip(
            filename_info, filename_parameters)])
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        with open(f"{self.save_path}/{filename}.txt", 'w') as f:
            f.write(f"MAE: {np.mean(MAE)}\n")
            f.write(f"MAPE: {np.mean(MAPE)}\n")
            f.write(f"RMSE: {np.sqrt(np.mean(MSE))}\n")
            f.write(f"grid_scale: {self.grid_scale}\n")
            f.write(f"resize_scale: {self.resize_scale}\n")
            f.write(f"confiance: {self.confiance}\n\n")
            f.write(f"random_seed: {self.random_seed}\n")
            f.write(f"data_augmentation: {self.data_augmentation}\n")
            f.write(f"samples: {self.samples}\n")
            f.write(f"total_pred: {total_pred}\n")
            f.write(f"total_real: {total_real}\n")
        
    def generate_metrics(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        MAE, MAPE, MSE = [], [], []
        total_pred, total_real = 0, 0
        for image_path, annotation_path in self.images_and_annotations:
            image = cv2.imread(image_path)
            image = im.augment(image) if self.data_augmentation else image
            image = im.resize(image, self.resize_scale)
            
            image_base64 = im.numpy_to_base64(image)

            result = self.model.count(
                _id='dummy_id',
                image=image_base64,
                grid_scale=self.grid_scale, 
                confiance=self.confiance, 
                return_image=self.show_image
            )
            annotations = YoloAnnotation().read_txt_annotation(annotation_path)
            
            result_MAE = np.abs(len(annotations) - result['total_count'])
            result_MAPE = result_MAE / len(annotations) * 100
            result_MSE = np.square(len(annotations) - result['total_count'])
            MAE.append(result_MAE)
            MAPE.append(result_MAPE)
            MSE.append(result_MSE)
            total_pred += result['total_count']
            total_real += len(annotations)

            if self.verbose:
                print(f"{'-'*20}")
                print(f"Model: {self.model_name}")
                print(f"MAE:  {result_MAE:.2f}  | acc: {np.mean(MAE):.2f}")
                print(f"MAPE: {result_MAPE:.2f}% | acc: {np.mean(MAPE):.2f}%")
                print(f"MSE:  {result_MSE:.2f}  | acc: {np.sqrt(np.mean(MSE)):.2f}")
                print(f"predito: {result['total_count']}")
                print(f"real: {len(annotations)}")
                print(f"grid_scale: {self.grid_scale}")
                print(f"resize_scale: {self.resize_scale}")
                print(f"confiance: {self.confiance}\n{'-'*20}\n")
            
            if self.show_image:
                image_annotated = im.base64_to_numpy(result['annotated_image'])
                im.show(image_annotated)
        
        if self.save_path:
            self.save_metrics(MAE, MAPE, MSE, total_pred, total_real)

@dataclass
class MetricsComparisonPool:
    n_workers: int
    args: dict
    shuffle_seed: int = 1010

    def __post_init__(self):
        permuted = list(itertools.product(*self.args.values()))
        self.args_permuted = [dict(zip(self.args.keys(), i)) for i in permuted]
        random.seed(self.shuffle_seed)
        random.shuffle(self.args_permuted)
        self.start_time = default_timer()
        self.timers_list = []
        self.total_args = len(self.args_permuted)
        self.remaining_args = self.total_args

    def worker(self):
        comparator = None
        while self.args_permuted:
            t0 = default_timer()
            with threading.Lock():
                args = self.args_permuted.pop()
            if comparator is None:
                try:
                    comparator = MetricsComparison(**args)
                except Exception as e:
                    print(f"{'#'*15} ERROR ON ARGS: {'#'*15}")
                    for key, value in args.items():
                        print(f"{key}: {value}")
                    raise e
            else:
                comparator.set(args)
            comparator.generate_metrics()

            t1 = default_timer()
            with threading.Lock():
                self.timers_list.append(t1 - t0)
                self.remaining_args -= 1

    def log(self):
        last_update = self.remaining_args
        while self.args_permuted:
            current_update = self.remaining_args
            if last_update == current_update:
                time.sleep(0.1)
                continue
            last_update = current_update
            elapsed_time = default_timer() - self.start_time
            average_time = np.mean(self.timers_list) / self.n_workers
            remaing_time = average_time * self.remaining_args
            
            print(f"Comparisons: {self.total_args - self.remaining_args}/{self.total_args}")
            print(f"Average time: {average_time:.2f}s")
            print(f"Elapsed time: {datetime.timedelta(seconds=elapsed_time)}")
            print(f"Remaining time: {datetime.timedelta(seconds=remaing_time)}\n")

    def start_no_threaded(self):
        threading.Thread(target=self.log).start()
        self.n_workers = 1
        self.worker()

    def start(self):
        threads = []
        threading.Thread(target=self.log).start()
        for _ in range(self.n_workers):
            t = threading.Thread(target=self.worker)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        end_time = default_timer()
        print(f"Total time: {end_time - self.start_time:.2f}s")


if __name__ == '__main__':
    pool = MetricsComparisonPool(
        n_workers=16,
        args=args,
    ).start()


