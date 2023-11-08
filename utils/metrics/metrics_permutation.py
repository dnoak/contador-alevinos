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
import threading

train_val_130_img = r'..\..\data\datasets\train_val\yolov8_originalres_train=130_val=0\train\images'
train_val_130_ann = r'..\..\data\datasets\train_val\yolov8_originalres_train=130_val=0\train\labels'
test_32_img = r'..\..\data\datasets\test\yolov8_originalres_test=32\test\images'
test_32_ann = r'..\..\data\datasets\test\yolov8_originalres_test=32\test\labels'

args = {
    'model_name': ['detr-resnet-50'],
    'grid_scale': [0.3],
    'resize_scale': [0.5],
    'confiance': [0.85],
    'data_augmentation': [False],
    'random_seed': [1011],
    'samples': ['all'],
    'images_folder': [test_32_img],
    'annotations_folder': [test_32_ann], 
    'show_image': [False],
    'verbose': [True],
}

max_g, min_g = args['grid_scale'][-1], args['grid_scale'][0]
max_r, min_r = args['resize_scale'][-1], args['resize_scale'][0]
max_c, min_c = args['confiance'][-1], args['confiance'][0]
samples = args['samples'][0] if args['samples'][0] != 'all' else \
    len(glob(f"{args['images_folder'][0]}/*"))
folder_name  = f"({'-'.join(args['model_name'])})"
folder_name += f"(g={min_g}-{max_g})(r={min_r}-{max_r})(c={min_c}-{max_c})"
folder_name += f"(s={samples})"
folder_name += f"(seed={args['random_seed'][0]})"

args['save_path'] = [rf"../../results/metrics/{folder_name}"]
args['samples'] = [samples]

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
    verbose: bool
    save_path: str

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
    
    def save_metrics(self, MAE, MAPE, MSE, pred, real):
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
            f.write(f"MAE: {np.mean(MAE)} {MAE}\n")
            f.write(f"MAPE: {np.mean(MAPE)} {MAPE}\n")
            f.write(f"RMSE: {np.sqrt(np.mean(MSE))} {MSE}\n\n")
            f.write(f"pred: {sum(pred)} {pred}\n")
            f.write(f"real: {sum(real)} {real}\n\n")

            f.write(f"grid_scale: {self.grid_scale}\n")
            f.write(f"resize_scale: {self.resize_scale}\n")
            f.write(f"confiance: {self.confiance}\n\n")
            f.write(f"random_seed: {self.random_seed}\n")
            f.write(f"data_augmentation: {self.data_augmentation}\n")
            f.write(f"samples: {self.samples}\n")
    
    def generate_metrics(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        MAE, MAPE, MSE = [], [], []
        real, pred = [], []
        for count, (image_path, annotation_path) in enumerate(
            self.images_and_annotations):
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
            pred.append(result['total_count'])
            real.append(len(annotations))

            if self.verbose:
                print(f"{'-'*20}")
                print(f"< {count+1}/{len(self.images_and_annotations)} >")
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
                im.show_pillow(image_annotated)
                input()

        x, y = zip(*sorted(zip(real, MAPE), key=lambda x: x[0]))

        if self.save_path:
            self.save_metrics(MAE, MAPE, MSE, pred, real)

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
        last_len = self.remaining_args
        while self.remaining_args:
            if last_len == self.remaining_args:
                time.sleep(0.02)
                continue
            last_len = self.remaining_args

            elapsed_time = datetime.timedelta(seconds=default_timer()-self.start_time)
            if not self.timers_list:
                continue
            average_time = np.mean(self.timers_list) / self.n_workers
            remaing_time = datetime.timedelta(seconds=average_time*self.remaining_args)
            print(f"Comparisons: {self.total_args - self.remaining_args}/{self.total_args}")
            print(f"Average time: {average_time}s")
            print(f"Elapsed time: {elapsed_time}")
            print(f"Remaining time: {remaing_time}\n")

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
