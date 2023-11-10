import json
from pathlib import Path
import shutil
import sys

from matplotlib.ticker import FuncFormatter
from scipy import stats
sys.path.append('../..')
from utils.common.image_utils import Image as im
from utils.common.yolo_utils import YoloAnnotation
#from src.predictor.counter import CounterModel
from dataclasses import dataclass, field, fields
import datetime
import itertools
import os
import random
import contextlib
import matplotlib.pyplot as plt
import time
from timeit import default_timer
import cv2
from glob import glob
import numpy as np
import threading

@contextlib.contextmanager
def timer(message="Time"):
    t0 = default_timer()
    yield
    t1 = default_timer()
    print(f"{message}: {t1 - t0:.2f}s\n")

@dataclass(kw_only=True)
class Args:
    model_name: str
    grid_scale: float
    confiance: float
    resize_scale: float = 0.5
    images_path: str
    annotations_path: str
    random_seed: int = 1010
    data_augmentation: bool = False
    samples: int | str = 'all'
    show_image: bool = False
    verbose: bool = False
    save_path: str = None

@dataclass
class ArgsPermutator:
    permuted: list[dict] = field(default_factory=list)
    def __len__(self):
        return len(self.permuted)
    
    def _permute_args(self, **args):
        _permuted = []
        for k, v in args.items():
            if not isinstance(v, list):
                args[k] = [v]
        for v in itertools.product(*args.values()):
            _permuted.append(dict(zip(args.keys(), v)))
        return _permuted

    def add(self, **kwargs):
        self.permuted += self._permute_args(**kwargs)

    def results(self) -> list[dict]:
        return self.permuted

@dataclass(kw_only=True)
class MetricsGenerator(Args):
    MAE:  list = field(default_factory=list)
    MAPE: list = field(default_factory=list)
    RMSE: list = field(default_factory=list)
    real: list[float] = field(default_factory=list) 
    pred: list[float] = field(default_factory=list) 

    def __post_init__(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.model = lambda x: random.randint(-x//2, x//2) #CounterModel(model_name=self.model_name)

    def set(self, args):
        for key, value in args.items():
            if value == getattr(self, key):
                continue
            setattr(self, key, value)
        self.MAE, self.MAPE, self.RMSE = [], [], []
        self.real, self.pred = [], []

    def save_metrics(self):
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
        
        metrics_dict = {
            'model_name': self.model_name,
            'MAE': self.MAE,
            'MAPE': self.MAPE,
            'RMSE': self.RMSE,
            'pred': self.pred,
            'real': self.real,
            'save_path': f"{self.save_path}/{filename}.json",
            'args': {
                field.name: getattr(self, field.name) 
                for field in fields(Args)
            },
        }

        with open(f"{self.save_path}/{filename}.json", 'w') as f:
           json.dump(metrics_dict, f, indent=4)

    def log_individual_metrics(self, index, error, percentual_error, squared_error):
        print(f"Model: {self.model_name} ({index + 1}/{self.samples})")
        print(f"Absolute Error: {error}")
        print(f"Percentual Error: {percentual_error}")
        print(f"Squared Error: {squared_error}")
        print(f"Real: {self.real[-1]}")
        print(f"Pred: {self.pred[-1]}")
        print(f"|-> MAE: {np.mean(self.MAE)}")
        print(f"|-> MAPE: {np.mean(self.MAPE)}")
        print(f"|-> RMSE: {np.sqrt(np.mean(self.RMSE))}\n")

    def start(self):
        samples = list(zip(glob(self.images_path+'/*'), glob(self.annotations_path+'/*')))
        images_paths, annotations_paths = zip(*samples[:self.samples])
        for index, (image, annotation) in enumerate(zip(images_paths, annotations_paths)):
            #image = cv2.imread(image)            
            #image = im.augment(image) if self.data_augmentation else image
            #image = im.resize(image, self.resize_scale)
            annotation = YoloAnnotation.read_txt_annotation(annotation)

            result = len(annotation) - self.model(x=int(len(annotation)*0.4))

            self.pred.append(result)
            self.real.append(len(annotation))
            
            error = np.abs(len(annotation) - result)
            percentual_error = error / len(annotation) * 100 
            squared_error = np.abs(len(annotation) - result) ** 2

            self.MAE += [error]
            self.MAPE += [percentual_error]
            self.RMSE += [squared_error]

            if self.verbose:
                self.log_individual_metrics(
                    index, error, percentual_error, squared_error)
            if self.show_image:
                im.show(image)

        self.MAE = np.mean(self.MAE)
        self.MAPE = np.mean(self.MAPE)
        self.RMSE = np.sqrt(np.mean(self.RMSE))
        self.save_metrics() if self.save_path else None

@dataclass(kw_only=True)
class ComparisonPool:
    args_permuted: list
    n_workers: int
    shuffle_seed: int = 1010

    def worker(self, paths_dict=None):
        comparator = None
        while self.args_permuted:
            t0 = default_timer()
            with threading.Lock():
                args = self.args_permuted.pop()
            if comparator is None:
                if paths_dict is not None:
                    args |= paths_dict
                comparator = MetricsGenerator(**args)
            else:
                comparator.set(args)
            comparator.start()
            with threading.Lock():
                self.timers_list.append(default_timer() - t0)
                self.remaining_args -= 1
    
    def log_progress(self):
        ...

    def start_threaded(self, paths_dict=None):
        self.start_time = default_timer()
        random.seed(self.shuffle_seed)
        random.shuffle(self.args_permuted)
        self.timers_list = []
        self.total_args = len(self.args_permuted)
        self.remaining_args = self.total_args

        threads = []
        threading.Thread(target=self.log_progress).start()
        for n in range(self.n_workers):
            print(f"Starting worker {n}")
            t = threading.Thread(target=self.worker, args=(paths_dict,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        end_time = default_timer()
        print(f"{'#'*50}\nTotal time: {end_time - self.start_time:.2f}s")

#2612648 ssp uf:ms 

@dataclass(kw_only=True)
class MetricsPlotter:
    train_metrics_path: str = None
    test_metrics_path: str = None
    
    def __post_init__(self):
        self.train_metrics = self.load_metrics(self.train_metrics_path)
        self.test_metrics = self.load_metrics(self.test_metrics_path)

    @staticmethod
    def load_metrics(path):
        metrics_paths = glob(f"{path}/*.json")
        metrics = []
        for metric_path in metrics_paths:
            with open(metric_path, 'r') as f:
                metrics += [json.load(f)]
        return metrics

    @staticmethod
    def boxplot_ax(ax, errors, xlabels, ylabel, fontsize):
        ax.boxplot([*errors], widths=0.5, showfliers=False)
        normal = lambda x: np.random.normal(x, 0.04, len(errors[0]))
        ax.scatter(list(map(normal, range(1, len(errors)+1))), errors, alpha=0.5, s=20) 
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_yticklabels(range(0, 100, 1), fontsize=fontsize)
        ax.set_xticklabels(xlabels, fontsize=fontsize)
        ax.set_yscale('symlog', base=2)
        ax.set_ylim(-1/4)
        ax.grid(True, which='both', linestyle='--', linewidth=0.2)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.16g}'))

    def plot_boxplot_erros(self, fontsize, sort_by, show=False):
        sort_metrics = lambda x: sorted(x, key=lambda x: x[sort_by])
        train = sort_metrics(self.train_metrics)
        test = sort_metrics(self.test_metrics) 
        ylabel = 'Erro Absoluto' if sort_by == 'MAE' else 'Erro Percentual Absoluto'
        fig, ax = plt.subplots(2, 1, figsize=(15, 5))
        MetricsPlotter.boxplot_ax(
            ax[0],
            errors=[np.array(m['real'])-np.array(m['pred']) for m in train],
            xlabels=[f"{m['model_name']}\n({sort_by.upper()}={m[sort_by]:.2f})" for m in train],
            ylabel=f'{ylabel} (Treino)',
            fontsize=fontsize)
        MetricsPlotter.boxplot_ax(
            ax[1],
            errors=[np.array(m['real'])-np.array(m['pred']) for m in test],
            xlabels=[f"{m['model_name']}\n({sort_by.upper()}={m[sort_by]:.2f})" for m in test],
            ylabel=f'{ylabel} (Teste)',
            fontsize=fontsize)
        plt.tight_layout()
        if show: plt.show()

    @staticmethod
    def regression_ax(ax, x, y, xy_max_label, model_name, first, last, fontsize):
        ax.scatter(x, y, alpha=0.5, s=20)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        y_regressed = intercept+slope*x
        ax.plot(x, y_regressed, 'r', c='red')
        ax.text(
            0.07, 0.95, f"r² = {r_value**2:.3f}",
            transform=ax.transAxes,verticalalignment='top',
            horizontalalignment='left', fontsize=fontsize,
            bbox=dict(boxstyle='round', facecolor='#2060FF', alpha=0.15)
        )
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks(np.arange(0, xy_max_label, 80))
        ax.set_yticklabels(np.arange(0, xy_max_label, 80), fontsize=fontsize)
        ax.set_xticks(np.arange(0, xy_max_label, 30))
        ax.set_xticklabels(np.arange(0, xy_max_label, 30), fontsize=fontsize)
        ax.set_xlim(0, xy_max_label)
        ax.set_ylim(-5, xy_max_label)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_ylabel(model_name)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.set_yticks(np.arange(0, xy_max_label, 80))
        ax.set_yticklabels(np.arange(0, xy_max_label, 80), fontsize=10)
        if first:
            ax.set_title(first, fontsize=fontsize)
            ax.xaxis.set_label_position('top')
        if last:
            ax.xaxis.set_label_position('bottom')
            ax.xaxis.set_tick_params(labelbottom=True)

        ax.grid(True, which='both', linestyle='--', linewidth=0.4)

    def plot_regression_real_pred(self, xy_max_label, fontsize, show=False):        
        r2 = lambda x: stats.linregress(x['real'], x['pred'])[2]**2
        train = sorted(self.train_metrics, key=r2, reverse=True)
        test = sorted(self.test_metrics, key=r2, reverse=True)
        fig, ax = plt.subplots(len(train), 2, figsize=(12, 16.3))
        for i, m in enumerate(train):
            MetricsPlotter.regression_ax(
                ax[i][0],
                x=np.array(m['real']), y=np.array(m['pred']),
                xy_max_label=xy_max_label,
                model_name=m['model_name'],
                first='Treino' if i == 0 else None,
                last=i==len(train)-1,
                fontsize=fontsize)
        for i, m in enumerate(test):
            MetricsPlotter.regression_ax(
                ax[i][1],
                x=np.array(m['real']), y=np.array(m['pred']),
                xy_max_label=xy_max_label,
                model_name=m['model_name'],
                first='Teste' if i == 0 else None,
                last=i==len(train)-1,
                fontsize=fontsize)
        plt.tight_layout()
        if show: plt.show()

@dataclass(kw_only=True)
class MetricsComparator(ComparisonPool):
    train_paths_dict: dict
    test_paths_dict: dict
    best_params_path: str = None

    def insert_sum_of_normalization(self, model_metrics):
        mae, mape, rmse = [], [], []
        for model_metric in model_metrics:
            mae += [model_metric['MAE']]
            mape += [model_metric['MAPE']]
            rmse += [model_metric['RMSE']]
        if len(mae) + len(mape) + len(rmse) != 3:
            mae_norm = (mae - np.min(mae)) / (np.max(mae) - np.min(mae))
            mape_norm = (mape - np.min(mape)) / (np.max(mape) - np.min(mape))
            rmse_norm = (rmse - np.min(rmse)) / (np.max(rmse) - np.min(rmse))
            sum_norms = mae_norm + mape_norm + rmse_norm
        else:
            sum_norms = [0]
        for model_metric, sum_norm in zip(model_metrics, sum_norms):
            model_metric['sum_norm'] = sum_norm
        return model_metrics

    def save_best_train_params(self):
        metrics_files = glob(f"{self.train_paths_dict['save_path']}/*.json")
        all_metrics = []
        for metric_file in metrics_files:
            with open(metric_file.replace('.txt', '.json'), 'r') as f:
                all_metrics += [json.load(f)]
        models_names = set([metric['model_name'] for metric in all_metrics])
        
        models_metrics = {}
        for model_name in models_names:
            models_metrics[model_name] = [m for m in all_metrics if m['model_name'] == model_name]
            models_metrics[model_name] = self.insert_sum_of_normalization(models_metrics[model_name])

        os.makedirs(f"{self.train_paths_dict['save_path']}/best", exist_ok=True)
        for key, value in models_metrics.items():
            models_metrics[key] = sorted(value, key=lambda x: x['sum_norm'])
            best = models_metrics[key][0]
            shutil.copy(best['save_path'], f"{self.train_paths_dict['save_path']}/best/{key}.json")
        self.best_params_path = f"{self.train_paths_dict['save_path']}/best"

    def train_metrics(self):
        self.start_threaded(self.train_paths_dict)
    
    def test_metrics(self):
        best_params = glob(f"{self.best_params_path}/*.json")
        for best_param in best_params:
            best_param = json.load(open(best_param, 'r'))['args']
            best_param['images_path'] = self.test_paths_dict['images_path'] 
            best_param['annotations_path'] = self.test_paths_dict['annotations_path']
            best_param['save_path'] = self.test_paths_dict['save_path']
            self.args_permuted += [best_param]
        self.start_threaded()

    def start(self):
        self.train_metrics()
        best_args_folder = self.save_best_train_params()
        self.test_metrics()
        plotter = MetricsPlotter(
            train_metrics_path=self.train_paths_dict['save_path']+f'/best',
            test_metrics_path=self.test_paths_dict['save_path']
        )
        #plotter.plot_boxplot_erros(fontsize=13.5,sort_by='MAE',show=True)
        #plotter.plot_boxplot_erros(fontsize=13.5, sort_by='MAPE', show=True)
        plotter.plot_regression_real_pred(xy_max_label=330, fontsize=13.5, show=True)
     

train_val_130_img = r'..\..\data\datasets\train_val\yolov8_originalres_train=130_val=0\train\images'
train_val_130_ann = r'..\..\data\datasets\train_val\yolov8_originalres_train=130_val=0\train\labels'
test_32_img = r'..\..\data\datasets\test\yolov8_originalres_test=32\test\images'
test_32_ann = r'..\..\data\datasets\test\yolov8_originalres_test=32\test\labels'

save_path = r'..\..\results\params_comparison'


args_permutator = ArgsPermutator()
args_permutator.add(
    model_name=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x', 'yolov8xx'],
    grid_scale=[0.4],
    confiance=[0.4],
    samples=50,
    #verbose=True,
)
# args_permutator.add(
#     model_name=['rtdetr-l'],
#     grid_scale=[0.3, 0.2],
#     confiance=0.5,
# )

MetricsComparator(
    train_paths_dict={
        'images_path': train_val_130_img,
        'annotations_path':train_val_130_ann,
        'save_path': save_path+'/train'
    },
    test_paths_dict={
        'images_path': test_32_img,
        'annotations_path': test_32_ann,
        'save_path': save_path+'/test'
    },
    args_permuted=args_permutator.results(),
    n_workers=1,
    shuffle_seed=1010
).start()
