import sys
sys.path.append('../..')
from utils.common.image_utils import Image as im
from utils.common.yolo_utils import YoloAnnotation
import json
from pathlib import Path
import shutil
from matplotlib.ticker import FuncFormatter
from scipy import stats
from src.predictor.counter import CounterModel
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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
        #self.fake_model = lambda x, n: x-int(random.randint(-int(x/len(n))//2, int(x/len(n))//2))
        self.model = CounterModel(model_name=self.model_name)

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
            'args': {field.name: getattr(self, field.name) for field in fields(Args)}
        }

        with open(f"{self.save_path}/{filename}.json", 'w') as f:
           json.dump(metrics_dict, f, indent=4)

    def log_individual_metrics(self, index, error, percentual_error, squared_error):
        print(f"Model: {self.model_name} ({index + 1}/{self.samples})")
        print(f"Absolute Error: {error}")
        print(f"Percentual Error: {percentual_error:.2f}")
        print(f"Squared Error: {squared_error}")
        print(f"Real: {self.real[-1]}")
        print(f"Pred: {self.pred[-1]}")
        print(f"|-> MAE: {np.mean([self.MAE])}")
        print(f"|-> MAPE: {np.mean([self.MAPE])}")
        print(f"|-> RMSE: {np.sqrt(np.mean([self.RMSE]))}\n")

    def start(self):
        paths_sample = list(zip(glob(self.images_path+'/*'), glob(self.annotations_path+'/*')))
        if self.samples != 'all':
            paths_sample = random.sample(paths_sample, self.samples)
        images_paths, annotations_paths = zip(*paths_sample)
        for index, (image, annotation) in enumerate(zip(images_paths, annotations_paths)):
            annotation = YoloAnnotation.read_txt_annotation(annotation)
            image = cv2.imread(image)            
            image = im.augment(image) if self.data_augmentation else image
            image = im.resize(image, self.resize_scale)
            pred = self.model.count(
                _id='dummy_id',
                image=im.numpy_to_base64(image),
                grid_scale=self.grid_scale, 
                confiance=self.confiance, 
                return_image=self.show_image
            )
            #pred = self.fake_model(len(annotation), self.model_name)
            
            real = len(annotation)
            
            self.pred.append(pred['total_count'])
            self.real.append(real)
            
            error = np.abs(len(annotation) - pred['total_count'])
            percentual_error = error / len(annotation) * 100 
            squared_error = np.abs(len(annotation) - pred['total_count']) ** 2

            self.MAE += [error]
            self.MAPE += [percentual_error]
            self.RMSE += [squared_error]

            if self.verbose:
                self.log_individual_metrics(
                    index, error, percentual_error, squared_error)
            if self.show_image:
                im.show_pillow(im.base64_to_numpy(pred["annotated_image"]))
                input()

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
        while self.args_permuted:
            t0 = default_timer()
            with threading.Lock():
                args = self.args_permuted.pop()
            if paths_dict is not None:
                args |= paths_dict
            comparator = MetricsGenerator(**args)
            comparator.start()
            with threading.Lock():
                self.timers_list.append(default_timer() - t0)
                self.remaining_args -= 1
    
    def log_progress(self):
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

@dataclass(kw_only=True)
class MetricsPlotter:
    train_metrics_path: str = None
    test_metrics_path: str = None
    save_path: str = None
    
    def __post_init__(self):
        self.train_metrics = MetricsComparator.load_metrics(self.train_metrics_path)
        self.test_metrics = MetricsComparator.load_metrics(self.test_metrics_path)

    @staticmethod
    def boxplot_ax(ax, errors, xlabels, ylabel, fontsize):
        ax.boxplot([*errors], widths=0.5, showfliers=False)
        normal = lambda x: np.random.normal(x, 0.06, len(errors[0]))
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
        mae_fn = lambda r, p: np.abs(np.array(r)-np.array(p))
        mape_fn = lambda r, p: mae_fn(r, p) / np.array(r) * 100  
        error_fn = mae_fn if sort_by == 'MAE' else mape_fn
        fig, ax = plt.subplots(2, 1, figsize=(18, 10))
        MetricsPlotter.boxplot_ax(
            ax[0],
            errors=[error_fn(m['real'], m['pred']) for m in train],
            xlabels=[f"{m['model_name']}\n({sort_by.upper()}={m[sort_by]:.2f})" for m in train],
            ylabel=f'{ylabel} (Treino)',
            fontsize=fontsize)
        MetricsPlotter.boxplot_ax(
            ax[1],
            errors=[error_fn(m['real'], m['pred']) for m in test],
            xlabels=[f"{m['model_name']}\n({sort_by.upper()}={m[sort_by]:.2f})" for m in test],
            ylabel=f'{ylabel} (Teste)',
            fontsize=fontsize)
        plt.tight_layout()
        if show: plt.show()
        os.makedirs(self.save_path, exist_ok=True)
        plt.savefig(f"{self.save_path}/boxplot_{sort_by}_train_test.png")

    @staticmethod
    def regression_ax(ax, x, y, xy_max_label, model_name, first, last, fontsize):
        ax.scatter(x, y, alpha=0.5, s=20)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        y_regressed = intercept+slope*x
        ax.plot(x, y_regressed, 'r')
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
                ax[i][0] if len(train) > 1 else ax[0],
                x=np.array(m['real']), y=np.array(m['pred']),
                xy_max_label=xy_max_label,
                model_name=m['model_name'],
                first='Treino' if i == 0 else None,
                last=i==len(train)-1,
                fontsize=fontsize)
        for i, m in enumerate(test):
            MetricsPlotter.regression_ax(
                ax[i][1] if len(train) > 1 else ax[1],
                x=np.array(m['real']), y=np.array(m['pred']),
                xy_max_label=xy_max_label,
                model_name=m['model_name'],
                first='Teste' if i == 0 else None,
                last=i==len(train)-1,
                fontsize=fontsize)
        plt.tight_layout()
        if show: plt.show()
        os.makedirs(self.save_path, exist_ok=True)
        plt.savefig(f"{self.save_path}/regression_train_test.png")

@dataclass(kw_only=True)
class MetricsComparator(ComparisonPool):
    save_path: str
    train_paths_dict: dict
    test_paths_dict: dict
    best_params_path: str = None

    def __post_init__(self):
        # assert not os.path.exists(self.save_path)
        runs = glob(f"{self.save_path}/*")
        self.save_path = f"{self.save_path}/run_{len(runs)}"
        
        self.train_paths_dict['save_path'] = self.save_path + '/train'
        self.test_paths_dict['save_path'] = self.save_path + '/test'

    @staticmethod
    def load_metrics(path):
        metrics_paths = glob(f"{path}/*.json")
        metrics = []
        for metric_path in metrics_paths:
            with open(metric_path, 'r') as f:
                metrics += [json.load(f)]
        return metrics

    def order_by_sum_of_normalization(self, unsorted):
        if len(unsorted) == 1:
            return unsorted
        mae, mape, rmse = [], [], []
        for metrics in unsorted:
            mae += [metrics['MAE']]
            mape += [metrics['MAPE']]
            rmse += [metrics['RMSE']]
        mae, mape, rmse = np.array(mae), np.array(mape), np.array(rmse)
        mae_norm = (mae - np.min(mae)) / (np.max(mae) - np.min(mae))
        mape_norm = (mape - np.min(mape)) / (np.max(mape) - np.min(mape))
        rmse_norm = (rmse - np.min(rmse)) / (np.max(rmse) - np.min(rmse))
        sum_norms = mae_norm + mape_norm + rmse_norm

        indexes = np.argsort(sum_norms)
        return [unsorted[i] for i in indexes]

    def save_best_train_params(self):
        all_metrics = self.load_metrics(self.train_paths_dict['save_path'])
        metrics_ordered = self.order_by_sum_of_normalization(all_metrics)
        metrics_by_model = {}
        for metric in metrics_ordered:
            if metric['model_name'] not in metrics_by_model:
                metrics_by_model[metric['model_name']] = []
            metrics_by_model[metric['model_name']] += [metric]

        os.makedirs(f"{self.train_paths_dict['save_path']}/best", exist_ok=True)
        best_models_ordered = {}
        for model, models_dict in metrics_by_model.items():
            best = models_dict[0]['save_path']
            best_models_ordered[model] = models_dict[0]
            shutil.copy(best, f"{self.train_paths_dict['save_path']}/best/{model}.json")
        
        os.makedirs(f"{self.train_paths_dict['save_path']}/best/ordered_models", exist_ok=True)
        with open(f"{self.train_paths_dict['save_path']}/best/ordered_models/models.json", 'w') as f:
            json.dump(best_models_ordered, f, indent=4)
        self.best_params_path = f"{self.train_paths_dict['save_path']}/best"

    def save_test_best_models(self):
        all_models = self.load_metrics(self.test_paths_dict['save_path'])
        ordered_models = self.order_by_sum_of_normalization(all_models)
        os.makedirs(f"{self.test_paths_dict['save_path']}/ordered_models", exist_ok=True)
        with open(f"{self.test_paths_dict['save_path']}/ordered_models/models.json", 'w') as f:
            json.dump(ordered_models, f, indent=4)
    
    def train_metrics(self):
        self.start_threaded(self.train_paths_dict)
        self.save_best_train_params()
    
    def test_metrics(self):
        best_params = glob(f"{self.best_params_path}/*.json")
        for best_param in best_params:
            best_param = json.load(open(best_param, 'r'))['args']
            best_param['images_path'] = self.test_paths_dict['images_path']
            best_param['annotations_path'] = self.test_paths_dict['annotations_path']
            best_param['save_path'] = self.test_paths_dict['save_path']
            self.args_permuted += [best_param]
        self.start_threaded()
        self.save_test_best_models()

    def start(self):
        self.train_metrics()
        input()
        self.test_metrics()
        plotter = MetricsPlotter(
            train_metrics_path=self.train_paths_dict['save_path']+f'/best',
            test_metrics_path=self.test_paths_dict['save_path'],
            save_path=Path(self.test_paths_dict['save_path']).parent / 'images'
        )
        plotter.plot_boxplot_erros(fontsize=13.5,sort_by='MAE',show=False)
        plotter.plot_boxplot_erros(fontsize=13.5, sort_by='MAPE', show=False)
        plotter.plot_regression_real_pred(xy_max_label=330, fontsize=13.5, show=False)

train_val_130_img = r'..\..\data\datasets\train_val\yolov8_originalres_train=130_val=0\train\images'
train_val_130_ann = r'..\..\data\datasets\train_val\yolov8_originalres_train=130_val=0\train\labels'
test_32_img = r'..\..\data\datasets\test\yolov8_originalres_test=32\test\images'
test_32_ann = r'..\..\data\datasets\test\yolov8_originalres_test=32\test\labels'
#train_val_130_img = r'C:\Users\Luiz\Documents\TCC\contador-alevinos\data\datasets\teste\train\images'
#train_val_130_ann = r'C:\Users\Luiz\Documents\TCC\contador-alevinos\data\datasets\teste\train\labels'
#test_32_img       = r'..\..\data\datasets\full_yolo\test\images'
#test_32_ann       = r'..\..\data\datasets\full_yolo\test\labels'

save_path = r'..\..\results\params_comparison'

args_permutator = ArgsPermutator()

# args_permutator.add(
#     model_name=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
#     grid_scale=[0.2, 0.3, 0.4, 0.5],
#     confiance=[0.4, 0.45, 0.5, 0.55, 0.60],
#     samples='all',
#     data_augmentation=False,
#     verbose=False,
#     show_image=False
# )

# args_permutator.add(
#     model_name=["rtdetr-l", "rtdetr-x", 'detr-resnet-50'],
#     grid_scale=[0.2, 0.3, 0.4, 0.5],
#     confiance=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#     samples='all',
#     data_augmentation=False,
#     verbose=False,
#     show_image=False
# )
# args_permutator.add(
#     model_name=['deformable-detr'],
#     grid_scale=[0.2, 0.3, 0.4, 0.5],
#     confiance=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
#     samples='all',
#     data_augmentation=True,
#     verbose=False,
#     show_image=False
# )

args_permutator.add(
    model_name=['rtdetr-x'],
    grid_scale=[0.5],
    confiance=[0.5],
    resize_scale=1,
    samples=50,
    data_augmentation=False,
    verbose=True,
    show_image=True
)

MetricsComparator(
    save_path=save_path,
    train_paths_dict={
        'images_path': train_val_130_img,
        'annotations_path':train_val_130_ann,
    },
    test_paths_dict={
        'images_path': test_32_img,
        'annotations_path': test_32_ann,
    },
    args_permuted=args_permutator.results(),
    n_workers=1,
    shuffle_seed=1010
).start()
