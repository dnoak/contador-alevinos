import sys
sys.path.append('../..')
from utils.common.image_utils import Image as im
from pathlib import Path
import random
import time
from dataclasses import dataclass
import requests
import contextlib
from timeit import default_timer
import cv2
from glob import glob

@contextlib.contextmanager
def timer(message="Time"):
    t0 = default_timer()
    yield
    t1 = default_timer()
    print(f"{message}: {t1 - t0:.2f}s\n")

def send_request(url, post):
    response = requests.post(url=url, json=post)
    if response.status_code != 200:
        print(f"Error <{url.split('/')[-1]}>:", response.status_code)
        print(response.text)
        return
    return response

def get_request(url):
    response = requests.get(url=url)
    if response.status_code != 200:
        print(f"Error <{url.split('/')[-1]}>:", response.status_code)
        print(response.text)
        return
    return response

def test_api_route_set_params(model_name):
    params_response = send_request(
        url='http://127.0.0.1:3000/set-params',
        post={'model_name': model_name}
    )
    if model_name == params_response.json()['model_name']:
        print(f"Model <{model_name}> setted successfully\n")
    else:
        print(f"Error setting model <{model_name}>")
        print(f"Default model setted: {params_response.json()['model_name']}\n")

@dataclass
class TestApiContadorAlevinos:
    base_url: str
    random_params: list[dict]
    random_images_path: list[str]
    random_grid_scales: list[float]
    random_confiances: list[float]

    def _params_randomizer(self):
        return random.choice(self.random_params)

    def _data_randomizer(self, samples, return_image=False):
        data = []
        for _ in range(samples):
            random.seed(random.randint(0, 1000))
            image_path = random.choice(self.random_images_path)
            image_base64 = im.numpy_to_base64(cv2.imread(image_path))
            data += [{
                    "_id": Path(image_path).stem,
                    "image": image_base64,
                    "grid_scale": random.choice(self.random_grid_scales),
                    "confiance": random.choice(self.random_confiances),
                    "return_image": return_image
                }]
        return data
    
    def results_validator(func):
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            # precisa garantir que os resultados estão 
            # na mesma ordem que os dados enviados
            for result, data in zip(results['results'], kwargs['data']):
                assert result['_id'] == data['_id']
                assert result['grid_scale'] == data['grid_scale']
                assert result['confiance'] == data['confiance']
                assert isinstance(result['total_count'], int)
                if data['return_image']:
                    assert im.base64_to_numpy(result['annotated_image']) is not None
                else:
                    assert result['annotated_image'] is None 
                for r in result['grid_results']:
                    assert all([isinstance(i, int) for i in r['grid_xyxy']])
                    assert all([isinstance(i, int) for i in r['grid_index']])
            print(f"<Test> {func.__name__} passed successfully")
            return results
        return wrapper
    
    def params_validator(func):
        def wrapper(*args, **kwargs):
            params = func(*args, **kwargs)
            max_tries = 3
            for tries in range(max_tries):
                time.sleep(10)
                try:
                    get_params_response = get_request(
                        url='http://127.0.0.1:3000/get-params')
                    assert params == get_params_response.json()
                    print(f"<Test> {func.__name__} passed successfully")
                    break
                except:
                    print(f"<Test> {func.__name__} failed, tries: {tries+1}/{max_tries}")
                    pass
            else:
                raise Exception(f"Error setting params: {params}")
            return params
        return wrapper
    
    @params_validator
    def _base_test_api_route_post_set_params(self, params):
        params_response = send_request(
            url='http://127.0.0.1:3000/set-params',
            post=params
        )
        return params_response.json()

    @results_validator
    def _base_test_api_route_post_contador_alevinos(self, data):
        model_response = send_request(
            url='http://127.0.0.1:3000/contador-alevinos',
            post=data)
        return model_response.json()
    
    def test_set_params(self):
        params = self._params_randomizer()
        return self._base_test_api_route_post_set_params(params=params)

    def test_one_image(self, return_image=False):
        data = self._data_randomizer(samples=1, return_image=return_image)
        return self._base_test_api_route_post_contador_alevinos(data=data)

    def test_many_images(self, return_image=False):
        data = self._data_randomizer(samples=3, return_image=return_image)
        return self._base_test_api_route_post_contador_alevinos(data=data)

    def test_one_image_many_times(self, times=3):
        for _ in range(times):
            self.test_one_image()
    
    def test_visual_one_image(self):
        results = self.test_one_image(return_image=True)
        for result in results['results']:
            image = im.base64_to_numpy(result['annotated_image'])
            im.show(image)
    
    def test_visual_many_images(self):
        results = self.test_many_images(return_image=True)
        for result in results['results']:
            image = im.base64_to_numpy(result['annotated_image'])
            im.show(image)

    def test_all(self):
        self.test_set_params()
        self.test_one_image()
        self.test_many_images()
        self.test_one_image_many_times()
    

tester = TestApiContadorAlevinos(
    base_url='',
    random_params=[
        # {'model_name': 'yolov8n'},
        # {'model_name': 'yolov8s'},
        # {'model_name': 'yolov8m'},
        # {'model_name': 'yolov8l'},
        # {'model_name': 'yolov8x'},
        {'model_name': 'yolov8n'},
        {'model_name': 'detr'},
        {'model_name': 'rtdetr-l'},
    ],
    random_images_path=glob(r'..\..\data\datasets\train\yolov8_640x640_train=2689_val=676\valid\images\*'),
    random_grid_scales=[0.35],
    random_confiances=[0.5]
)

for _ in range(10):
    tester.test_all()

