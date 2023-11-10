import sys
sys.path.append('..')
from utils.common.image_utils import Image as im
from src.predictor.counter import CounterModel
from glob import glob
from pathlib import Path
import cv2
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

images_paths = glob('NCAUF_samples/*.jpg')
random.shuffle(images_paths)
csvs_paths = glob('NCAUF_samples/*.csv')
counter_model = CounterModel('yolov8n')

MAE = []
MAPE = []
RMSE = []
for image_path in tqdm(images_paths):
    image = cv2.imread(image_path)
    print(image.shape)
    image = im.resize(image, 10)
    image_id = Path(image_path).stem.split('_')[1]
    csv_path = [i for i in csvs_paths if image_id in i][0]
    coords = pd.read_csv(csv_path).to_numpy()
    print(len(coords))
    counting = counter_model.count(
        _id=image_path,
        image=im.numpy_to_base64(image),
        grid_scale=1,
        confiance=0.25,
        return_image=True,
    )
    print(counting['total_count'])
    counted_image = im.base64_to_numpy(counting['annotated_image'])
    im.show(counted_image)
    MAE.append(np.abs(counting['total_count'] - len(coords)))
    MAPE.append(MAE[-1] / len(coords) * 100)
    RMSE.append(np.square(counting['total_count'] - len(coords)))

print(f"MAE: {np.mean(MAE)}")
print(f"MAPE: {np.mean(MAPE)}")
print(f"RMSE: {np.sqrt(np.mean(RMSE))}")

    
    #for coord in coords:
    #    cv2.circle(image, (int(coord[0]), int(coord[1])), 1, (0, 0, 255), 2)
    #im.show(image)

    