import sys
sys.path.append('../..')
from utils.common.image_utils import Image, Intersection
import json
from glob import glob
import math
import cv2
from pathlib import Path

class YoloAnnotation:
    def __init__(self):
        pass

    def plot_xcycwh(self, annotation, image):
        for square in annotation:
            xc = square[1] * image.shape[1]
            yc = square[2] * image.shape[0]
            width = square[3] * image.shape[1]
            height = square[4] * image.shape[0]
            pt1 = int(xc - width/2), int(yc + height/2)
            pt2 = int(xc + width/2), int(yc - height/2)
            cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)
        return image
    
    def read_txt_annotation(self, annotation_path):
        with open(annotation_path, 'r') as file:
            annotation = file.readlines()
        annotation = [line.strip().split() for line in annotation]
        annotation = list(map(lambda x: [int(x[0]), *map(float, x[1:])], annotation))
        return annotation
    
    def coordinates_to_yolo(self, coordinates, image_shape, side_in_pixels=20):
        yolo_annotations = []
        for x, y in coordinates:
            xc = x / image_shape[1]
            yc = y / image_shape[0]
            width = side_in_pixels / image_shape[1]
            height = side_in_pixels / image_shape[0]
            yolo_annotations.append([0, xc, yc, width, height])
        return yolo_annotations
    
    def points_annotations_to_yolo(self, points_annotations, size=None, side_in_pixels=20):
        if isinstance(points_annotations, str):
            with open(points_annotations, 'r') as file:
                annotations = json.load(file)
            size = annotations['size']['width'], annotations['size']['height']
            objects = annotations['objects']
            points_annotations = [obj['points']['exterior'][0] for obj in objects]

        yolo_annotations = []
        for point in points_annotations:
            xc = point[0]/size[0]
            yc = point[1]/size[1]
            width = side_in_pixels / size[0]
            height = side_in_pixels / size[1]
            yolo_annotations.append([0, xc, yc, width, height])
        return yolo_annotations
    
    def save_yolo_annotation(self, yolo_annotation, file_name):
        if Path(file_name).parents[0].exists() == False:
            Path(file_name).parents[0].mkdir(parents=True, exist_ok=True)
        with open(file_name, 'w') as txt:
            for annotation in yolo_annotation:
                yclass = annotation[0]
                xc_yc_w_h = ' '.join([str(a) for a in annotation[1:]])
                txt.write(f"{yclass} {xc_yc_w_h}\n")
    
    def xcycwh_to_xyxy(self, annotation, image_shape):
        for rectangle in annotation:
            xc = rectangle[1] * image_shape[1]
            yc = rectangle[2] * image_shape[0]
            width = rectangle[3] * image_shape[1]
            height = rectangle[4] * image_shape[0]
            rectangle[1] = int(xc - width)
            rectangle[2] = int(yc - height)
            rectangle[3] = int(xc + width)
            rectangle[4] = int(yc + height)
        return annotation

    def xyxy_to_xcycwh(self, annotation, image_shape):
        for rectangle in annotation:
            _, x1, y1, x2, y2 = rectangle
            xc = ((x1 + x2) / 2) / image_shape[1]
            yc = ((y1 + y2) / 2) / image_shape[0]
            width = (x2 - x1) / image_shape[1]
            height = (y2 - y1) / image_shape[0]
            rectangle[1] = xc
            rectangle[2] = yc
            rectangle[3] = width
            rectangle[4] = height
        return annotation

    def crop_xyxy_annotation(self, xyxy_crop, annotation, image):
        x1, y1, x2, y2 = xyxy_crop
        
        intersected_annotations = list(filter(
            lambda a: Intersection.overlap_area(xyxy_crop, a[1:]) > 0.6*Intersection.reactangle_area(a[1:]),
            annotation
        ))
        cropped_annotations = []
        for intersected_annotation in intersected_annotations:
            class_ = intersected_annotation[0]
            xc1 = intersected_annotation[1]-x1
            yc1 = intersected_annotation[2]-y1
            xc2 = intersected_annotation[3]-x1
            yc2 = intersected_annotation[4]-y1
            if xc1 < 0: xc1 = 0
            if yc1 < 0: yc1 = 0
            if xc2 > x2-x1: xc2 = x2-x1
            if yc2 > y2-y1: yc2 = y2-y1
            cropped_annotations.append([class_, xc1, yc1, xc2, yc2])
        return cropped_annotations

    
    def grid_crop(
            self, image_path, annotation_path,
            round_floor=True, grid_size=640):
        image = cv2.imread(image_path)
        annotation = self.read_txt_annotation(annotation_path)
        annotation = self.xcycwh_to_xyxy(annotation, image.shape)

        if round_floor: round_fn = math.floor
        else: round_fn = math.ceil

        crop = []
        for y in range(round_fn(image.shape[0] / grid_size)):
            for x in range(round_fn(image.shape[1] / grid_size)):
                x1, y1 = x * grid_size, y * grid_size
                x2, y2 = x1 + grid_size, y1 + grid_size

                cropped_xyxy_annotation = self.crop_xyxy_annotation(
                    [x1, y1, x2, y2], annotation, image)
                cropped_xcycwh_annotation = self.xyxy_to_xcycwh(
                    cropped_xyxy_annotation, (grid_size, grid_size))
                
                #im = self.plot_xcycwh(cropped_xcycwh_annotation, image[y1:y2, x1:x2])
                #Image.show(im, max_res=1000)

                crop.append({
                    'image': image[y1:y2, x1:x2],
                    'xcycwh': cropped_xcycwh_annotation,
                    'index': f"{x}_{y}"
                })
        return crop
    
    def grid_crop_dataset(
            self, images_path, annotations_path, save_path,
            ignore_empty_crop=True, round_floor=True, grid_size=640):
        images_path = glob(f"{images_path}/*")
        annotations_path = glob(f"{annotations_path}/*")

        for image_path, annotation_path in (zip(images_path, annotations_path)):
            cropped = self.grid_crop(image_path, annotation_path, round_floor, grid_size)
            for crop in cropped:
                if ignore_empty_crop and len(crop['xcycwh']) == 0:
                    continue
                filename = Path(image_path).stem
                Image.save( crop['image'], f"{save_path}/images/{filename}_{crop['index']}.jpg")
                self.save_yolo_annotation(crop['xcycwh'], f"{save_path}/labels/{filename}_{crop['index']}.txt")

        
if __name__ == '__main__':
    import os; os.system('cls')
    ya = YoloAnnotation()
    image_path = r'C:\Users\Luiz\Documents\TCC\contagem-ovos-larvas-peixe\images\images'
    annotation_path = r'C:\Users\Luiz\Documents\TCC\contagem-ovos-larvas-peixe\images\labels'
    
    ya.grid_crop_dataset(image_path, annotation_path, f"data\YOLO_cropped_dataset", ignore_empty_crop=False)

    # test crop
    import numpy as np
    while True:
        random_number = np.random.randint(0, 100)
        p1 = glob(r"C:\Users\Luiz\Documents\TCC\contagem-ovos-larvas-peixe\utils\data\YOLO_cropped_dataset\images\*")[random_number]
        p2 = glob(r"C:\Users\Luiz\Documents\TCC\contagem-ovos-larvas-peixe\utils\data\YOLO_cropped_dataset\labels\*")[random_number]
        image = cv2.imread(p1)
        annotation = ya.read_txt_annotation(p2)
        Image.show(ya.plot_xcycwh(annotation, image))