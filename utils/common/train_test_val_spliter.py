from glob import glob
import random
import os
from pathlib import Path
import shutil

class SplitData:
    def __init__(self, train=0.8, test=0.2, validation=0, samples='all'):
        self.train_prop = train
        self.test_prop = test
        self.validation_prop = validation
        self.samples = samples
        self.train_files = []
        self.test_files = []
        self.validation_files = []

    def create_split(self, images_path, annotations_path):
        images_path = glob(images_path)
        annotations_path = glob(annotations_path)
  
        images_and_annotations = []
        for image_path, annotation_path in zip(images_path, annotations_path):
            image_name = Path(image_path).stem
            annotation_name = Path(annotation_path).stem
            assert image_name in annotation_name, f'{image_name} != {annotation_name}'
            images_and_annotations.append((Path(image_path), Path(annotation_path)))

        random.shuffle(images_and_annotations)
        if self.samples != 'all':
            if self.samples > len(images_and_annotations):
                print(f"Limite excedido, usando {len(images_and_annotations)} imagens")
            images_and_annotations = images_and_annotations[:self.samples]

        train_size = round(len(images_and_annotations)*self.train_prop)
        test_size = round(len(images_and_annotations)*self.test_prop)
        validation_size = round(len(images_and_annotations)*self.validation_prop)

        self.train_files = images_and_annotations[:train_size]
        self.test_files = images_and_annotations[train_size:train_size+test_size]
        self.validation_files = images_and_annotations[train_size+test_size:train_size+test_size+validation_size]

        print(f"train: {len(self.train_files)}\ntest: {len(self.test_files)}\nvalidation: {len(self.validation_files)}")
        print(f"total: {len(self.train_files) + len(self.test_files) + len(self.validation_files)}")

    def create_if_not_exists(self, folder):
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'images', 'test'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'labels', 'test'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'labels', 'val'), exist_ok=True)

    def save_split_to_yolo(self, yolo_folder):
        self.create_if_not_exists(yolo_folder)
        join = lambda x: os.path.join(yolo_folder, *x)
        for image_path, annotation_path in self.train_files:
            shutil.copy(image_path, join(['images', 'train']))
            shutil.copy(annotation_path, join(['labels', 'train']))
        for image_path, annotation_path in self.test_files:
            shutil.copy(image_path, join(['images', 'test']))
            shutil.copy(annotation_path, join(['labels', 'test']))
        for image_path, annotation_path in self.validation_files:
            shutil.copy(image_path, join(['images', 'val']))
            shutil.copy(annotation_path, join(['labels', 'val']))

if __name__ == '__main__':
    sd = SplitData(samples='all')
    sd.create_split(
        r'data\YOLO_cropped_dataset\images224\*',
        r'data\YOLO_cropped_dataset\labels\*',
    )
    sd.save_split_to_yolo(r'data\YOLO_cropped_dataset\224_train_test_val_split')