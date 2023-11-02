from torch.utils.data import DataLoader
from Classes.coco_detection import CocoDetection
from Classes.models import Detr
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import supervision as sv
import random
import cv2
import matplotlib.pyplot as plt
import os

MODEL_PATH = r'../../models/detr_2' 
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join('Dataset', "train")
VAL_DIRECTORY = os.path.join('Dataset', "valid")
TEST_DIRECTORY = os.path.join('Dataset', "test")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer():
    def __init__(self, log_every_n_steps, max_epochs, image_processor, model):
        TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=image_processor, train=True)
        VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=image_processor, train=False)
        TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=self.collate_fn, batch_size=12, shuffle=True)
        VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=self.collate_fn, batch_size=12)
        categories = TRAIN_DATASET.coco.cats
        id2label = {k: v['name'] for k,v in categories.items()}
        self.image_processor = image_processor
        self.model = Detr(lr=1e-5, lr_backbone=1e-5, weight_decay=1e-4, id2label=id2label, train_dataloader=TRAIN_DATALOADER, val_dataloader=VAL_DATALOADER)
        early_stop_callback = EarlyStopping(monitor="validation/loss", patience=10)
        
        self.trainer = Trainer(
            devices=1,
            accelerator="gpu", 
            max_epochs=max_epochs, 
            gradient_clip_val=0.1, 
            accumulate_grad_batches=8, 
            log_every_n_steps=log_every_n_steps,
            callbacks=[early_stop_callback])
    
    def train(self, model_path):
        self.trainer.fit(self.model)# , ckpt_path=r'D:\Documentos\Projetos\tcc\contagem-ovos-larvas-peixe\lightning_logs\version_15\checkpoints\epoch=41-step=804.ckpt')
        self.model.model.save_pretrained(model_path)
    
    def collate_fn(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }

class ModelTester():
    def __init__(self, model_name, image_processor):
        self.model = self.get_model(model_name)
        self.image_processor = image_processor
        self.model.to(DEVICE)
    
    def get_model(self, model_name):
        if model_name == 'SenseTime/deformable-detr':
            return DeformableDetrForObjectDetection.from_pretrained(MODEL_PATH)
        elif model_name == 'facebook/detr-resnet-50':
            return DetrForObjectDetection.from_pretrained(MODEL_PATH)
    
    def test_model(self, DATASET, threshold):
        detections_output = None
        categories = DATASET.coco.cats
        id2label = {k: v['name'] for k,v in categories.items()}
        box_annotator = sv.BoxAnnotator()
        image_ids = DATASET.coco.getImgIds()
        for i in range(10):
            image_id = random.choice(image_ids)
            image_ids.remove(image_id)
            # load image and annotatons 
            image = DATASET.coco.loadImgs(image_id)[0]
            annotations = DATASET.coco.imgToAnns[image_id]
            image_path = os.path.join(DATASET.root, image['file_name'])
            image = cv2.imread(image_path)
            detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
            labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]

            # Annotate detections
            with torch.no_grad():
                # load image and predict
                inputs = self.image_processor(images=image, return_tensors='pt').to(DEVICE)
                outputs = self.model(**inputs)
                # post-process
                target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
                results = self.image_processor.post_process_object_detection(
                    outputs=outputs, 
                    threshold=threshold, 
                    target_sizes=target_sizes
                )[0]
                detections_output = sv.Detections.from_transformers(transformers_results=results)
                labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections_output]
                frame_detections = box_annotator.annotate(scene=image.copy(), detections=detections_output, labels=labels)
            plt.imshow(cv2.cvtColor(frame_detections, cv2.COLOR_BGR2RGB))
            plt.title(f'Detections - {len(detections_output)} - Real {len(detections_output)} - Image {image_id}')
            plt.show()
            
            
MODEL_NAME = 'facebook/detr-resnet-50'
image_processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
t = ModelTrainer(5, 400, image_processor, MODEL_NAME)
t.train(MODEL_PATH)

#TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor, train=False)
#tester = ModelTester(MODEL_NAME, image_processor)
#tester.test_model(TEST_DATASET, 0.5)