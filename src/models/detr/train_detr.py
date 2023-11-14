from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from transformers import DetrImageProcessor, DetrForObjectDetection, DeformableDetrForObjectDetection, AutoImageProcessor
import torch
import supervision as sv
import random
import cv2
import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

MODEL_PATH = r'../../../data/models/deformable-detr'
#MODEL_PATH = r'../../../data/models/detr-resnet-50' 
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = r'../../../data/datasets/coco/train'
VAL_DIRECTORY = r'../../../data/datasets/coco/valid'
#TEST_DIRECTORY = r'../../../data/datasets/coco/teste'

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, 
        image_directory_path: str, 
        image_processor, 
        train: bool = True
    ):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)        
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target


class Model(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, id2label, train_dataloader, val_dataloader, CHECKPOINT = "facebook/detr-resnet-50"):
        super().__init__()
        if (CHECKPOINT == 'SenseTime/deformable-detr'):
            self.model = DeformableDetrForObjectDetection.from_pretrained(
                pretrained_model_name_or_path=CHECKPOINT, 
                num_labels=len(id2label),
                ignore_mismatched_sizes=True
            )
        else:
            self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT, 
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )
            
        self.TRAIN_DATALOADER = train_dataloader
        self.VAL_DATALOADER = val_dataloader
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
            
        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here: 
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return self.TRAIN_DATALOADER

    def val_dataloader(self):
        return self.VAL_DATALOADER

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer():
    def __init__(self, log_every_n_steps, max_epochs, image_processor, model):
        TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=image_processor, train=True)
        VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=image_processor, train=False)
        TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=self.collate_fn, batch_size=2, shuffle=True)
        VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=self.collate_fn, batch_size=2)
        categories = TRAIN_DATASET.coco.cats
        id2label = {k: v['name'] for k,v in categories.items()}
        self.image_processor = image_processor
        self.model = Model(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, id2label=id2label, train_dataloader=TRAIN_DATALOADER, val_dataloader=VAL_DATALOADER, CHECKPOINT=model)
        early_stop_callback = EarlyStopping(monitor="validation/loss", patience=50)
        checkpoint_callback = ModelCheckpoint(dirpath=MODEL_PATH, 
                                              filename='best-deformable-detr-{epoch:02d}-{val_loss:.2f}',
                                              save_top_k=1, 
                                              monitor="validation/loss",
                                              verbose=True)
        
        self.trainer = Trainer(
            devices=1,
            accelerator="gpu", 
            max_epochs=max_epochs, 
            gradient_clip_val=0.1, 
            log_every_n_steps=log_every_n_steps,
            callbacks=[early_stop_callback, checkpoint_callback])
    
    def train(self, model_path):
        self.trainer.fit(self.model)# ckpt_path=r"C:\Users\Luiz\Documents\TCC\contador-alevinos\data\models\deformable-detr\best-deformable-detr-epoch=06-val_loss=0.00.ckpt")
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
            
            
#MODEL_NAME = 'facebook/detr-resnet-50'
MODEL_NAME = 'SenseTime/deformable-detr'

if (MODEL_NAME == 'SenseTime/deformable-detr'):
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
else:
    image_processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
    
t = ModelTrainer(5, 100, image_processor, MODEL_NAME)
t.train(MODEL_PATH)

#TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor, train=False)
#tester = ModelTester(MODEL_NAME, image_processor)
#tester.test_model(TEST_DATASET, 0.5)
