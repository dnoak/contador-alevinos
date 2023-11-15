from ultralytics import YOLO
from datetime import datetime

project_path = r'.'

def train(model):
    alevino_model = YOLO(model)
    train_data = r'F:\TCC\contagem-larvas\data\datasets\yolo\data.yaml'
    time = datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
    print(time)
    results = alevino_model.train(
        data=train_data, 
        epochs=400, 
        imgsz=640, 
        batch=-1, 
        workers=1,
        pretrained=True,
        project=f'/runs/detect',
        name=model.split('.')[0] + '_' + time,
    )

def train_all_models():
    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt']#, 'yolov8x.pt']
    for model in models:
        try: train(model)
        except: print(f'Error on model {model}')


if __name__ == '__main__':
    #train(f'yolov8n.pt')
    train_all_models()