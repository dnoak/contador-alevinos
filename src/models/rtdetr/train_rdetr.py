from ultralytics import RTDETR
from glob import glob
from datetime import datetime

def train(model):
    alevino_model = RTDETR(model)
    train_data = r'D:\Documentos\Projetos\tcc\contagem-ovos-larvas-peixe\data\alevinos 130.v1i.yolov8\data.yaml'
    time = datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
    print(time)
    results = alevino_model.train(
        data=train_data,
        epochs=400,
        imgsz=640,
        batch=10,
        workers=1,
        pretrained=True,
        project=f'runs/detect',
        name=model.split('.')[0] + '_' + time,
    )

def train_all_models():
    models = ['rtdetr-l.pt', 'rtdetr-x.pt']
    for model in models:
        try: train(model)
        except: print(f'Error on model {model}')

def mini_test():
    model = RTDETR(r"D:\Documentos\Projetos\tcc\contagem-ovos-larvas-peixe\runs\detect\rtdetr-l_2023-11-01_15-40-21\weights\best.pt")
    image_path = r"D:\Documentos\Projetos\tcc\contagem-ovos-larvas-peixe\data\YOLO_cropped_dataset\images+labels\N1_jpg.rf.a3959adf471b2630f8bda1a156b72654_1_2.jpg"
    model.predict(
        source=image_path,
        show=True
    )
    input()

if __name__ == '__main__':
    train_all_models() 
