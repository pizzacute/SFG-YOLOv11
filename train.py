import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'C:\Users\cby\Desktop\小论文代码\ultralytics-main\ultralytics\cfg\models\11\SFG-YOLOv11.yaml')  # 修改为您的YAML绝对路径
    model.train(data=r'D:\soft\flower class.v1i.yolov8\data.yaml',  # 修改为您的数据集YAML路径
                imgsz=640,
                epochs=100,
                batch=16,
                workers=0,
                device='cuda',  # 如果用CPU，改成 'cpu'
                optimizer='SGD',
                close_mosaic=10,
                resume=True,
                project=r'path/to/runs',  # 修改为您的运行目录
                name='sfg-yolov11',
                single_cls=False,
                cache=False,
                )