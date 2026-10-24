import json
from pathlib import Path
import cv2, yaml
from ultralytics import YOLO
root=Path(__file__).resolve().parents[1]
out=root/'build-windows-2.9.1/installer-inputs'; out.mkdir(exist_ok=True)
dataset=out/'dataset'; dataset.mkdir(exist_ok=True)
names=YOLO(root/'models/yolov8n.pt').names
config={'path':(root/'build-windows-2.9.1/fixtures/coco8').as_posix(),'train':'images/train','val':'images/val','names':names}
source=dataset/'data.yaml'; source.write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8')
image=cv2.imread(str(root/'assets/bus.jpg')); image=cv2.resize(image,(640,480))
images=[]
for i in range(2):
    path=out/f'temporary-{i}.png'; cv2.imwrite(str(path),image); images.append(path.as_posix())
video=out/'acceptance.avi'; writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(640,480))
for i in range(30): writer.write(image)
writer.release()
evaluation={'model_path':(root/'models/yolov8n.pt').as_posix(),'dataset_yaml':source.as_posix(),'task':'yolov8','input_size':640,'confidence':.25,'nms_iou':.45,'model_class_id':0,'dataset_class_id':0,'split':'val','device':'cpu','scan_start':.05,'scan_end':.95,'scan_step':.01,'match_iou':.5,'batch_size':3,'cpu_threads':1,'figure_formats':['svg'],'figure_dpi':72,'verify_decision_layout':True}
preview={'dataset_directory':dataset.as_posix(),'dataset_yaml':source.as_posix(),'video':video.as_posix(),'model':(root/'models/yolov5n.onnx').as_posix(),'images':images}
for name,value in [('evaluation-request',evaluation),('preview-request',preview)]:
    (out/(name+'.json')).write_text(json.dumps(value,indent=2),encoding='utf-8')
print(out)
