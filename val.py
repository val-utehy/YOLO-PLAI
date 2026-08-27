from ultralytics import YOLO
model = YOLO('/home/share/YOLOv12-H/runs/detect/yolov12n_egsa_v3-9/weights/best.pt')
model.val(data='/home/share/yolov7/HAZE_dataset_3level25.8_real/3level_haze_gen.yaml', save_json=True, split='test')

#from ultralytics import YOLO
#from ultralytics.utils.torch_utils import get_flops
#model = YOLO('/home/share/YOLOv12-H/runs/detect/yolov12n_egsa_v3-10/weights/best.pt')
#print(get_flops(model.model, imgsz=640))