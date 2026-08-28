from ultralytics import YOLO
model = YOLO('best.pt')
model.val(data='3level_haze_gen.yaml', save_json=True, split='test')

#from ultralytics import YOLO
#from ultralytics.utils.torch_utils import get_flops
#model = YOLO('best.pt')
#print(get_flops(model.model, imgsz=640))
