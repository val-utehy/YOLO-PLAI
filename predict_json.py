import os
import json
from ultralytics import YOLO
from glob import glob
from pathlib import Path
import torch

# ================= CONFIG =================
image_folder = "/home/share/YOLOV12/yolov12/CONF2_HAZEV2/dataset_combined/test/images"
model_path = "/home/share/YOLOv12-H/runs/detect/yolov12n_egsa_v3-9/weights/best.pt"

output_folder = "/home/share/YOLOV12-H/outputs"
output_json_path = "/home/share/yolov7/inference/yolov12_H/yolov12.json"

batch_size = 44
CONFIDENCE_THRESHOLD = 0.5

os.makedirs(output_folder, exist_ok=True)

# ================= LOAD MODEL =================
try:
    print(f"ℹ️ Loading model: {model_path}")
    model = YOLO(model_path)

    if torch.cuda.is_available():
        print("✅ GPU is available")
    else:
        print("⚠️ Using CPU")

except Exception as e:
    raise ValueError(f"❌ Model loading error: {e}")

# ================= LOAD IMAGES =================
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

image_files = []
for ext in image_extensions:
    image_files.extend(glob(os.path.join(image_folder, ext)))

if len(image_files) == 0:
    raise ValueError("❌ No images found!")

print(f"📸 Found {len(image_files)} images")

# ================= INFERENCE =================
all_detections = []

print(f"🚀 Running inference (batch={batch_size})")

for i in range(0, len(image_files), batch_size):

    batch_image_files = image_files[i:i + batch_size]
    if not batch_image_files:
        continue

    try:
        results_list = model(batch_image_files, stream=False, verbose=False)

    except torch.cuda.OutOfMemoryError as e:
        print("❌ CUDA OOM, skipping batch")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue

    except Exception as e:
        print(f"❌ Error: {e}")
        continue

    # ================= FIX IMPORTANT PART =================
    for img_path, result in zip(batch_image_files, results_list):

        image_id = Path(img_path).stem   # ✅ FIX: always correct filename

        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.data.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2, score, cls = box

            if score < CONFIDENCE_THRESHOLD:
                continue

            bbox = [
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2 - x1), 2),
                round(float(y2 - y1), 2)
            ]

            all_detections.append({
                "image_id": image_id,
                "category_id": int(cls),
                "bbox": bbox,
                "score": round(float(score), 5)
            })

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ================= SAVE JSON =================
try:
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=4)

    print(f"✅ Saved: {len(all_detections)} detections")
    print(f"📁 Output: {output_json_path}")

except Exception as e:
    print(f"❌ Save error: {e}")

print("🎉 DONE")