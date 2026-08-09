# 🚗 Object Detection using YOLOv8

An object detection project using **YOLOv8** (via the `ultralytics` library) applied to a **self-driving car dataset**, detecting objects like cars, trucks, pedestrians, bicyclists, and traffic lights in road scene images.

## 📌 Overview

This notebook walks through:

1. **Understanding YOLO** — what "You Only Look Once" object detection is and how it differs from older, slower detection frameworks (e.g. Faster R-CNN, sliding-window classifiers).
2. **Loading & exploring the dataset** — reading bounding-box annotations from a CSV and visualizing labeled sample images.
3. **Running inference with a pretrained YOLOv8 model** (`yolov8m.pt`) — detecting and localizing objects in road images.
4. **Extracting detection results** — bounding box coordinates, class labels, and confidence scores.
5. **Visualizing predictions** — plotting YOLO's detected bounding boxes directly on images.
6. **Exercise** — a custom script to randomly select an image, pull its ground-truth annotations from the CSV, and draw bounding boxes + class labels using `PIL.ImageDraw`.

## 🧠 What is YOLO?

**YOLO (You Only Look Once)** is a real-time object detection algorithm. Unlike earlier approaches that repurposed image classifiers and scanned images at multiple scales/regions (slow and inefficient), YOLO looks at the entire image in a single pass, directly predicting bounding boxes and class probabilities simultaneously — making it significantly faster while remaining accurate.

This project uses **YOLOv8**, the latest (at time of writing) iteration from Ultralytics, loaded with pretrained weights (`yolov8m.pt` — the medium-sized model variant).

## 📂 Dataset

**Self-Driving Cars** dataset (via Kaggle: `alincijov/self-driving-cars`) — road scene images with bounding-box annotations for 5 object classes:

| Class ID | Label |
|---|---|
| 1 | Car |
| 2 | Truck |
| 3 | Pedestrian / Person |
| 4 | Bicyclist / Bicycle |
| 5 | Traffic Light |

Annotations are provided in `labels_train.csv` with columns for the image filename (`frame`), class ID, and bounding box coordinates (`xmin`, `xmax`, `ymin`, `ymax`).

## 🛠️ Tech Stack

- Python
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection model & inference
- OpenCV (`cv2`) — image reading/processing
- Pillow (PIL) — image drawing & annotation
- NumPy, Pandas — data handling
- Matplotlib — visualization
- Kaggle API — dataset download
- Google Colab — original runtime environment (uses `google.colab.files` for uploads)

## 🚀 Getting Started

### Prerequisites
```bash
pip install ultralytics opencv-python pandas numpy matplotlib pillow scikit-learn kaggle
```

### Dataset Setup (Kaggle API)
1. Generate a Kaggle API token from your [Kaggle account settings](https://www.kaggle.com/settings) → downloads a `kaggle.json` file.
2. Place it in the right location:
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Download and unzip the dataset:
   ```bash
   kaggle datasets download -d alincijov/self-driving-cars
   unzip self-driving-cars.zip -d self-driving-cars/
   ```

> **Note:** This notebook was originally written for **Google Colab** (it uses `google.colab.files.upload()` to upload `kaggle.json`). If running locally or in Jupyter, replace that cell with the manual `kaggle.json` setup shown above.

### Project Structure
```
.
├── Object_detection_by_using_YOLOv8.ipynb   # Main notebook
└── self-driving-cars/
    ├── labels_train.csv                      # Bounding box annotations
    └── images/                                # Road scene images
```

### Running the Notebook
```bash
jupyter notebook
```
Open `Object_detection_by_using_YOLOv8.ipynb` and run all cells in order. The pretrained `yolov8m.pt` weights download automatically on first use (requires internet access).

## 🔍 How Detection Works Here

```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
results = model.predict(source="path/to/image.jpg", save=True, conf=0.2, iou=0.5)

for box in results[0].boxes:
    class_id = results[0].names[box.cls[0].item()]
    cords = [round(x) for x in box.xyxy[0].tolist()]
    conf = round(box.conf[0].item(), 2)
    print(class_id, cords, conf)
```
- `conf=0.2` — minimum confidence threshold for a detection to be kept
- `iou=0.5` — IoU threshold used for Non-Max Suppression (removing duplicate overlapping boxes)

## 📊 What You'll See

- Ground-truth labeled images with bounding boxes drawn from the CSV annotations
- YOLOv8 predictions overlaid on road images, showing detected class, bounding box, and confidence score
- A hands-on exercise reproducing bounding-box visualization manually with `PIL.ImageDraw`, reinforcing how annotation data maps to image coordinates

## 🔑 Key Takeaways

- YOLOv8 provides fast, accurate, out-of-the-box object detection using pretrained weights — no training required to get started.
- Detection confidence and IoU thresholds directly control precision/recall trade-offs and duplicate box suppression.
- Understanding how to parse and visualize ground-truth annotations (CSV → bounding boxes) is foundational before evaluating or fine-tuning any detection model.

## 📖 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- Dataset: [Self-Driving Cars (Kaggle)](https://www.kaggle.com/datasets/alincijov/self-driving-cars)

## 📄 License

This project is for educational purposes.
