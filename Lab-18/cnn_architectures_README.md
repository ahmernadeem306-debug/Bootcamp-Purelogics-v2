# 🏗️ CNN Architectures: VGG, ResNet & InceptionNet — Transfer Learning

An exploration of the landmark **ImageNet-winning CNN architectures** — VGG16, VGG19, ResNet50, and InceptionV3 — using their **pretrained ImageNet weights** for image feature extraction and prediction, plus a hands-on exercise building the VGG19 architecture from scratch.

## 📌 Overview

This notebook covers:

1. **Custom CNN block** — a small VGG-style convolutional block built manually to illustrate the fundamentals (stacked Conv2D + MaxPooling layers).
2. **Pretrained model inference** — loading **VGG16**, **VGG19**, **ResNet50**, and **InceptionV3** with ImageNet weights via `keras.applications`, and running predictions on real flower images to compare how each architecture "sees" the same input.
3. **Architecture deep-dive** — explanations of what makes each network distinct:
   - **VGG16 / VGG19** — deep stacks of small 3×3 convolutions
   - **ResNet50** — skip/residual connections to solve vanishing gradients and the degradation problem in very deep networks
   - **InceptionV3 (GoogLeNet family)** — inception modules for multi-scale feature extraction with dimensionality-reducing 1×1 convolutions
4. **Hands-on activity** — manually defining the full **VGG19** architecture (5 convolutional blocks + 3 dense layers) using the Keras Sequential API.

## 🧠 Architectures Covered

| Model | Key Idea | Layers |
|---|---|---|
| VGG16 | Deep stack of 3×3 convolutions | 13 conv + 3 dense |
| VGG19 | Deeper version of VGG16 | 16 conv + 3 dense |
| ResNet50 | Residual / skip connections | 50 layers |
| InceptionV3 | Multi-scale inception modules | 22+ layers |

Each pretrained model is loaded with `weights='imagenet'` and used to classify sample images, decoding the top-3 predicted ImageNet classes per image.

### VGG19 — Built From Scratch (Activity)
```
Input (224x224x3)
→ Block 1: Conv2D(64) x2 → MaxPool
→ Block 2: Conv2D(128) x2 → MaxPool
→ Block 3: Conv2D(256) x4 → MaxPool
→ Block 4: Conv2D(512) x4 → MaxPool
→ Block 5: Conv2D(512) x4 → MaxPool
→ Flatten → Dense → Dense → Dense(num_classes, softmax)
```

## 📂 Dataset

**Flowers Recognition** dataset (via Kaggle: `alxmamaev/flowers-recognition`) — used purely for sample images to run predictions through the pretrained models. Categories include daisy, dandelion, sunflower, and tulip.

## 🛠️ Tech Stack

- Python
- Keras / TensorFlow — `keras.applications` (VGG16, VGG19, ResNet50, InceptionV3)
- NumPy, Pandas — data handling
- Matplotlib, Seaborn — visualization of predictions
- Kaggle API — dataset download
- Google Colab — original runtime environment (uses `google.colab.files` for uploads)

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn pillow kaggle
```

### Dataset Setup (Kaggle API)
1. Generate a Kaggle API token from your [Kaggle account settings](https://www.kaggle.com/settings) → this downloads a `kaggle.json` file.
2. Place it in the right location:
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Download and unzip the dataset:
   ```bash
   kaggle datasets download -d alxmamaev/flowers-recognition
   unzip flowers-recognition.zip -d flowers/
   ```

> **Note:** This notebook was originally written for **Google Colab** (it uses `google.colab.files.upload()` to upload `kaggle.json`). If running locally or in Jupyter, replace that cell with the manual `kaggle.json` setup shown above.

### Project Structure
```
.
├── cnn_architectures_vgg_resnet_inception_tl.ipynb   # Main notebook
└── flowers/
    └── flowers/
        ├── daisy/
        ├── dandelion/
        ├── sunflower/
        └── tulip/
```

### Running the Notebook
```bash
jupyter notebook
```
Open `cnn_architectures_vgg_resnet_inception_tl.ipynb` and run all cells in order. Each pretrained model will download its ImageNet weights automatically on first use (requires internet access).

## 📊 What You'll See

- Side-by-side sample flower images fed into each architecture
- Top-3 predicted ImageNet class probabilities per image, per model (bar plots)
- A visual/comparative sense of how VGG, ResNet, and Inception differ in what features they pick up on
- A fully defined, from-scratch VGG19 `Sequential` model as a practice exercise

## 🔍 Key Takeaways

- **Transfer learning** lets you reuse powerful, pretrained feature extractors instead of training deep CNNs from scratch.
- **VGG** favors simplicity and depth with small filters; **ResNet** solves the vanishing gradient problem via skip connections, enabling much deeper networks; **Inception** captures multi-scale features efficiently within a single layer.
- Since these models were trained on ImageNet's 1000 classes, predictions on flower images reflect the closest ImageNet categories rather than the flower's specific species — a great illustration of why fine-tuning matters for domain-specific transfer learning.

## 📖 References

- Original ResNet Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
- [Keras Applications Documentation](https://keras.io/api/applications/)
- Dataset: [Flowers Recognition (Kaggle)](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

## 📄 License

This project is for educational purposes.
