# 🔢 CNN for MNIST Digit Classification

A Convolutional Neural Network built with **Keras / TensorFlow** to classify handwritten digits (0–9) from the classic **MNIST** dataset, with full data exploration, model training, evaluation, and a bonus simplified-model comparison exercise.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" width="400">
</p>

## 📌 Overview

This project walks through a complete deep learning pipeline for image classification:

1. **Data Exploration** — inspecting class distribution and sample images
2. **Preprocessing** — reshaping, normalizing pixel values, train/validation split
3. **Model Building** — a deep CNN with Conv2D, BatchNormalization, MaxPooling & Dropout layers
4. **Training** — with early stopping and best-model checkpointing
5. **Evaluation** — accuracy/loss curves, per-class precision/recall/F1, correct vs. incorrect predictions
6. **Bonus Exercise** — a simpler CNN built for comparison against the deeper model

## 🧠 Model Architecture

**Main CNN Model**

| Layer Type | Details |
|---|---|
| Conv2D | 32 filters, 3×3, ReLU, same padding, He-normal init |
| BatchNormalization | |
| Conv2D | 32 filters, 3×3, ReLU |
| BatchNormalization | |
| Conv2D | 32 filters, 5×5, stride 2, same padding, ReLU |
| MaxPooling2D | 2×2 |
| BatchNormalization | |
| Dropout | 0.4 |
| Conv2D | 64 filters, 3×3, stride 2, same padding, ReLU |
| MaxPooling2D | 2×2 |
| BatchNormalization | |
| Conv2D | 64 filters, 3×3, stride 2, same padding, ReLU |
| Dropout | 0.4 |
| Flatten | |
| Dense | 128 units, ReLU |
| Dropout | 0.4 |
| Dense (Output) | 10 units, Softmax |

**Compilation:** `loss='categorical_crossentropy'`, `optimizer='adam'`, `metrics=['accuracy']`
**Callbacks:** `EarlyStopping` (monitors loss) + `ModelCheckpoint` (saves best weights by `val_accuracy`)

**Bonus — Simpler CNN Model** (for the reduced-layers exercise)
```
Conv2D(32, 3x3, relu) → MaxPooling2D(2x2) → Dropout(0.25) →
Flatten → Dense(64, relu) → Dense(10, softmax)
```
Used to compare how reducing depth (1 conv layer vs. 5, 64 dense units vs. 128) affects accuracy on the same dataset.

## 📂 Dataset

The classic **MNIST** handwritten digit dataset:
- 28×28 grayscale images (784 pixels), 10 classes (digits 0–9)
- Provided as `train.csv` and `test.csv` (Kaggle "Digit Recognizer" format — pixel values as flattened columns + a `label` column for train)
- Classes are roughly balanced (~9–11% per class)

Preprocessing steps:
- Reshape flattened pixels → `(28, 28, 1)` image tensors
- Normalize pixel values to `[0, 1]`
- One-hot encode labels
- Split: 90% train / 10% validation

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras — model building & training
- NumPy, Pandas — data handling
- scikit-learn — train/validation split, classification report
- Matplotlib, Seaborn — visualization

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

### Project Structure
```
.
├── cnn_for_mnist.ipynb     # Main notebook
├── input/
│   ├── train.csv           # Training data (Kaggle Digit Recognizer format)
│   └── test.csv             # Test data
└── model.weights.h5        # Saved best model weights (generated after training)
```

### Dataset Setup
Download the MNIST digit data in Kaggle "Digit Recognizer" CSV format and place `train.csv` and `test.csv` inside an `input/` folder in the project root (or update the `PATH` variable in the notebook).

### Running the Notebook
1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Place the dataset files as described above.
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `cnn_for_mnist.ipynb` and run all cells.

## 📊 Results

- **Validation accuracy** reaches **~99%** for most classes, with digit **4** slightly lower (~98%).
- The model shows no signs of overfitting — validation accuracy/loss stay in step with training accuracy/loss, thanks to Dropout regularization.
- Misclassified validation images are visualized and are largely digits that are ambiguous even to a human eye.
- Final predictions are generated for the test set using `argmax` over the softmax output.

## 🔍 Key Takeaways

- Stacking Conv2D + BatchNormalization + Dropout layers helps build a robust, non-overfitting CNN.
- `EarlyStopping` + `ModelCheckpoint` callbacks let you train safely without manually watching for overfitting.
- The bonus exercise (simpler 1-conv-layer model) demonstrates the trade-off between model depth and accuracy on MNIST.

## 📖 References

1. Yann LeCun, [MNIST Database](http://yann.lecun.com/exdb/mnist/)
2. DanB, CollinMoris — [Deep Learning From Scratch](https://www.kaggle.com/dansbecker/deep-learning-from-scratch)
3. DanB — [Dropout and Strides for Larger Models](https://www.kaggle.com/dansbecker/dropout-and-strides-for-larger-models)
4. BGO — [CNN with Keras](https://www.kaggle.com/bugraokcu/cnn-with-keras)

## 📄 License

This project is for educational purposes.
