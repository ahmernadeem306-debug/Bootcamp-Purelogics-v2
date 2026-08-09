# 🐱 Deep Neural Network for Image Classification — Cat vs Non-Cat

A from-scratch implementation of a Deep Neural Network (built purely with **NumPy**, no deep learning frameworks) that classifies images as **cat** or **non-cat**. This project builds both a 2-layer neural network and a fully generalized L-layer deep neural network, then compares their performance.

## 📌 Overview

This notebook demonstrates the core building blocks of deep learning by implementing a neural network **from the ground up**:

- Forward propagation (Linear → ReLU → ... → Linear → Sigmoid)
- Cost computation (cross-entropy loss)
- Backward propagation (manual gradient computation)
- Parameter updates via gradient descent
- Model evaluation and prediction

Two architectures are trained and compared:

| Model | Architecture | Test Accuracy |
|---|---|---|
| 2-Layer Neural Network | LINEAR → RELU → LINEAR → SIGMOID | ~72% |
| 4-Layer Deep Neural Network | [LINEAR → RELU] × 3 → LINEAR → SIGMOID | ~80% |

## 🧠 Model Architectures

**2-Layer Neural Network**
```
INPUT (12288) → LINEAR → RELU → LINEAR → SIGMOID → OUTPUT (0/1)
```

**L-Layer Deep Neural Network**
```
layers_dims = [12288, 20, 7, 5, 1]
INPUT → [LINEAR → RELU] × (L-1) → LINEAR → SIGMOID → OUTPUT (0/1)
```

## 📂 Dataset

The project uses the **"Cat vs Non-Cat"** dataset (`data.h5`), containing:
- Training set of labeled cat (1) and non-cat (0) images
- Test set of labeled cat and non-cat images
- Each image has shape `(64, 64, 3)`

Images are flattened and standardized (pixel values scaled to `[0, 1]`) before being fed into the network:
```python
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
train_x = train_x_flatten / 255
```

## 🛠️ Tech Stack

- Python
- NumPy — core computation
- h5py — reading the `.h5` dataset
- Matplotlib — visualization
- PIL / SciPy — image loading & preprocessing

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy h5py matplotlib pillow scipy
```

### Project Structure
```
.
├── Deep_Neural_Network_Image_Classification.ipynb   # Main notebook
├── dnn_app_utils_v3.py                               # Helper functions (forward/backward prop, etc.)
├── datasets/                                         # train/test .h5 dataset files
└── images/                                            # Sample & custom test images
```

### Running the Notebook
1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Ensure `dnn_app_utils_v3.py` and the `datasets/` folder are in the working directory.
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `Deep_Neural_Network_Image_Classification.ipynb` and run all cells.

### Testing with Your Own Image
You can test the trained model on your own image:
```python
fileImage = Image.open("your_image.jpg").convert("RGB").resize([num_px, num_px])
my_image = np.array(fileImage).reshape(num_px * num_px * 3, 1) / 255.
my_predicted_image = predict(my_image, [1], parameters)
```

## 📊 Results

- **Train Accuracy (4-layer model):** ~98.5%
- **Test Accuracy (4-layer model):** ~80%
- The deeper L-layer network **outperforms** the simpler 2-layer network, showing the benefit of increased model depth.

### Common Misclassifications
The model tends to underperform on images with:
- Unusual cat body position
- Background color similar to the cat
- Uncommon cat color/species
- Odd camera angle or brightness
- Extreme scale (cat too small or too large in frame)

## 📖 Acknowledgements

This project is based on the **Deep Learning Specialization** by Andrew Ng (deeplearning.ai / Coursera) — "Neural Networks and Deep Learning" course, Week 4 programming assignment.

## 📄 License

This project is for educational purposes. Dataset and starter code credit: deeplearning.ai.
