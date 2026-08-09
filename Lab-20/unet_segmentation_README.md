# 🐾 U-Net for Image Segmentation — From Scratch (TensorFlow)

A from-scratch **TensorFlow/Keras** implementation of the **U-Net** architecture for semantic image segmentation, trained on the **Oxford-IIIT Pet Dataset** to segment pet images into background, pet, and outline classes.

## 📌 Overview

This notebook builds U-Net entirely from its core building blocks (no pre-built segmentation library) and walks through the full pipeline:

1. **Data loading & preprocessing** — reading paired images and segmentation masks, resizing/normalizing them into NumPy arrays.
2. **U-Net architecture** — custom encoder and decoder blocks assembled according to the original paper.
3. **Training** — compiling and fitting the model with sparse categorical cross-entropy loss.
4. **Evaluation** — bias/variance (over/underfitting) analysis via loss & accuracy curves.
5. **Visualization** — comparing predicted segmentation masks against ground truth.
6. **Fun Activity** — hyperparameter tuning, saving/loading model weights, and using the trained model to perform **background removal** on a pet image.

## 🧠 U-Net Architecture

Based on: Ronneberger et al., [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597)

U-Net follows a symmetric **encoder–decoder** ("contracting–expanding") structure with **skip connections** linking corresponding encoder and decoder levels — preserving spatial detail lost during downsampling.

**Encoder Block** (`EncoderMiniBlock`)
- 2× Conv2D (ReLU, He-normal init, same padding)
- Optional Dropout for regularization
- MaxPooling2D for downsampling
- Returns both the pooled output *and* a skip connection for the decoder

**Decoder Block** (`DecoderMiniBlock`)
- Conv2DTranspose to upsample
- Concatenation with the corresponding encoder skip connection
- 2× Conv2D to refine the merged features

**Full Model** (`UNetCompiled`)
```
Input (128, 128, 3)
→ Encoder Block 1 (32 filters)  ─┐
→ Encoder Block 2 (64 filters)  ─┤
→ Encoder Block 3 (128 filters) ─┤  skip connections
→ Encoder Block 4 (256 filters) ─┤
→ Bottleneck (512 filters)      │
→ Decoder Block 1 (256 filters) ←┘
→ Decoder Block 2 (128 filters) ←┘
→ Decoder Block 3 (64 filters)  ←┘
→ Decoder Block 4 (32 filters)  ←┘
→ Output Conv2D (n_classes, softmax logits)
```
- **Loss:** `SparseCategoricalCrossentropy(from_logits=True)`
- **Optimizer:** Adam
- **Classes:** 3 — background, pet body, pet outline

## 📂 Dataset

**[Oxford-IIIT Pet Dataset](https://www.kaggle.com/tanlikesmath/the-oxfordiiit-pet-dataset)** — downloaded directly from the source:
```bash
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf images.tar.gz && tar -xzf annotations.tar.gz
```
- **Images:** `.jpg` photos of cats and dogs
- **Masks (trimaps):** `.png` segmentation masks (`annotations/trimaps/`), same filenames as their corresponding images
- Images and masks are resized to `128×128` before training

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras — model architecture, training
- NumPy — array/data handling
- scikit-learn — train/validation split
- imageio, Pillow (PIL) — image I/O
- Matplotlib — visualization

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow numpy scikit-learn imageio pillow matplotlib
```

### Dataset Setup
```bash
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf images.tar.gz && tar -xzf annotations.tar.gz
```
This creates two folders in your working directory: `images/` and `annotations/trimaps/`.

### Project Structure
```
.
├── U_Net_for_Image_Segmentation_From_Scratch_Using_TensorFlow.ipynb   # Main notebook
├── images/                          # Pet photos (.jpg)
├── annotations/
│   └── trimaps/                     # Segmentation masks (.png)
└── unet_pet_segmentation_weights.weights.h5   # Saved model weights (after training)
```

### Running the Notebook
```bash
jupyter notebook
```
Open the notebook and run all cells in order. Training runs for 5 epochs by default (`batch_size=32`) — increase epochs/filters for better accuracy.

## 📊 Results

- The model is evaluated with training vs. validation **loss** and **accuracy** curves to check for underfitting (high bias) or overfitting (high variance).
- Predicted segmentation masks are visualized side-by-side with the original image and ground-truth mask.
- The **Fun Activity** section extends the project by:
  - Tuning hyperparameters (e.g. filter counts) for improved accuracy
  - Saving trained weights (`unet_pet_segmentation_weights.weights.h5`)
  - Reloading the model to perform **background removal** on new pet images using the predicted segmentation mask

## 🔑 Key Takeaways

- U-Net's skip connections are what allow it to produce **pixel-precise** segmentation masks despite the encoder's aggressive downsampling — a critical design choice for biomedical/fine-grained segmentation tasks.
- Building the architecture block-by-block (rather than using a prebuilt model) makes the encoder–decoder + skip-connection mechanics transparent and easy to modify.
- A trained segmentation model can be repurposed for practical tasks like automatic background removal, simply by using the predicted mask to isolate the foreground.

## 📖 References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Dataset: [Oxford-IIIT Pet Dataset](https://www.kaggle.com/tanlikesmath/the-oxfordiiit-pet-dataset) / [Original Source](http://www.robots.ox.ac.uk/~vgg/data/pets/)

## 📄 License

This project is for educational purposes.
