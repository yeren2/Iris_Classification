# 🌸 Iris Species Classification with PyTorch

This repository contains a Deep Learning project built with **PyTorch** to classify Iris flower species into three categories (*Setosa, Versicolor, Virginica*) using the classic Iris dataset.

---

## 📌 Features & Highlights

* **Data Preprocessing:** Standardized data split with `train_test_split` (stratified sampling) and label encoding with `scikit-learn`.
* **Neural Network Architecture:** Multi-Layer Perceptron (MLP) built with `torch.nn.Sequential` using `ReLU` activations.
* **Loss & Optimization:** Trained using `nn.CrossEntropyLoss` and the `Adam` optimizer ($lr = 0.01$).
* **Metrics & Evaluation:** Performance tracked using **TorchMetrics** (`MulticlassAccuracy` and `MulticlassConfusionMatrix`).

---

## 🏗️ Model Architecture

The network consists of fully connected (Linear) layers:

$$\text{Input (4 Features)} \longrightarrow \text{Linear(4, 4)} \longrightarrow \text{ReLU} \longrightarrow \text{Linear(4, 4)} \longrightarrow \text{ReLU} \longrightarrow \text{Linear(4, 3)} \longrightarrow \text{Output (3 Logits)}$$

---

## 📊 Performance & Results

The model was trained for **200 epochs** and achieved optimal convergence:

* **Final Training Accuracy:** ~98.3%
* **Final Test Accuracy:** **100.0%**
* **Final Test Loss:** ~0.064

### Confusion Matrix
The model correctly predicted all 30 test samples without any misclassifications:

```text
tensor([[10,  0,  0],
        [ 0, 10,  0],
        [ 0,  0, 10]])
