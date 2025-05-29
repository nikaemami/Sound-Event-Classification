# 🔊 Sound Event Classification

This project implements a neural network pipeline to classify environmental sounds using the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50). The pipeline includes audio preprocessing, Mel-spectrogram feature extraction, and training PyTorch-based models for classification.

---

## 📁 Dataset: ESC-50

The ESC-50 dataset consists of:

- 2,000 environmental audio clips (5 seconds each)
- 50 human-labeled sound categories (e.g., dog bark, thunder, crying baby)
- Metadata CSV file specifying labels and folds

As part of preprocessing, we visualize:
<h4>Waveform and Log-Mel Spectrogram</h4>
<p float="left">
  <img src="img/1.png" width="400" style="margin-right:10px;"/>
  <img src="img/2.png" width="400"/>
</p>

---

## Models

We train and evaluate two different classifiers. For both:

- **Input:** Log-Mel spectrogram, shape: `(batch, 128, 431)`  
- **Target:** Class label (as index), shape: `(batch, 1)`  
- **Output:** Probability vector over 50 classes, shape: `(batch, 50)`

### 1. Multi-Layer Perceptron (MLP)

- A simple 2-layer fully connected network  
- The spectrogram is flattened before input (MLPs do not preserve spatial structure)  
- Uses ReLU activation between layers

### 2. 1D Convolutional Neural Network (Conv1D)

- Two stacked `Conv1D` layers with ReLU activations  
- Followed by pooling and a final linear layer  
- Preserves temporal structure in the spectrogram, enabling learning of local sequential patterns

---

## 📊 Performance

### MLP Results

| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.29  |
| Precision | 0.28  |
| Recall    | 0.29  |
| F1 Score  | 0.27  |


<h4>MLP Training Loss and Accuracy Curves:</h4>
<img src="img/3.png" width="500"/>

---

### Conv1D Results

| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.35  |
| Precision | 0.40  |
| Recall    | 0.35  |
| F1 Score  | 0.34  |

<h4>Conv1D Loss and Accuracy Curves:</h4>
<img src="img/4.png" width="500"/>

---

## Observations

Conv1D consistently outperforms the MLP across all evaluation metrics. This suggests that Conv1D models are better suited for this task, as they can exploit the temporal and local patterns present in spectrogram data. In contrast, MLPs treat each input feature independently, which limits their ability to model sequential dependencies.

---

