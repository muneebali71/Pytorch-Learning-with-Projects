# 🔥 PyTorch Learning Journey

A collection of hands-on notebooks documenting my journey learning **PyTorch** from scratch — covering fundamentals, model building, training workflows, neural network classification, and computer vision with CNNs.

---

## 📂 Notebooks Overview

| Notebook | Topic | Key Concepts |
|---|---|---|
| `pytorch_fundamentals.ipynb` | PyTorch Basics | Tensors, dtypes, reshaping, NumPy interop |
| `pytorch_workflow.ipynb` | End-to-End Workflow | Linear regression, training loop, model saving |
| `Neural_Network_classification.ipynb` | Binary Classification | Neural nets, BCELoss, sigmoid, accuracy |
| `pytorch_computer_vision.ipynb` | Computer Vision | FashionMNIST, CNNs, DataLoaders, model comparison |

---

## 📓 Notebook Details

### 1. `pytorch_fundamentals.ipynb` — PyTorch Basics

The starting point. This notebook covers the building blocks of PyTorch before diving into any models.

**What's covered:**
- Scalars, vectors, matrices, and tensors — creating and understanding dimensions/shapes
- Random tensors, zeros, ones, and `torch.arange()`
- Tensor datatypes (`float32`, `float16`) and how they interact
- Tensor operations: **reshape**, **view**, **stack** (`dim=0` and `dim=1`), **squeeze** and **unsqueeze**
- `torch.permute()` to rearrange dimensions
- Converting between **NumPy arrays and PyTorch tensors** (`torch.from_numpy`)
- **Reproducibility** using `torch.manual_seed()`

---

### 2. `pytorch_workflow.ipynb` — End-to-End PyTorch Workflow

Covers the full lifecycle of building, training, evaluating, saving, and loading a model.

**What's covered:**
- Creating synthetic linear data with known `weight=0.7` and `bias=0.3`
- Splitting data into **train/test sets** (80/20 split)
- Visualizing training vs test data with `matplotlib`
- Building a **Linear Regression model** using `nn.Module` with `nn.Parameter`
- Making predictions with `torch.inference_mode()`
- Setting up **L1Loss** and **SGD optimizer**
- Writing a complete **training loop** (200 epochs):
  - Forward pass → Loss → Zero grad → Backprop → Optimizer step
  - Evaluating on test data each epoch with `model.eval()`
- Plotting **training and test loss curves**
- **Saving and loading models** using:
  - `torch.save()` — saves state dict as `.pth` file
  - `torch.load()` — loads back from disk
  - `model.load_state_dict()` — restores model weights

---

### 3. `Neural_Network_classification.ipynb` — Binary Classification

Applies neural networks to a non-linear classification problem.

**What's covered:**
- Generating a **circles dataset** using `sklearn.datasets.make_circles` (1000 samples, noise=0.03)
- Visualizing the dataset with `matplotlib` scatter plot
- Converting NumPy data to PyTorch tensors and splitting into train/test sets
- Building a `CircleModel` with two `nn.Linear` layers:
  - Input: 2 features → Hidden: 5 → Output: 1
- Understanding **logits vs probabilities** — raw model output → sigmoid → rounded label
- Loss function: `nn.BCEWithLogitsLoss` (includes built-in sigmoid)
- Optimizer: `SGD` with `lr=0.1`
- Custom `accuracy_fn` to track prediction accuracy
- Full **training loop** with 5-step process:
  1. Forward pass
  2. Calculate loss
  3. Zero gradients
  4. Backpropagation
  5. Optimizer step
- Understanding the **logit → probability → label** pipeline

> 💡 **Key Note:** `BCEWithLogitsLoss` is preferred over `BCELoss` because it combines the sigmoid and loss in one numerically stable operation.

---

### 4. `pytorch_computer_vision.ipynb` — Computer Vision with CNNs

The most advanced notebook — builds and compares three models on the **FashionMNIST** dataset.

**What's covered:**

**Data & Setup**
- Loading **FashionMNIST** (60,000 train / 10,000 test, 28×28 grayscale images, 10 classes) via `torchvision.datasets`
- Visualizing sample images with class labels
- Creating **DataLoaders** with `batch_size=32`, understanding why batching matters

**Models Built (3 models compared)**

| Model | Architecture | Notes |
|---|---|---|
| `FashionMNISTModelV0` | Flatten → Linear → Linear | Baseline linear model |
| `FashionMNISTModelV1` | Flatten → Linear → ReLU → Linear → ReLU | Added non-linearity |
| `FashionMNISTModelV2` | Conv Block 1 → Conv Block 2 → Classifier | Full CNN |

**CNN Architecture (Model V2):**
- **Conv Block 1:** `Conv2d → ReLU → Conv2d → ReLU → MaxPool2d(2)`
- **Conv Block 2:** `Conv2d → ReLU → Conv2d → ReLU → MaxPool2d(2)`
- **Classifier:** `Flatten → Linear(hidden*7*7 → num_classes)`

**Training & Evaluation**
- `CrossEntropyLoss` + `SGD` optimizer
- Reusable `train_step()` and `test_step()` functions
- Timing experiments with `timeit.default_timer`
- `eval_model()` function returning model name, loss, and accuracy
- **Comparing all three models** side-by-side in a `pandas` DataFrame
- Accuracy bar chart visualization

---

## 🛠️ Tech Stack

- **Python 3.x**
- **PyTorch** — core deep learning framework
- **torchvision** — datasets, transforms, and vision utilities
- **scikit-learn** — dataset generation and train/test split
- **NumPy** — array operations and interop with PyTorch
- **pandas** — results comparison
- **matplotlib** — data and loss curve visualization
- **tqdm** — progress bars during training

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install scikit-learn numpy pandas matplotlib tqdm
```

### Running the Notebooks

Clone the repo and launch Jupyter:

```bash
git clone <your-repo-url>
cd <repo-folder>
jupyter notebook
```

Open the notebooks in this recommended order:

```
1. pytorch_fundamentals.ipynb
2. pytorch_workflow.ipynb
3. Neural_Network_classification.ipynb
4. pytorch_computer_vision.ipynb
```

---

## 📈 Learning Path

```
Tensor Basics
    ↓
Linear Regression (Full Workflow)
    ↓
Binary Classification (Neural Network)
    ↓
Multi-class Classification (CNN + FashionMNIST)
```

---

## 📝 Notes

- These are personal learning notebooks — code focuses on clarity over optimization
- GPU support is handled automatically: `device = "cuda" if torch.cuda.is_available() else "cpu"`
- Models are saved to a local `models/` directory in the workflow notebook
- The CNN in the computer vision notebook significantly outperforms the linear baselines

---

## 🙌 Acknowledgements

Learning resources and inspiration from the [Learn PyTorch for Deep Learning](https://www.learnpytorch.io/) course by Daniel Bourke.
