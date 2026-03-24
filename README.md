# Multilayer Perceptron (MLP) Diagnostic Workbench

Numpy-based implementation of a Multilayer Perceptron (MLP) engine and diagnostic GUI. This system provides a transparent environment for architectural experimentation with up to three hidden layers, bypassing high-level ML frameworks to expose the underlying mechanics of gradient descent and matrix calculus.

---

## Technical Architecture & Engineering Highlights
* **Vectorized Batch Processing**: NumPy-based linear algebra implementation utilizing column-major memory alignment to avoid redundant copies of large grids.
* **Augmented Feature Space**: Integration of bias-intercept terms directly into the input tensors to enable single-matrix weight transformations.
* **Numerical Stability**: Implementation of stable Softmax (max-subtraction) and [0, 1] input scaling to maintain signal integrity.
* **Asynchronous Task Threading**: QThread-based offloading of training cycles to maintain a non-blocking GUI.

---

## Technical Stack
* **Language:** Python 3.10+
* **Mathematics:** NumPy (Vectorized Matrix Operations)
* **GUI Framework:** PyQt5 (Signal-Slot Architecture, Custom Widgets)
* **Visualization:** Matplotlib (Dynamic Real-time Plotting)
* **Data Handling:** Pandas (CSV Ingestion and Matrix Transformation)

---

## System Modules

### 1. Mathematical Engine (/modules)
* **backpropagation.py**: Implements the backpropagation algorithm for multi-layer architectures, handling weight and bias updates for the Softmax-Cross-Entropy loss without external autograd libraries.
* **functions.py**: A modular library of activation functions (ReLU, Sigmoid) and their derivatives, utilizing dictionary-based mapping for runtime architectural flexibility.
* **Grids.py (Model State & Buffer Manager)**: A centralized class for initializing and storing the model state (weights) and computational buffers (activations, gradients) across all network layers.

### 2. Experimental Sandbox (gui.py)
* **Feature & Gradient Monitor**: Provides real-time heatmap rendering of the input-to-hidden weights and their corresponding gradients to visually inspect the network state and weight evolution.

* **Model Audit: Provides interface to browse training and validation results.

* **Hyperparameter Configuration**: A dedicated interface for defining learning rates, batch sizes, and layer architectures, enabling rapid iteration without manual script modification.

### 3. Data Pipeline (load.py)
* **CSV Processing**: An automated data-loading pipeline for MNIST datasets using Pandas and NumPy.
* **Feature Scaling**: Performs manual Min-Max scaling and vectorized "One-Hot" encoding for label transformation.
* **Matrix Augmentation**: Integrates bias terms directly into the input feature space.

---

## Use Cases
* **Algorithm Validation**: Testing non-standard activation functions and custom cost-function derivatives.
* **Diagnostic Debugging**: Visualizing weight evolution in a fully transparent environment without "black box" abstractions.
* **Performance Benchmarking**: Evaluating the efficiency of manual vectorization against compiled frameworks in a sandboxed environment.

---

## Installation & Usage
1. Clone the repository:
  ```
  git clone https://github.com/yourusername/mnist-gui.git
  cd mnist-gui
  ```


2. Install dependencies:
  ```pip install numpy pandas PyQt5 matplotlib```

3. Run the application:
  ```python main.py```

---
*A NumPy-based MLP implementation focused on Network Architecture and State Monitoring.*
