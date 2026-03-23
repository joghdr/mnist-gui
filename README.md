# Multi-Threaded Deep Learning Diagnostic Workbench

A high-performance, from-scratch implementation of a Deep Neural Network (DNN) engine and diagnostic GUI. This project demonstrates the transition from theoretical mathematical modeling to functional software architecture, bypassing high-level ML libraries to expose the underlying mechanics of gradient descent and matrix calculus.

## Key Engineering Highlights
* Vectorized Engine: Full implementation of the Forward/Backward propagation cycle using optimized NumPy linear algebra (matrix-calculus) for O(n) batch processing.
* Numerical Stability: Integrated Log-Sum-Exp shift-invariance in the Softmax layer to prevent floating-point overflow/underflow during high-intensity training.
* Memory Optimization: Utilizes Fortran-contiguous memory layouts (np.asfortranarray) to maximize cache locality and computational throughput during matrix multiplications.
* Asynchronous Processing: Multi-threaded architecture built on PyQt5 QThreads, decoupling the heavy training worker from the UI thread to ensure a responsive, zero-latency user interface.

---

## Technical Stack
* Language: Python 3.x
* Mathematics: NumPy (Vectorized Matrix Operations)
* GUI Framework: PyQt5 (Signal-Slot Architecture, Custom Widgets)
* Visualization: Matplotlib (Real-time Dynamic Plotting)
* Data Engineering: Binary Byte-stream parsing for raw MNIST datasets

---

## Core Modules

### 1. The Mathematical Engine (/modules)
* backpropagation.py: Manual derivation of the Chain Rule across N-hidden layers, handling the Jacobian of the Softmax-Cross-Entropy loss.
* functions.py: A library of activation functions (ReLU, Sigmoid) and their derivatives, utilizing a dictionary-based mapping for architectural flexibility.
* Grids.py: A dedicated Tensor Manager that handles dynamic weight initialization and validates matrix dimensionality (W * A + b) to prevent runtime shape-mismatch errors.

### 2. The Diagnostic GUI (gui.py)
* Explainable AI (XAI) Visualizer: Real-time rendering of Weight and Gradient Heatmaps, allowing for visual inspection of feature extraction and vanishing gradient issues.
* Error Analysis Suite: Interactive "Next-Fail" navigation logic to isolate and analyze specific edge cases where the model's prediction diverges from ground truth.
* Dynamic Hyperparameter Control: On-the-fly adjustment of learning rates, batch sizes, and layer architectures via an event-driven interface.

### 3. Data Engineering (load.py)
* Raw Data Pipeline: Implements a custom loader to unpack high-endian binary idx data.
* Normalization & Encoding: Automated Min-Max scaling and One-Hot encoding of categorical labels.
* Stratification: Sophisticated data slicing for Training, Validation, and Testing sets to ensure statistical integrity.

---

## Use Cases
* Algorithm Validation: Testing non-standard activation functions and cost-function derivatives.
* Educational Debugging: Visualizing how weights evolve over time in a fully transparent environment without "black box" library abstractions.
* Performance Benchmarking: Evaluating the efficiency of manual vectorization versus compiled frameworks in a sandboxed environment.

---

## Installation & Usage
1. Clone the repository:
git clone https://github.com/yourusername/mnist-gui.git
cd mnist-gui

2. Install dependencies:
pip install numpy pandas pyqt5 matplotlib

3. Run the application:
python gui.py

---
Developed as a deep-dive into Neural Network architecture and Scientific Software Design.
