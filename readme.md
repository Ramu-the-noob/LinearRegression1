# Linear Regression from Scratch using Gradient Descent

A lightweight Python project demonstrating how **Linear Regression** and **Gradient Descent** work under the hood without relying on high-level machine learning frameworks like `scikit-learn` or `PyTorch`. 

This repository contains pure NumPy code to train a model iteratively and Plotly code to visualize the loss reduction over time.

---

## 📌 Project Overview

Linear Regression models a linear relationship between input variables ($X$) and continuous output targets ($Y$) using the equation:

$$\hat{y} = w \cdot X + b$$

Where:
- **$w$ (Weight/Slope):** Controls the steepness of the line.
- **$b$ (Bias/Intercept):** Controls the vertical shift of the line.
- **$\hat{y}$ (Prediction):** The output predicted by the model.

Instead of solving $w$ and $b$ analytically, this project uses **Gradient Descent** to iteratively optimize the parameters by minimizing the **Mean Squared Error (MSE)**.

---

## 📊 Training Results & Visualization

By setting a small learning rate ($\alpha = 0.000001$), we can track the exact trajectory of gradient descent reducing model error step-by-step over 1,000 steps:

- **Initial State:** $w = 0.0, b = 0.0$, initial $MSE \approx 14,600$.
- **Convergence:** The loss smoothly decreases following a standard exponential decay curve, plateauing close to zero as $w$ and $b$ stabilize near optimal values.

![Training Loss Curve](https://raw.githubusercontent.com/placeholder/loss_curve.png) <!-- Replace link with screenshot after uploading -->

---

## 🧮 How It Works (Step-by-Step)

For each step in the training loop:

1. **Prediction:**
   $$\hat{y} = w \cdot X + b$$

2. **Error & Loss Calculation (Mean Squared Error):**
   $$E = \hat{y} - Y$$
   $$MSE = \frac{1}{N} \sum (\hat{y} - Y)^2$$

3. **Gradient Calculation (Partial Derivatives):**
   $$\frac{\partial MSE}{\partial w} = \frac{2}{N} \sum (\hat{y} - Y) \cdot X$$
   $$\frac{\partial MSE}{\partial b} = \frac{2}{N} \sum (\hat{y} - Y)$$

4. **Parameter Updates (Gradient Descent):**
   $$w \leftarrow w - \alpha \cdot \frac{\partial MSE}{\partial w}$$
   $$b \leftarrow b - \alpha \cdot \frac{\partial MSE}{\partial b}$$

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed along with the required dependencies:

```bash
pip install numpy plotly
```

### Running the Script

Clone this repository and run the Python script:

```bash
git clone https://github.com/your-username/linear-regression-from-scratch.git
cd linear-regression-from-scratch
python main.py
```

---

## 🐍 Source Code (`main.py`)

```python
import numpy as np
import plotly.graph_objects as go

# Features (X) and Target (Y) values
X = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
Y = np.array([45.2, 55.1, 72.8, 88.4, 105.1, 118.9, 135.2, 152.0, 164.5, 185.3])

# Initial parameters
w, b, lr = 0.0, 0.0, 0.000001
lossHistory = []

# Training loop
for step in range(1000):
    prediction = w * X + b
    error = prediction - Y

    # Store current Mean Squared Error
    mse = (error ** 2).mean()
    lossHistory.append(mse)

    # Calculate gradients
    dw = 2 * (error * X).mean()
    db = 2 * error.mean()

    # Update parameters
    w -= lr * dw
    b -= lr * db

print(f"Final weights: w = {w:.4f}, b = {b:.4f}")
print(f"Final error (MSE): {lossHistory[-1]:.4f}")

# Plotting the loss curve
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(range(1000)),
    y=lossHistory,
    mode='lines',
    name='Mean Squared Error',
    line=dict(color='blue', width=2)
))

fig.update_layout(
    title='Training Loss Curve (Error over 1000 Steps)',
    xaxis_title='Step',
    yaxis_title='Mean Squared Error (Loss)',
    template='plotly_white'
)

fig.show()
```

---

## 🛠️ Tech Stack

- **Python 3**
- **NumPy** - Vectorized matrix operations
- **Plotly** - Interactive data visualization
