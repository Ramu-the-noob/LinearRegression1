import numpy as np
import plotly.graph_objects as go

X = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) # independents / features i.e. input values
Y = np.array([45.2, 55.1, 72.8, 88.4, 105.1, 118.9, 135.2, 152.0, 164.5, 185.3]) # dependents / targets i.e. observed values

# w = weights, start at 0
# b = bias, start at 0
# lr = learning rate, set low
w, b, lr = 0.0, 0.0, 0.000001
lossHistory = []  # List to store error values over time

for step in range(1000):
    prediction = w * X + b
    error = prediction - Y # difference between predicted and actual value

    # Calculate Mean Squared Error (MSE) and save it
    mse = (error ** 2).mean()
    lossHistory.append(mse)

    dw = 2 * (error * X).mean() # goal is to make dw close to zero
    db = 2 * error.mean() # bias change needed

    # gradient descent, -= means go to direction reducing error
    w -= lr * dw
    b -= lr * db

print(f"Final weights: w = {w:.4f}, b = {b:.4f}") # final weights and bias
print(f"Final error (MSE): {lossHistory[-1]:.4f}") # final error 

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