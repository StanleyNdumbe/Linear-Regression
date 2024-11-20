import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv('NairobiOfficePriceEx.csv')
office_size = data['SIZE'].values
office_price = data['PRICE'].values

# Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function for Linear Regression
def gradient_descent(x, y, learning_rate=0.0001, epochs=10):
    m, c = np.random.rand(2)  # random values for slope (m) and intercept (c)
    n = len(y)

    # Iteratively update weights and print MSE for each epoch
    for epoch in range(epochs):
        y_pred = m * x + c
        error = mean_squared_error(y, y_pred)

        # Compute gradients
        m_grad = -(2 / n) * sum(x * (y - y_pred))
        c_grad = -(2 / n) * sum(y - y_pred)

        # Update weights
        m -= learning_rate * m_grad
        c -= learning_rate * c_grad

        print(f"Epoch {epoch + 1}: MSE = {error:.2f}")
    return m, c

# Train the model
slope, intercept = gradient_descent(office_size, office_price)

# Plot the data points and the line of best fit
plt.scatter(office_size, office_price, color="blue", label="Data points")
plt.plot(office_size, slope * office_size + intercept, color="red", label="Line of best fit")
plt.xlabel("Office Size (sq. ft)")
plt.ylabel("Office Price (KES)")
plt.legend()
plt.title("Office Size vs. Office Price (Nairobi)")
plt.show()

# Predict office price for 100 sq. ft
predicted_price = slope * 100 + intercept
print("Price for 100 = ",predicted_price)