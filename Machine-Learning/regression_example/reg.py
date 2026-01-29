import numpy as np
import matplotlib.pyplot as plt

# 1) Function to calculate a and b (lineart reg)
def compute_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    a = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    b = y_mean - a * x_mean

    return a, b


# 2) prediction function
def predict(x, a, b):
    return a * x + b


# 3) function to plot the graph
def plot_regression(x, y, a, b):
    y_pred = predict(x, a, b)

    plt.scatter(x, y, label="Real data")
    plt.plot(x, y_pred, label="regression line")
    plt.xlabel("years of experience")
    plt.ylabel("Salary (in thousands of DA))")
    plt.title("linear regression (manual calcul using functions)")
    plt.legend()
    plt.show()



# main program

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([30, 35, 45, 50, 60], dtype=float)

a, b = compute_regression(x, y)

print("a (slope) =", a)
print("b (intercept) =", b)
print("Equation : y = a*x + b")

# prediction
x_test = 3
print("salary predicted for 3 years =", predict(x_test, a, b))

# graph
plot_regression(x, y, a, b)
