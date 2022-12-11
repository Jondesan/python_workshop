import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Lorentzian function
def lorentzian(x, amplitude, x0, gamma):
    return amplitude * gamma**2 / ((x - x0)**2 + gamma**2)

# Sum of Lorentzian functions
def sum_of_lorentzians(x, *params):
    y = 0
    for i in range(0, len(params), 3):
        y += lorentzian(x, params[i], params[i+1], params[i+2])
    return y

# Generate sample data
x = np.linspace(-10, 10, 1000)
y = sum_of_lorentzians(x, 1, 0, 1, 1, 5, 1, 1, -5, 1) + np.random.normal(0, 0.1, len(x))

# Fit the model to the data
params, _ = curve_fit(sum_of_lorentzians, x, y, (1, 1, 1, 1, 1, 1, 1, 1, 1), maxfev=100000000)

# Print the fitting parameters
print(params)


plt.scatter(x, y, s=2, c='b', label='data')
plt.plot(x, sum_of_lorentzians(x, *params), 'g-', label='fit')

plt.show()