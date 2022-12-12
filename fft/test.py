import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz

def voigt(x, sigma, gamma, delta):
  z = ((x - delta) + 1j*gamma) / (sigma*np.sqrt(2))
  return wofz(z).real / (sigma*np.sqrt(2*np.pi))

# define the range of x values
x = np.linspace(-5, 5, 1000)

# compute the Voigt profile for sigma = 1 and gamma = 1
y = voigt(x, sigma=1, gamma=1)

# plot the result
plt.plot(x, y)
plt.show()
