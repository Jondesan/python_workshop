import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Create a product of sine waves
x = np.linspace(0, 10, 1000)
y = np.sin(2 * np.pi * x) + np.sin(2 * np.pi * 2 * x) + np.sin(2 * np.pi * 3 * x) + np.random.normal(0, 0.1, len(x))



# Perform the FFT on the temperature data
frequencies = np.fft.fft(y)

N = len(y)
n = np.arange(N)
# get the sampling rate
sr = len(x)/(max(x)-min(x))
T = N/sr
freq = n/T
# Get the one-sided specturm
n_oneside = N//2
# get the one side frequency
f_oneside = freq[:n_oneside]

# Calculate the amplitudes of the frequency components
amplitudes = np.abs(frequencies)

# Plot the frequency spectrum
fig = plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
ax1.plot(1/f_oneside , amplitudes[:N//2]/n_oneside)
ax1.set_xticks(1/f_oneside[find_peaks(amplitudes[:N//2]/n_oneside, distance=1, height=.2)[0]])
ax1.set_xlim(0,1.5)
ax2.plot(x, y)

# Add labels to the plot
ax1.set_xlabel("Frequency")
ax1.set_ylabel("Amplitude")

fig.tight_layout()
# Show the plot
plt.show()