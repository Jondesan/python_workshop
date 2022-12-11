import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the temperature data using pandas
temp_data = pd.read_csv("temp_kumpula3.csv")
print(temp_data)
# Convert the data to a NumPy array
temperatures = temp_data["Air temperature (degC)"].to_numpy()

# Perform the FFT on the temperature data
frequencies = np.fft.fft(temperatures)

N = len(frequencies)
n = np.arange(N)
# get the sampling rate and frequency
# (one measurement per day)
sr = 1
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
ax1.plot(1/f_oneside, amplitudes[:N//2]/max(amplitudes[:N//2]))
ax2.plot(temperatures)

# Add labels to the plot
ax1.set_xlabel("Frequency")
ax1.set_ylabel("Amplitude")

fig.tight_layout()
# Show the plot
plt.show()