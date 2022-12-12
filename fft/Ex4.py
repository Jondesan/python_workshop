import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Create a product of sine waves
x = np.linspace(0, 10, 1000)
y = np.sin(2 * np.pi * x) + np.sin(2 * np.pi * 2 * x) + np.sin(2 * np.pi * 3 * x) + np.random.normal(0, 0.1, len(x))


# Tässä tehtävässä demonstroidaan FFT:n toimintaa raaimmillaan. Yllä on tuotettu jaksollista dataa
# käyttämällä kolmen eritaajuisen sini-funktion summaa. Signaaliin on lisätty hieman kohinaa.
# Fourier muunnos paljastaa pohjimmiltaan signaalin taajuusspektrin, joten kolmen siniaallon summassa
# pitäisi luonnollisesti esiintyä kolmea eri taajuutta.

# Suorita signaalille FFT analyysi ja piirrä taajuusspektri. Tässäkin tehtävässä kuvaajan tulkintaa
# helpottaa, jos x-akseli skaalataan taajuuden sijaan ajan mukaan. Spektrin piirto kannattanee tehdä
# käyttämällä plt.stem() funktiota.