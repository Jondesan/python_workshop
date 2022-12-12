import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the temperature data using pandas
temp_data = pd.read_csv("temp_kumpula3.csv")
print(temp_data)
# Convert the data to a NumPy array
temperatures = temp_data["Air temperature (degC)"].to_numpy()

# Tehtävänäsi on käyttää sisään luettua lämpötiladataa joka on mitattu rakkaassa Kumpulassamme.
# Tarkoituksena on tehdä FFT (Fast Fourier Transform) lämpötiladatasta ja piirtää saatu spektri.
# Tästä spektristä pitäisi pystyä tulkitsemaan millä taajuudella lämpötila vaihtelee, toisin sanoen
# esiintyykö ilmanlämpötilassa periodisuutta.

# Huomaa, että Fourier muunnoksen spektri on kompleksiarvoinen, joten saadaksesi taajuusspektrin
# sinun pitää laskea absoluuttinen arvo FFT:stä saadusta spektristä.
# FFT spektrin kuvaajan piirtämiseen hyvä erillisfunktio on plt.stem().

# Tehtävän simppeli versio (tuota FFT spektri ja piirrä sen  kuvaaja) on verrattain yksinkertainen.
# Jos kuvaajaa haluaa kuitenkin fiksusti tulkita, kannattaa jokainen taajuuspiste skaalata siten,
# että ne vastaavatkin ajanarvoja taajuuden sijaan. Tällöin kuvaajasta nähdään selkeästi millä
# aikavälillä jaksollista käytöstä esiintyy, jos sitä esiintyy.