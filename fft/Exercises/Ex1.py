import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d

df = pd.read_csv('surface.txt', sep='\s', header=None, names=['x', 'y', 'z'])

ax = plt.figure().add_subplot(111, projection='3d')

# Yllä ohjelma lukee tiedoston ja tallentaa sen pandas dataframeen.
# Taulukossa on kolme saraketta, joista ensimmäinen on x-koordinaatti,
# toinen y-koordinaatti ja kolmas z-koordinaatti.
# Taulukon sarakkeita voidaan kutsua komennoilla df['x'], df['y'] ja df['z'].

# Tässä tehtävässä halutaan piirtää 3-ulotteinen hajontakuvaaja taulukon 3-ulotteisista pisteistä.
# Pisteet muodostavat helposti havaittavan pinnan kun ne piirretään 3-ulotteisena.

# Kuvaajan piirtämisen jälkeen voit harjoitella kuinka saisit värjättyä pisteet niiden paikan mukaan.
# Yritä siis värjätä pisteet siten, että x-koordinaatin arvon kasvaessa pisteen väri muuttuu kirkkaammaksi.
# Tässä kannattaa käyttää apuna colormappeja jotka määräävät käytettävän väriskaalan.

# Tämä on hyvin yksinkertainen esimerkki 3-ulotteisesta kuvaajasta, jossa annettu data vastaa fyysisiä koordinaatteja.
# Oheinen data on Helsingin yliopiston avaruusplasmafysiikan työryhmän simuloimaa dataa.

# Huomaa, että tehtävä ei onnistu suoraan näytetyillä esimerkeillä, sillä nyt data ei vastaa skalaarikenttää. Siksi 
# sen kuvaamiseen tarvitaan hajontakuvaaja (scatter plot).
