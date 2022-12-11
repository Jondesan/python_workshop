# Tätä tehtävää varten joudut mahdollisesti asentamaan paketin netCDF4.
# Tämä on ilmakehätieteissä paljon käytettyä tiedostoformaattia nc varten
# luotu paketti, jonka avulla voidaan käsitellä tämän tyyppistä dataa.

import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# U = itä-suunta, V = pohjois-suunta

frame = netCDF4.Dataset('ERA5_20111226.nc')
lat = frame.variables['latitude']
lon = frame.variables['longitude']
u_wind = frame.variables['u']
v_wind = frame.variables['v']

# Tässä tehtävässä tutkitaan tuulen suuntaa ja nopeutta.
# Tiedostossa on 2-ulotteista dataa, joka on tallennettu 4-ulotteiseen muotoon.
# Tiedosto on valmiiksi luettu sisään. Tehtävänäsi on muodostaa kaksi lämpökarttaa,
# yksi itä-suuntaan ja toinen pohjois-suuntaan. Tämän jälkeen voit piirtää 3-ulotteisen
# kuvaajan itä-suuntaisen tuulen nopeudelle pituus- ja leveysasteittain. Lopuksi voit
# piirtää 3-ulotteiseen kuvaajaan vakionopeuskäyriä käyttämällä esiteltyä contour komentoa.
# Kannattaa aloittaa määrittämällä x ja y arvot kaksiulotteiseen ruudukkoon.

# Tuulen nopeus itä- ja pohjois-suunnassa on tallennettu muuttujiin u_wind ja v_wind.
# Ne ovat 4-ulotteisia muuttujia, joiden ensimmäinen ulottuvuus on aika, toinen on korkeus.
# Pääset käsiksi koko kaksiulotteiseen nopeusdataan käyttämällä muuttujia u_wind[0,0,:,:] ja v_wind[0,0,:,:].
# Jos haluat tutkia miten nopeudet eroavat eri aikoina tai eri korkeuksissa, voit muuttaa ensimmäistä tai toista
# indeksiä. Aika ja korkeus koostuvat kumpikin kolmesta kerroksesta, joten mahdollisia indeksointeja ovat 0, 1 ja 2.