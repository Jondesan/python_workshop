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

print(frame.variables['v'])

x, y = np.meshgrid(lon, lat, indexing='xy')

fig, ax = plt.subplots(1,2)

ax[0].pcolormesh(x, y, u_wind[0,1,:,:])
ax[1].pcolormesh(x, y, v_wind[0,0,:,:])
#plt.colorbar(plot1, ax=[ax, ax2])

ax2 = plt.figure().add_subplot(111, projection='3d')
ax2.plot_surface(x, y, u_wind[0,0,:,:], cmap='plasma', alpha=0.7)
ax2.contour(x, y, u_wind[0,0,:,:], zdir='z', offset=0, levels=10, cmap='coolwarm')

plt.show()