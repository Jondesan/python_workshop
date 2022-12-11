import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d

df = pd.read_csv('surface.txt', sep='\s', header=None, names=['x', 'y', 'z'])

ax = plt.figure().add_subplot(111, projection='3d')

ax.scatter(df['x'], df['y'], df['z'], c=df['x'], cmap='plasma', s=5)

plt.show()