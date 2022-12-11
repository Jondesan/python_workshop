# %%
%matplotlib widget

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.optimize as opt
import radialProfile

def lorentzian( x, x0, a, gam ):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)

def radial_profile(data):
    center = (np.round(data.shape[0]), np.round(data.shape[1]))
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

if __name__ == "__main__":
    img = cv2.imread('test4.png',cv2.IMREAD_GRAYSCALE)


    # Create the FFT of img
    freq_space = np.fft.fft2(img)
    # Shift the Fourier transform to the center
    fshift = np.fft.fftshift(freq_space)
    # Calculate the magnitude spectrum
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    # Calculate the radial profile
    rad_prof = radialProfile.azimuthalAverage(magnitude_spectrum)
    rad_prof = rad_prof[~np.isnan(rad_prof)]
    rad_prof = (rad_prof-np.min(rad_prof))/(np.max(rad_prof)-np.min(rad_prof))
    
    fig, ax = plt.subplots(2,2, figsize=(16,8))

    peaks, props = scipy.signal.find_peaks(rad_prof, width=4)
    #print(peaks)
    
    X = range(rad_prof.shape[0])
    popt, pcov = opt.curve_fit(lorentzian, X, rad_prof, [peaks[0], 1, 1])

    ax[0,0].imshow(img, cmap='gray')
    ax[0,1].imshow(magnitude_spectrum, cmap='gray')
    
    ax[1,0].plot(X, rad_prof)
    ax[1,0].scatter(peaks, rad_prof[peaks], c='r', marker='x', zorder=3)
    x_param = np.linspace(0,len(X), 100)
    ax[1,0].plot(x_param, lorentzian(x_param, popt[0], popt[1], popt[2]), c='g')

    rev_im = np.fft.ifft2(freq_space)
    ax[1,1].imshow(np.abs(rev_im), cmap='gray')

    plt.show()

# %%
x = np.linspace(-5, 5, 101)
y = np.linspace(-5, 5, 101)
# full coordinate arrays
xx, yy = np.meshgrid(x, y)
zz = np.sqrt(xx**2 + yy**2)
xx.shape, yy.shape, zz.shape

# %%
# sparse coordinate arrays
xs, ys = np.meshgrid(x, y, sparse=True)
zs = np.sqrt(xs**2 + ys**2)
xs.shape, ys.shape, zs.shape

# %%
ax1 = plt.figure().add_subplot()
h = plt.contourf(x, y, zs)
plt.axis('scaled')
plt.colorbar()
plt.show()

# %%
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(projection='3d')


ax.plot_surface(xs, ys, zs, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                alpha=0.3, cmap=plt.cm.YlGnBu_r)

#for angle in range(0, 360):
ax.view_init(30, 35)

ax.contour(xx, yy, zs, zdir='z', offset=0, levels=10, cmap='coolwarm')
ax.contour(xx, yy, zs, zdir='x', offset=-6, levels=10, cmap='coolwarm')
ax.contour(xx, yy, zs, zdir='y', offset=-6, levels=10, cmap='coolwarm')

ax.set(xlim=(-6, 6), ylim=(-6, 6), zlim=(0, 7),
       xlabel='X', ylabel='Y', zlabel='Z')


plt.show()


# %% [markdown]
# ## 3D plotting and 3-dimensional data
# 
# - Many systems and their behaviour can be studied using multi-dimensional data
#     - From simple systems such as an electric potential over a conducting surface to abstractions of physical systems, such as micromagnetic structure energies w.r.t. 2 parameters
# - If possible, problems should always be reduced such that the analysis is easier to understand visually
#     - In scientific publishing clear visualization is key: it is much easier to explain system behaviour using simple to understand visualization of data instead of throwing complicated and cumbersome graphs at people
# - Sometimes reducing the problem to a simpler one isn't possible or reasonable.
#     - Sometimes more complex visualization techniques give a much fuller picture of the system behaviour we are studying
# 
# 
# ### Multi-dimensional data and handling it
# 
# <i>One of the best ways to handle data in Python is with Pandas library</i>.<br>
# Pandas is specifically developed for data analysis and handling and presents a <b>lot</b> of pre-built tools for manipulating complex dataframes.
# 
# In addition <i>numerical python</i>, or Numpy as its called among friends, is also a great tool and we'll be employing that very soon.
# 
# Let's take a look at a couple examples.
# 
# 
# #### You can mostly skip this first code block for now, since Pandas will be the topic of Nico and Teemu in the coming workshops.

# %%
# Import the necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Calculate a Lorentzian value
def Lorentzian(x, x0, a, gam):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)

np.random.seed(42)

X = np.linspace(-5,5,100)
Y = Lorentzian(X, 0, 1, 1) + np.random.normal(0, 0.1, 100)
# Create a Pandas dataframe filled with random numbers in a Lorentzian distribution
df = pd.DataFrame({'X': X,'Y': Y})

# Show the dataframe transposed (for better visualization)
df.T

# %% [markdown]
# So now we have a 2-dimensional <b>dataframe</b>, a Pandas data structure, to store some created data. In actual application this could be simulated, calculated or experimentally measured data.
# 
# Let us see how we did with the creation of the data. Let's plot it!

# %%
# Create a figure and plot the data
ax = plt.figure().add_subplot()
ax.plot(df['X'], df['Y'], 'o')
plt.show()

# %% [markdown]
# So now we see very clearly the data we created is a Lorentzian distribution with some error added in for good measure. Now then, how should we proceed if instead we wanted to create a 3-dimensional image?
# 
# So what we want to do is plot a 3rd variable as a function of other 2, say if we wanted to visualize a similar Lorentzian peak in 3D space. Then the 2 known variables would be our $x$ and $y$ spatial directions and the 3rd would be the height $z$ of the distribution at a given point $(x,y)$.
# 
# In this problem we can employ Numpy and one of its incredibly useful functions: a <b>meshgrid</b>.
# 
# Meshgrid takes in as arguments $n$ arrays (or what are known as list-like objects). It then determines these as the "axes" of an $n$-dimensional array. The function returns $n$ $n$-dimensional arrays, for each given dimension and fills in the values of the corresponding coordinate. Afterwards these arrays can be used to calculate, say, function values at each of the points in the defined space. So meshgrid shines the brightest when you need to <b>evaluate scalar or vector fields</b>.

# %%

# Create meshgrids for x and y
gridx, gridy = np.meshgrid(X, X, indexing='xy')

gridx, gridy


# %%

# Calculate z values
zvals = Lorentzian(gridx, 0, 1, 1) * Lorentzian(gridy, 0, 1, 1)

#zvals = (Lorentzian(gridx, 0, 1, 1) + np.random.normal(0, 0.1, 100)) * (Lorentzian(gridy, 0, 1, 1) + np.random.normal(0, 0.1, 100))

ax = plt.figure(figsize=(12,8)).add_subplot(projection='3d')
ax.plot_wireframe(gridx, gridy, zvals, cmap='viridis', rstride=5, cstride=5)

plt.show()

# %% [markdown]
# Now then! We have acquired some tools to visualize multidimensional data or, say 2-dimensional scalar fields for examples. What if we had some actual data to visualize. Well gladly we do! So let's do some actual science!
# 
# Let's read in some actual measurement data:

# %%
# Load in the equipotential dataset, courtesy of Simo Soini
df = pd.read_csv('equipotential.csv', index_col=0, header=0)
df

# %%
ax = plt.figure(figsize=(10,6)).add_subplot()
image = ax.pcolormesh(df, cmap='seismic')
plt.colorbar(image)
plt.show()

# %% [markdown]
# Okay! Now we are cooking! How about a 3-dimensional representation?

# %%
# Create meshgrids for x and y positions
# Don't mind the map function too much, it only converts the dataframe 
# axis labels to floats so they can be passed to matplotlib
X, Y = np.meshgrid(df.columns.map(float), df.index, indexing='xy')
ax = plt.figure(figsize=(12,8)).add_subplot(projection='3d')

Z = df.values

image = ax.plot_surface(X, Y, Z, cmap='seismic')
plt.colorbar(image)
plt.show()

# %% [markdown]
# Now this is all well and good. But what if we wanted the data to look a bit smoother. If we could go back to redo the measurements that would obviously be our top choice. But how should we proceed when that is not possible?
# 
# There are many ways to estimate intermediate values between measured ones. One way to smoothen this kind of data is <i>interpolation</i>. Interpolation estimates the values between known ones by different methods. Typical methods are linear, nearest neighbour, cubic and quintinc interpolation. To those familiar with image processing these terms may ring a bell: When image resolution is increased the new image is actually an interpolation of the original one. The resolution is increased by estimating the intermediate values between known pixels using interpolation.

# %%
import scipy.interpolate as interp

ax = plt.figure(figsize=(12,8)).add_subplot(211)

x = np.linspace(0, 29, 100)
y = np.linspace(0, 21, 100)


fit_points = [df.columns.map(float), df.index]
values = Z.T

ut, vt = np.meshgrid(x, y, indexing='xy')
test_points = np.array([ut.ravel(), vt.ravel()]).T

interpolator = interp.RegularGridInterpolator(fit_points, values, bounds_error=False)
interpolated_z = interpolator(test_points, method='linear', ).reshape(ut.shape)
interpolated_z2 = interpolator(test_points, method='quintic', ).reshape(ut.shape)


image = ax.pcolormesh(df, cmap='seismic')

ax2 = plt.subplot(223)
ax3 = plt.subplot(224)

ax2.pcolormesh(interpolated_z, shading='gouraud', cmap='seismic')
ax2.set_xlim(0, 96)
ax2.set_ylim(0, 95)

ax3.pcolormesh(interpolated_z2, shading='gouraud', cmap='seismic')
ax3.set_xlim(0, 96)
ax3.set_ylim(0, 95)

plt.colorbar(image, ax=[ax, ax2, ax3])

plt.show()

# %%

ax = plt.figure(figsize=(12,8)).add_subplot(211)

ax.pcolormesh(df, cmap='seismic')

ax2 = plt.subplot(223)
ax3 = plt.subplot(224)

ax2.pcolormesh(interpolated_z, shading='gouraud', cmap='seismic')
newx, newy = np.meshgrid(np.linspace(0,100,100), np.linspace(0,100,100), indexing='xy')
ax2.contour(newx, newy, interpolated_z, levels=20, cmap='coolwarm')
ax2.set_xlim(0, 96)
ax2.set_ylim(0, 95)

ax3.pcolormesh(interpolated_z2, shading='gouraud', cmap='seismic')
ax3.contour(newx, newy, interpolated_z, levels=20, cmap='coolwarm')
ax3.set_xlim(0, 96)
ax3.set_ylim(0, 95)

plt.colorbar(image, ax=[ax, ax2, ax3])

plt.show()

# %%
ax = plt.figure(figsize=(12,8)).add_subplot(projection='3d')


ax.plot_surface(ut, vt, interpolated_z, cmap='seismic', alpha=0.5)

ax.contour(ut, vt, interpolated_z, zdir='z', offset=0, levels=10, cmap='coolwarm')
ax.contour(ut, vt, interpolated_z, zdir='x', offset=0, levels=6, cmap='coolwarm_r')

ax.set(xlim=(0, 30), zlim=(0, 10), xlabel='X', zlabel='Z')

plt.colorbar(image, ax=ax)

plt.show()


