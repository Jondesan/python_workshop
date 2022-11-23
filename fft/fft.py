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
    #print(img)


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
    print(peaks)
    X = range(rad_prof.shape[0])
    popt, pcov = opt.curve_fit(lorentzian, X, rad_prof, [peaks[0], 1, 1])

    ax[0,0].imshow(img, cmap='gray')
    ax[0,1].imshow(magnitude_spectrum, cmap='gray')
    print(np.max(rad_prof))
    ax[1,0].plot(X, rad_prof)
    ax[1,0].scatter(peaks, rad_prof[peaks], c='r', marker='x', zorder=3)
    x_param = np.linspace(0,len(X), 100)
    ax[1,0].plot(x_param, lorentzian(x_param, popt[0], popt[1], popt[2]), c='g')

    ax[1,1].plot(X, np.gradient(rad_prof))

    plt.show()