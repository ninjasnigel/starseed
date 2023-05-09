import matplotlib.pyplot as plt
import numpy as np
import scipy
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

print("Starting...")
FITS_PATH = 'PROMISE-Q1-8micron-filled-v0_3.fits'

# Open currimg and display info using memmap to not load the whole image into memory
# imageHDU = fits.open(FITS_PATH, memmap=True, mode='denywrite')
imageHDU = fits.open(FITS_PATH)
imageHDU.info()
print("Opened file")

# Get display image as Numpy Array called data
data = np.array(imageHDU[0].data)
print("Loaded data")
imageHDU.close() # Close the file
print("Closed file")

sigma_clip = SigmaClip(sigma=3.0)
print("Created sigma clip")
bkg_estimator = MedianBackground()
print("Created background estimator")
bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
print("Created background")

np.save('PROMISE-Q1-8micron-filled-v0_3-no-background.npy', data-bkg.background)
# crashes on 16GB RAM machine, so use numpy and the use npy_to_fits.py to convert to FITS
# fits.writeto('PROMISE-Q1-8micron-filled-v0_3-no-background.fits', data-bkg.background, overwrite=True)
print("Wrote file, done!")