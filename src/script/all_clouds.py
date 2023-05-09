import glob
import os

import numpy as np
from astropy.io import fits
from cloud_detection import *

print("Starting...")
FITS_PATH = '../../data/fits/PROMISE-Q1-8micron-filled-v0_3.fits'
SIZE = (120000, 12000) # 120000, 12000

# Open img using memmap to not load the whole image into memory
imageHDU = fits.open(FITS_PATH, memmap=True, mode='denywrite')
imageHDU.info() # Should match with SIZE
print("Opened file")

output = []

# process small squares of the image
batchsize_x = 12000
batchsize_y = 12000
overlap = 4000

# delete all files in masks folder
print("Deleting old masks")
files = glob.glob('../../data/masks/*.npy')
for f in files:
    os.remove(f)

print("Deleted all old masks")

for X in range(0, SIZE[0], batchsize_x):
    for Y in range(0, SIZE[1], batchsize_y):
        overlap_x = overlap
        overlap_y = overlap
        if X+batchsize_x+overlap_x > SIZE[0]:
            overlap_x = 0

        if Y+batchsize_y+overlap_y > SIZE[1]:
            overlap_y = 0

        print("Processing square at X: ({}, {}), Y: ({}, {})".format(X, X+batchsize_x+overlap_x, Y, Y+batchsize_y+overlap_y))

        data = np.array(imageHDU[0].data[Y:Y+batchsize_y+overlap_y, X:X+batchsize_x+overlap_x], dtype=np.float32) # type: ignore

        # do stuff with data
        mass_centers, objects, new_labels, dust_areas_new, remove_small_clouds, min_values, max_values, circularity = detect_clouds(data, overlap_x, overlap_y, 25, "fourier", 25)
        output += format_for_ouput(mass_centers, objects, new_labels, dust_areas_new, min_values, max_values, circularity, X, Y)
    

save_reg_and_csv(output)
print("Done")