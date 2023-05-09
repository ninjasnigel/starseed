from astropy.io import fits
import astropy
import numpy
import scipy
import matplotlib.pyplot as plt

IMAGE_1_PATH = 'Q1-latest-whigal-85.fits'
IMAGE_2_PATH = '../../data/fits/PROMISE-Q1-8micron-filled-v0_3.fits'

FITS_PATH = IMAGE_2_PATH
# 2 är den vi ska använda

SIZE = 1000
X = 15000
Y = 5500

def testfits():
    # Open currimg and display info
    imageHDU = fits.open(FITS_PATH)
    imageHDU.info()

    # Get display image as Numpy Array called data
    data = imageHDU[0].data
    height, width  = numpy.shape(data)
    print(data[0][0])



    images = divide_image(1000, data)
    print(len(images))

    # Display part of the imported image
    plt.imshow(images[3])
    plt.show()
    print('done')

def main():
    testfits()

def divide_image(size, image):
    height, width  = numpy.shape(image)
    images = []
    for i in range(0, width-size, size):
        for j in range(0, height-size, size):
            images += [image[i:i+size,j:j+size]]
    return images


if __name__ == "__main__":
    main()