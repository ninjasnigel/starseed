import copy
from astropy.stats import sigma_clipped_stats
import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans, convolve_fft

def removestars(data):
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    subData = copy.deepcopy(data)-copy.deepcopy(median)
    subData[subData > 5.*std] = np.nan
    kernel = Gaussian2DKernel(x_stddev=8)
    fixed_img = interpolate_replace_nans(subData, kernel, convolve_fft)
    return fixed_img