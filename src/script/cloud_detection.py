"""
Cloud detection is a couple of functions that detect clouds in a given image. With some additional parameters, it can be used to detect clouds in a larger image by splitting it into smaller images and then combining the results.
"""

import csv
import secrets

import numpy as np
import pandas as pd
import scipy
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from tqdm import tqdm
import pywt


def remove_background_phot(data: np.ndarray) -> np.ndarray:
    """
    Remove background from image using photutils
    """
    # remove background
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    # bkg_estimator = MedianBackground()

    bkg = Background2D(data, (50, 50), filter_size=(5, 5), sigma_clip=sigma_clip)
    diff = data - bkg.background  # type: ignore

    # subtract background
    diff = diff - np.min(diff)  # normalize to 0

    return diff


def remove_background_wavelet(data):
    """
    Remove background from image using wavelet transform as a filter
    """

    wavelet = "haar"
    max_level = 3
    LL = data

    background = np.zeros(np.shape(data), dtype=np.float32)
    for level in range(1, max_level + 1):
        coeffs = pywt.dwt2(LL, wavelet)
        LL, (LH, HL, HH) = coeffs
        all_details = np.kron(LH + HL + HH, np.ones((2**level, 2**level))) * (2**(-level)) * 2
        background += all_details

    diff = data - background
    diff = diff - np.min(diff)  # normalize to 0

    return diff


def remove_background_fourier(data, radius=25):
    """
    Remove background from image using fourier transform as a filter
    """
    # create kernel
    SIZE = data.shape

    # create it using Gaussian2DKernel
    mask1 = np.float32(
        Gaussian2DKernel(
            x_stddev=radius, y_stddev=radius, x_size=SIZE[1], y_size=SIZE[0]
        ).array
    )
    mask1 = mask1 / np.max(mask1)

    # high pass kernel = 1 - low pass kernel
    mask2 = (1 - mask1).clip(0, 1)

    # high pass first
    fft_shift = np.fft.fftshift(np.fft.fft2(data, axes=(0, 1)))
    fft_masked_high = np.multiply(fft_shift, mask2)
    img_high_pass = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_masked_high)))
    data_clipped = data - img_high_pass

    # fourier transform
    fft_shift = np.fft.fftshift(np.fft.fft2(data_clipped, axes=(0, 1)))

    fft_masked_low = np.multiply(fft_shift, mask1)
    fft_masked_high = np.multiply(fft_shift, mask2)

    img_high_pass = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_masked_high)))
    img_low_pass = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_masked_low)))

    # remove low pass and high pass from original
    diff = data_clipped - img_low_pass - img_high_pass
    diff = diff - np.min(diff)  # normalize to 0

    return diff


def detect_clouds(
    data: np.ndarray,
    overlap_x: int = 0,
    overlap_y: int = 0,
    min_area: int = 40,
    remove_background: str = "fourier",
    fourier_radius: float = 15,
):
    if remove_background == "fourier":
        diff = remove_background_fourier(data, radius=fourier_radius)
    elif remove_background == "wavelet":
        diff = remove_background_wavelet(data)
    elif remove_background == "photutils":
        diff = remove_background_phot(data)

    # blur the image a little
    gauss = scipy.ndimage.gaussian_filter(diff, sigma=3)
    # create mask by thresholding
    # threshold = np.percentile(gauss, 2)

    if remove_background == "fourier":
        # threshold = np.mean(gauss) - 2.5 * np.std(gauss)
        mean, median, stddev = sigma_clipped_stats(gauss, sigma=7)
        threshold = mean - 6 * stddev
    elif remove_background == "wavelet":
        mean, median, stddev = sigma_clipped_stats(gauss, sigma=3)
        threshold = mean - 1 * stddev
    elif remove_background == "photutils":
        mean, median, stddev = sigma_clipped_stats(gauss, sigma=5)
        threshold = mean - 3 * stddev

    mask = np.where(gauss < threshold, True, False)
    return mask_manipulate(mask, data, overlap_x, overlap_y, min_area)


def mask_manipulate(
    mask: np.ndarray,
    data: np.ndarray,
    overlap_x: int = 0,
    overlap_y: int = 0,
    min_area: int = 40,
):
    # group pixels into labels
    labels, nlabels = scipy.ndimage.label(mask)

    # sum the areas of mask with labels
    dust_areas = np.array(scipy.ndimage.sum(mask, labels, range(0, nlabels + 1))).astype(np.int32)

    # remove small clouds
    mask_small = dust_areas > min_area

    # remove clouds in overlap zone
    mask_overlap = np.ones(mask.shape, dtype=bool)
    mask_overlap[mask.shape[0] - overlap_y: mask.shape[0], 0: mask.shape[1]] = False
    mask_overlap[0: mask.shape[0], mask.shape[1] - overlap_x: mask.shape[1]] = False

    labels_outside_overlap = np.unique(mask_overlap * labels)

    # remove clouds touching the edges of processed image
    mask_edge = np.zeros(mask.shape, dtype=bool)
    mask_edge[0:1, :] = True
    mask_edge[:, 0:1] = True
    mask_edge[mask.shape[0] - 1: mask.shape[0], :] = True
    mask_edge[:, mask.shape[1] - 1: mask.shape[1]] = True
    labels_touching_edge = np.unique(mask_edge * labels)

    # remove clouds touching the black border
    objects_first = scipy.ndimage.find_objects(labels)
    masked_data = mask * (data - 1) + 1

    for cloud in range(0, nlabels):
        if (
            not np.all(masked_data[objects_first[cloud]])
            or not (cloud + 1) in labels_outside_overlap
            or (cloud + 1) in labels_touching_edge
        ):
            mask_small[cloud + 1] = False

    # remove small clouds and edge clouds
    remove_small_clouds = mask_small[labels.ravel()].reshape(labels.shape)
    dust_areas_new = dust_areas[mask_small]

    # unnecssary step, but I don't know how to do it otherwise
    new_labels, new_nlabels = scipy.ndimage.label(remove_small_clouds)

    mass_centers = scipy.ndimage.center_of_mass(
        data, new_labels, range(1, new_nlabels + 1)
    )
    objects = scipy.ndimage.find_objects(new_labels)

    min_values = np.zeros(new_nlabels)
    max_values = np.zeros(new_nlabels)
    circularity = np.zeros(new_nlabels)
    for i in range(len(mass_centers)):
        mask = new_labels[objects[i]] == i + 1
        min_values[i] = np.min(data[objects[i]])
        max_values[i] = np.max(data[objects[i]])
        circularity[i] = ((4 * np.pi * (np.sum(mask))) / np.power(np.sum(np.bitwise_xor(scipy.ndimage.binary_dilation(mask), mask)), 2))


    return mass_centers, objects, new_labels, dust_areas_new, remove_small_clouds, min_values, max_values, circularity


def format_for_ouput(
    mass_centers: np.ndarray,
    objects: np.ndarray,
    new_labels: np.ndarray,
    dust_areas_new: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
    circularity: np.ndarray,
    X: int = 0,
    Y: int = 0,
    path: str = "../../data/",
) -> list[dict[str, int]]:
    output = []
    # save data to file

    for i in range(len(mass_centers)):
        mask_file_name = (
            str(round(mass_centers[i][1]) + X)
            + "_"
            + str(round(mass_centers[i][0]) + Y)
            + "_"
            + secrets.token_urlsafe(5)
            + ".npy"
        )
        np.save(path + "masks/" + mask_file_name, new_labels[objects[i]] == i + 1)

        try:
            output.append(
                {
                    "x_center": round(mass_centers[i][1]) + X,
                    "y_center": round(mass_centers[i][0]) + Y,
                    "approx_size": int(dust_areas_new[i]),
                    "box_x1": objects[i][1].start + X,
                    "box_x2": objects[i][1].stop + X,
                    "box_y1": objects[i][0].start + Y,
                    "box_y2": objects[i][0].stop + Y,
                    "min_value": min_values[i],
                    "max_value": max_values[i],
                    "circularity": circularity[i],
                    "mask_file": mask_file_name,
                }
            )
        except Exception as e:
            print("Something went wrong:", e)

    return output


def save_reg(output: list, filename: str = "catalog", path: str = "../../data/"):
    """
    Save as a DS9 region file
    """

    region = """# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
physical
"""

    new_path = path + "regions/" + filename

    if path == "":
        new_path = filename

    if new_path[-4:] != ".reg":
        new_path += ".reg"

    for row in output:
        # example: box(63605.884,5115.8396,207.36,167.9616,0)
        region += "box({},{},{},{},0)\n".format(
            (row["box_x1"] + row["box_x2"]) / 2,
            (row["box_y1"] + row["box_y2"]) / 2,
            row["box_x2"] - row["box_x1"],
            row["box_y2"] - row["box_y1"],
        )

    with open(new_path, "w") as f:
        f.write(region)


def save_reg_and_csv(
    output: list, filename: str = "catalog", path: str = "../../data/"
):
    """Save as a CSV catalog and DS9 region file"""

    # save as csv
    csv_columns = output[0].keys()
    csv_file = path + filename + ".csv"
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for row in output:
            writer.writerow(row)

    # save as region file
    save_reg(output, filename, path)


def save_as_fits(
    fits_path: str,
    catalog_path: str,
    output_path: str,
    mask_path: str = "../data/masks/",
    box: tuple[int, int, int, int] = (0, 120000, 0, 12000),
):
    """
    Save the catalog and the masks as fits files by marking them with pixels.

    Example:
    ```python
    save_as_fits(
        "../../data/fits/PROMISE-Q1-8micron-filled-v0_3.fits",
        "../../data/catalog_v2.2.csv",
        "../../data/fits/test2.fits",
        "../../data/masks/",
        (0, 120000, 0, 12000),
    )
    ```
    """
    imageHDU = fits.open(fits_path, memmap=True, mode="denywrite")
    imageHDU.info()  # Should match with BOX
    data = np.array(imageHDU[0].data[box[2]: box[3], box[0]: box[1]])  # type: ignore
    header = imageHDU[0].header
    imageHDU.close()

    catalog = pd.read_csv(
        catalog_path,
        index_col=False,
        dtype={
            "x_center": np.int32,
            "y_center": np.int32,
            "approx_size": np.int32,
            "box_x1": np.int32,
            "box_x2": np.int32,
            "box_y1": np.int32,
            "box_y2": np.int32,
            "min_value": np.float32,
            "max_value": np.float32,
            "circularity": np.float32,
            "mask_file": str,
        },
    )

    error_counter = 0
    max_value = np.max(data)*1.1


    for i, row in tqdm(catalog.iterrows(), "Processing masks", len(catalog)):  # type: ignore
        mask = np.load(mask_path + row["mask_file"])

        # 1px outline mask
        # if mask.shape == (row["box_y2"] - row["box_y1"], row["box_x2"] - row["box_x1"]):
        #     data[row["box_y1"]: row["box_y2"], row["box_x1"]: row["box_x2"]] = (
        #         data[row["box_y1"]: row["box_y2"], row["box_x1"]: row["box_x2"]]
        #         + (mask + -1 * scipy.ndimage.binary_erosion(mask)) * 1000
        #     )
        # else:
        #     error_counter += 1
        #     mask.resize((row["box_y2"] - row["box_y1"], row["box_x2"] - row["box_x1"]))
        #     data[row["box_y1"]: row["box_y2"], row["box_x1"]: row["box_x2"]] = (
        #         data[row["box_y1"]: row["box_y2"], row["box_x1"]: row["box_x2"]]
        #         + (mask + -1 * scipy.ndimage.binary_erosion(mask)) * 3000
        #     )

        # fill and leave everything else as it is
        if mask.shape != (row["box_y2"] - row["box_y1"], row["box_x2"] - row["box_x1"]):
            error_counter += 1
            mask.resize((row["box_y2"] - row["box_y1"], row["box_x2"] - row["box_x1"]))

            
        data[row["box_y1"]: row["box_y2"], row["box_x1"]: row["box_x2"]] = (mask * max_value + data[row["box_y1"]: row["box_y2"], row["box_x1"]: row["box_x2"]]*(np.invert(mask)))


    print("Done! Incorrect boxes:", error_counter)

    fits.writeto(output_path, data, header, overwrite=True)


def npy_to_fits(npy_path: str, output_path: str):
    """
    Convert a numpy array to a fits file
    """
    data = np.load(npy_path)
    data = np.float32(data)
    fits.writeto(output_path, data, overwrite=True)


def read_regfile(path):
    with open(path, "r") as f:
        lines = f.readlines()

    output = []
    for line in lines:
        if line.startswith("box"):
            output.append(stringbox_to_box(line))
    return output


def stringbox_to_box(stringbox):
    line = stringbox.replace("(", "")
    line = line.replace(")", "")
    line = line.replace("box", "")
    line = line.replace("\n", "")
    line = line.split(",")

    return {
        "box_x1": round(float(line[0]) - float(line[2]) / 2),
        "box_x2": round(float(line[0]) + float(line[2]) / 2),
        "box_y1": round(float(line[1]) - float(line[3]) / 2),
        "box_y2": round(float(line[1]) + float(line[3]) / 2),
    }
