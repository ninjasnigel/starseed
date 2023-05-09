#import cv2
from PIL import Image, ImageOps
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
import os
#import matplotlib.colormaps as cm
import matplotlib

def sigma_plots(fitspath):
    with fits.open(fitspath) as hdul:
        data = hdul[0].data
    image = np.array(data)
    for i in range(3, 11):
        mean, std = np.mean(image), np.std(image)
        image_standardised= (image - mean) / std
        image_log = np.log(image_standardised - np.min(image_standardised) + 1)
        
        # Calculate threshold value
        min_val = np.min(image_log[image_log > 0])
        thresh = min_val - 0.01 * min_val
        
        # Truncate values below threshold
        image_log[image_log < thresh] = thresh

        image_clip = sigma_clip(image_log, sigma=i, maxiters=5, copy=True)

        plt.imshow(image_clip, cmap='gray')
        plt.title(f"Sigma = {i}")
        plt.show()

        plt.hist(image_clip.ravel(), bins=1000, density=True, log=True)
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Pixel Values for Sigma = {i}")
        plt.show()

def fits_to_png(folder_path, color_map):
    """
    For creating images to train cnn on
    Takes a folder of fits files and saves them as png files
    - Saves the png files in the same folder as the fits files
        - ie the training and val folders
    Applies sigma clip, standardisation, log scaling, and normalisation
    sigma = 4 atm, might be changed in future
    sigma = 4 because "best" histogram. Otherwise histogram was very discrete. Cba to fix atm
    #TODO fix other sigmas? See sigma_plots function for testing
    """
    
    fits_files = [f for f in os.listdir(folder_path) if f.endswith('.fits')]
    for fits_file in fits_files:
        fits_path = os.path.join(folder_path, fits_file)
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
        image = np.array(data)
        #Standardises the image
        mean, std = np.mean(image), np.std(image)
        image_standardised= (image - mean) / std
        #Log scales the image
        image_log = np.log(image_standardised - np.min(image_standardised) + 1)
        #Finds the sigma clipped bounds for the image
        image_clip = sigma_clip(image_log, sigma=4, maxiters=5, copy=True, masked = False, return_bounds=True)
        #Sets the lower and upper bounds for the image
        im_lower = image_clip[1]
        im_upper = image_clip[2]

        #Set values above upper bound to upper bound for image_log, and values below lower bound to lower bound
        #Because sigma clipping can only return a masked array which doesn't save as png correctly
        # or an array with missing values for some godforsaken reason (which can't be turned in to an image at all)
        image_log[image_log > im_upper] = im_upper
        image_log[image_log < im_lower] = im_lower
        image_clip_norm = (image_log - np.min(image_log)) / (np.max(image_log) - np.min(image_log))

        #Converts the image to a color image if color_map is not gray
        image_color = matplotlib.colormaps.get_cmap(color_map)(image_clip_norm)
        #Save png file using PIL
        png_file = os.path.splitext(fits_file)[0] + '.png'  # Change file extension to .png
        png_path = os.path.join(folder_path, png_file)  # Set output path to folder_path

        if color_map == "gray":
            img = Image.fromarray(np.uint8(image_clip_norm * 255)).convert("L")
        else:
           img = Image.fromarray(np.uint8(image_color * 255))
        img.save(png_path, 'PNG')


def preprocess(img_data, sig = 4, color_map = "gray"):

        image = img_data
        #Standardises the image
        mean, std = np.mean(image), np.std(image)
        image_standardised= (image - mean) / std
        #Log scales the image
        image_log = np.log(image_standardised - np.min(image_standardised) + 1)
        #Finds the sigma clipped bounds for the image
        image_clip = sigma_clip(image_log, sigma=sig, maxiters=5, copy=True, masked = False, return_bounds=True)
        #Sets the lower and upper bounds for the image
        im_lower = image_clip[1]
        im_upper = image_clip[2]

        #Set values above upper bound to upper bound for image_log, and values below lower bound to lower bound
        #Because sigma clipping can only return a masked array which doesn't save as png correctly
        # or an array with missing values for some godforsaken reason (which can't be turned in to an image at all)
        image_log[image_log > im_upper] = im_upper
        image_log[image_log < im_lower] = im_lower
        image_clip_norm = (image_log - np.min(image_log)) / (np.max(image_log) - np.min(image_log))

        #Converts the image to a color image if color_map is not gray
        image_color = matplotlib.colormaps.get_cmap(color_map)(image_clip_norm)

        if color_map == "gray":
            img = Image.fromarray(np.uint8(image_clip_norm * 255)).convert("L")
        else:
            img = Image.fromarray(np.uint8(image_color * 255))
        return img

def feed_cnn(fits_loc, overlap, chunk_size = 640, color_map = "gray", save_dir = "./saved_chunks"):
    #open the original fits image, and loop over it, saving each image as a png
    with fits.open(fits_loc) as hdul:
        data = hdul[0].data
    image = np.array(data)
    height, width = image.shape
    
    # create the directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    indx = 0
    #Loop over the image in 640x640 chunks, with a overlap of overlap pixels
    #hardcoded cap on y to ignore black regions
    for y in range(3000, 9000, chunk_size - overlap):
        for x in range(0, width - chunk_size, chunk_size - overlap):
            # Extract a chunk of the image
            chunk = image[y:y+chunk_size, x:x+chunk_size]
            # Preprocess the chunk and save it as a png
            chunk_name = os.path.join(save_dir, f"{indx:05d}_x_{x}_y_{y}.png")
            im = preprocess(chunk, 4, "gray")
            im.save(chunk_name)
            indx += 1
#pathsss = r"E:\\Kandidatarbete FITS\\GitLab Repo\\starseed\\PROMISE-Q1-8micron-filled-v0_3.fits"
#feed_cnn(pathsss, 320, 640, "gray")


def model_to_reg(res):
    pass
    #Converts the model output to a region file for DS9
    #res is the output of the model



def calculate_global_coords(detections, tile_size = 640, overlap= 320):
    "might be helpful. takes the detected values from the model and outputs the global coordinates."

    global_coords = []
    for detection in detections:
        tile_coords = detection[:4]
        confidence = detection[4]
        tile_x, tile_y, tile_width, tile_height = tile_coords
        global_x = tile_x + tile_size * (tile_x // (tile_size - overlap))
        global_y = tile_y + tile_size * (tile_y // (tile_size - overlap))
        global_width = tile_width
        global_height = tile_height
        global_detection = (global_x, global_y, global_width, global_height, confidence)
        global_coords.append(global_detection)
    return global_coords
    
def calc_global_coord(res):

    results = res
    overlap = 320
    x_chunks = 373
    y_chunks = 19

    bbox_list = []
    x = -1
    y = 0
    for i in range(len(results)):
        image = results[i].boxes
        x += 1
        if x == int(x_chunks):
            x = 0
            y += 1
        for box in image:
            bbox_x = float(box.xywh[0][0])
            bbox_y = float(box.xywh[0][1])
            bbox_w = float(box.xywh[0][2])
            bbox_h = float(box.xywh[0][3])

            global_x = bbox_x + x * overlap
            global_y = bbox_y + y * overlap + 3000

            bbox_list.append((global_x, global_y, bbox_w, bbox_h))

    with open("E:\\Kandidatarbete FITS\\GitLab Repo\\starseed\\saved_chunks\\bbox_list.txt", "w") as file:
        for box in bbox_list:
            x, y, w, h = box
            file.write("box({},{},{},{},0)\n".format(x, y, w, h))




#fits_to_png(r"E:\Kandidatarbete FITS\GitLab Repo\starseed\cnn\datasets\dataset\valid\images")
#fits_to_png(r"E:\Kandidatarbete FITS\GitLab Repo\starseed\cnn\datasets\dataset\train\images")
#fits_to_png(r"C:\Users\mull_\Desktop\Globala System\Kandidatarbete\Gitlab Repo\starseed-1\cnn\fits_for_annotation", "gist_heat")
#fits_to_png("/Users/adamjohansson/Desktop/Skola/Kandidatarbete/starseed/cnn/fits_for_annotation", "gray")
#fits_to_png("/Users/adamjohansson/Desktop/Skola/Kandidatarbete/starseed/cnn/fits_for_annotation", "gray")