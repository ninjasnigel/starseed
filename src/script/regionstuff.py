from astropy.io import fits
import numpy as np
import glob
import os 
import matplotlib.pyplot as plt
import platform
import script.cloud_detection as cd

def copy_fits_header(source_path, target_path):
    # Read the header from the source FITS file
    with fits.open(source_path) as source_file:
        headr = source_file[0].header
    # Write the header to the target FITS file
    with fits.open(target_path, mode='update') as target_file:
        for hdu in target_file:
            hdu.header.update(headr)
    return print("Header copied")

def region_to_fits(region_path, fits_path, pixel_width, pixel_height): 
    """
    Finds all region files in /cnn/regions_to_fits_file, and creates a fits file of 
    pixel_width*pixel_height size, centered on the center of the region.
    Creates the fits files in cnn/fits_for_annotation
    The new fits files are to be annotated
    """

    # Converts input to properly align image
    x = pixel_width//2
    y = pixel_height//2
    file_dir = os.path.dirname(__file__)
    main_dir = os.path.dirname(os.path.dirname(file_dir))
    folder_path = os.path.join(main_dir, "cnn", "regions_to_fits_file")
    files = [file for file in glob.glob(os.path.join(folder_path, "*")) if file.endswith(".reg")]
         
    for file_name in files:
            regions = region_to_list_of_tuples(file_name)
            with fits.open(fits_path, memmap = True, mode = "denywrite") as imageHDU:
                    for region in regions:
                        X = int(region[0])
                        Y = int(region[1])
                        data = np.array(imageHDU[0].data[Y-y:Y+y, X-x:X+x], dtype=np.float32)
                    
                        #Creating the new files!
                        hdu = fits.PrimaryHDU(data)
                        hdul = fits.HDUList([hdu])
                        folder_path = os.path.join("cnn", "fits_for_annotation", f"box{region}.fits")
                        hdul.writeto(folder_path)

def region_to_list_of_tuples(region_path):
    """
    converts region info to list of tuples with float values in them
    region_n = (Center X, Center Y, Length X, Length Y, 0) <- Why 0? idunno ¯\\\\_(ツ)_/¯
    regions = [region_0, ..., region_n]
    """
    with open(region_path) as data:
        regions = data.readlines()[3:]
    regions = [_[4:-2] for _ in regions]
    regions = [tuple(float(s) for s in i.split(',')) for i in regions]
    return regions

def region_to_annotation(width, height, format="yolo"):
    """
    Folder with region files, to annotated files for CNN
    Width and height of the original fits file is required (original to region file, ie not 120000*12000)
    Should be the same as used for center_to_fits
    Outputs yolo formatted .txt files so far
    """
    #TODO fix other formats, probably easiest to use packages that convert yolo format

    #Processes all files in the region folder
    file_dir = os.path.dirname(__file__)
    main_dir = os.path.dirname(os.path.dirname(file_dir))
    folder_path = os.path.join(main_dir, "cnn", "regions_to_annotation")
    pattern = os.path.join(folder_path, "*")

    files = glob.glob(pattern)
    for file_name in files:
            regions = region_to_list_of_tuples(file_name)
            if format == "yolo":
                string = ""
                for region in regions:
                    string += f"0 {region[0]/width} {region[1]/height} {region[2]/width} {region[3]/height}\n"

                #removes last \n
                string = string[:-1]
                file_path = os.path.join("cnn", "annotations", os.path.basename(file_name) + ".txt")
                with open(file_path, "w") as file:
                    file.write(string)
            elif format == "voc":
                string = ""
                string = "<annotation>\n"
                #TODO maybe add actual folder name
                string += "\t<folder>Unknown</folder>\n"
                #TODO fix filename (not.reg, should be .fits)
                string += f"\t<filename>{os.path.basename(file_name)}</filename>\n"
                # TODO add relative path to image
                string += "\t<path>{Insert path here}</path>\n"
                string += "\t<source>\n"
                string += "\t\t<database>Unknown</database>\n"
                string += "\t</source>\n"
                string += "\t<size>\n"
                string += f"\t\t<width>{width}</width>\n"
                string += f"\t\t<height>{height}</height>\n"
                string += "\t\t<depth>1</depth>\n"
                string += "\t</size>\n"
                string += "\t<segmented>0</segmented>\n"

                for region in regions:
                    x_min = int(region[0] - region[2] / 2)
                    y_min = int(region[1] - region[3] / 2)
                    x_max = int(region[0] + region[2] / 2)
                    y_max = int(region[1] + region[3] / 2)
                    string += "<object>\n"
                    string += "\t<name>region</name>\n"
                    string += "\t<pose>Unspecified</pose>\n"
                    string += "\t<truncated>0</truncated>\n"
                    string += "\t<difficult>0</difficult>\n"
                    string += "\t<bndbox>\n"
                    string += f"\t\t<xmin>{x_min}</xmin>\n"
                    string += f"\t\t<ymin>{y_min}</ymin>\n"
                    string += f"\t\t<xmax>{x_max}</xmax>\n"
                    string += f"\t\t<ymax>{y_max}</ymax>\n"
                    string += "\t</bndbox>\n"
                    string += "</object>\n"
                string += "</annotation>"
                file_path = os.path.join("cnn", "annotations", os.path.basename(file_name) + ".xml")
                with open(file_path, "w") as file:
                    file.write(string)
            else:
                raise ValueError(f"Invalid format argument: {format}")

region_to_fits("test", r"E:\Kandidatarbete FITS\GitLab Repo\starseed\PROMISE-Q1-8micron-filled-v0_3.fits", 640, 640 )

def calculate_region_size(regions):
    """
    Calculates pixelsizes of all regions in the region file.
    Outputs a list with size for all regions, and the average, max, and min values
    For checking suitable input size for cnn :)
    [(avg, max, min), region_size_0, ..., region_size_n]
    """

    sizes = []
    for region in regions:
        size = region[2] * region[3]
        sizes.append(round(size,2))
    
    # not sure we want these on just the specific regionmap? Added these to calculate_all_region_sizes() also.
    avg_size = round( (sum(sizes) / len(sizes)) ,2)
    max_size = round(max(sizes),2)
    min_size = round(min(sizes),2)

    result = [(avg_size, max_size, min_size)] + sizes

    return result


def calculate_all_region_sizes():
    """
    Calculates the sizes of all regions in all files in the region file.
    Outputs a dictionary with the name of each region file as a key and a list
    containing the average, max, and min region sizes, as well as the size of
    each individual region, as the value. 
    
    NEW: Also shows distribution of sizes.
    """
    # not sure if we need specific region-file as a key? 
    file_dir = os.path.dirname(__file__)
    main_dir = os.path.dirname(os.path.dirname(file_dir))
    file_path = os.path.join(main_dir, "cnn", "regions", "*")
    region_files = glob.glob(file_path)
    results = {}
    all_sizes = []

    for region in region_files:
        regions = region_to_list_of_tuples(region)
        sizes = calculate_region_size(regions)
        all_sizes.extend(sizes[1:])
        results[os.path.basename(region)] = sizes

    avg_size = round( (sum(all_sizes) / len(all_sizes)) ,2)
    max_size = round(max(all_sizes),2)
    min_size = round(min(all_sizes),2)
    print("Average size of regions: ", avg_size , ", MaxSize region: ", max_size, ", MinSize region: ", min_size)

    #Showing the distibution of sizes. Just for information. Felt cute, might delete later  
    plt.hist(all_sizes, bins=3), plt.xlabel("Region Size"), plt.ylabel("Frequency")
    plt.title("Distribution of Region Sizes"), plt.show()

    return results


def split_region_fits_to_quarter(regionfile, fitsfile):
    """
    Splits the region fits file to a quarter, for more training/validation images
    Both the region data, and the fits file.
    should also split regions_to_annotation and fits_for_annotation
    _________     _________
    |       |     |   |   |
    |       | --> |---+---|
    |_______|     |___|___|

    """
    # Load the fits file containing region data
    with fits.open(regionfile) as hdul:
        data = hdul[0].data

    height, width = data.shape
    half_height = height // 2
    half_width = width // 2

    top_left = data[:half_height, :half_width]
    top_right = data[:half_height, half_width:]
    bottom_left = data[half_height:, :half_width]
    bottom_right = data[half_height:, half_width:]

    # Create four new fits files for the quarters
    new_filenames = ['{}_top_left.fits'.format(fitsfile),
                     '{}_top_right.fits'.format(fitsfile),
                     '{}_bottom_left.fits'.format(fitsfile),
                     '{}_bottom_right.fits'.format(fitsfile)]

    quarters = [top_left, top_right, bottom_left, bottom_right]

    for i, quarter in enumerate(quarters):
        hdu = fits.PrimaryHDU(quarter)
        hdul = fits.HDUList([hdu])
        hdul.writeto(new_filenames[i], overwrite=True)
    return 


def split_region_to_size(regionfile: str, xstart: int, xend: int, ystart: int, yend: int, save: bool = True) -> list:
    """
    Splits DS9 region file to specified size by cutting boxes and moving them

    Usage:
    ```python
    split_region_to_size("./data/regions/catalog_test2.reg", 800, 1600, 400, 2000)
    ```

    Could be adopted to take a list of regions as input, and return a list of regions as output or something for batch processing.
    """
    data = cd.read_regfile(regionfile)
    # data = [{ "box_x1": N, "box_x2": N, "box_y1": N, "box_y2": N }]

    if xstart > xend:
        raise ValueError("xstart must be smaller than xend")

    if ystart > yend:
        raise ValueError("ystart must be smaller than yend")

    output = []    

    for i, region in enumerate(data):
        if (region["box_x1"] > xend or region["box_x2"] < xstart) or (region["box_y1"] > yend or region["box_y2"] < ystart):
            continue
        else:
            if region["box_x1"] < xstart:
                region["box_x1"] = xstart
            if region["box_x2"] > xend:
                region["box_x2"] = xend
            if region["box_y1"] < ystart:
                region["box_y1"] = ystart
            if region["box_y2"] > yend:
                region["box_y2"] = yend

            output.append(region)

    new_path : str = regionfile.replace(".reg","_"+str(xstart)+"_"+str(xend)+"_"+str(ystart)+"_"+str(yend)+".reg")

    if save:
        cd.save_reg(output, new_path, "")
        
    return(output)
    
# region_to_fits("../../data/region_to_fits_files/1536 annotated region.reg", r"C:/Users/Emrik/OneDrive - Chalmers/Dokument/Programmering/Chalmers/starseed/data/fits/PROMISE-Q1-8micron-filled-v0_3.fits", 3072, 3072)