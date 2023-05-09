def annotate_from_fits(X_LEFT, Y_TOP, SIZE, PATH, NAME):
    global img
    from astropy.io import fits
    import astropy
    import scipy
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from script.cloud_detection import detect_clouds
    from matplotlib.colors import ListedColormap
    from PIL import Image
    from script.cnnstuff import preprocess

    FITS_PATH = '../data/fits/PROMISE-Q1-8micron-filled-v0_3.fits'
    # Get FITS file
    imageHDU = fits.open(FITS_PATH, memmap=True, mode='denywrite')

    # Get display image as Numpy Array called data
    data = np.array(imageHDU[0].data[Y_TOP:Y_TOP+SIZE, \
                                     X_LEFT:X_LEFT+SIZE], dtype=np.float32)
    
    #data2 = detect_clouds(data)[4]

    x_fix, y = int(SIZE/0.75), SIZE
    shift_diff_fix = int((x_fix - y) / 2)

    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.autoscale(tight=True)

    plt.axis('off')

    my_cmap = ListedColormap(['#FFFFFF00', 'yellow'])

    prep_img = preprocess(data, 4, color_map='gray')
    plt.imshow(prep_img)
    plt.savefig(f'{PATH}/{NAME}.png', transparent=True)
    img = cv2.imread(f'{PATH}/{NAME}.png', cv2.IMREAD_UNCHANGED)

    """
    plt.imshow(data2, cmap=my_cmap, vmin=0, vmax=1)
    plt.savefig(f'{PATH}/detected.png', transparent=True)
    detected = cv2.imread(f'{PATH}/detected.png', cv2.IMREAD_UNCHANGED)

    layer1 = Image.open(f'{PATH}/detected.png')
    layer2 = Image.open(f'{PATH}/{NAME}.png')

    layer2.paste(layer1, (0, 0), layer1)
    layer2.save(f'{PATH}/combined.png')
    """

    img = cv2.imread(f'{PATH}/{NAME}.png', cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (x_fix, y))

    # When plotting an image to annotate matplotlib will add a border
    # around the image. This is not desired so we need to remove it.
    # I cannot seem to remove an x border on the left and right of the
    # image but i realized that the border is always 25% of the image size
    # so we adjust the image size and make it larger on the x axis and
    # later adjust the annotated box coordinates to compensate.

    reset = img.copy()
    out = img.copy()

    drawing = False
    ix,iy = -1,-1

    boxes = []

    # define mouse callback function to draw circle
    def draw_rectangle(event, x, y, flags, param):
        global ix, iy, drawing, img
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix = x
            iy = y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            new = np.zeros_like(img, np.uint8)
            cv2.rectangle(new, (ix, iy),(x, y),(255, 0, 255),cv2.FILLED)
            out = img.copy()
            alpha = 0.5
            mask = new.astype(bool)
            out[mask] = cv2.addWeighted(img, alpha, new, 1 - alpha, 0)[mask]
            cv2.imshow("Annotation", out)
            img = out.copy()
            boxes.append((abs(ix-shift_diff_fix), abs(iy) \
                          , abs(x-shift_diff_fix), abs(y)))

    #Create window
    cv2.namedWindow("Annotation")
    cv2.setMouseCallback("Annotation", draw_rectangle)

    # Keep annotating until escape key is pressed
    cv2.imshow("Annotation", out)
    while True:
        if cv2.waitKey(33) == ord('s'): break # Save and exit

        if cv2.waitKey(33) == 27: # Exit without saving
            boxes = []
            break

        if cv2.waitKey(33) == ord('r'): # Reset
            boxes = []
            img = reset.copy()
            cv2.imshow("Annotation", out)

        if cv2.waitKey(33) == ord('z'): # Undo
            boxes.pop()
            # TODO FIX SOME WAY TO DISPLAY PREV STATE
            
    cv2.destroyAllWindows()

    import os

    if os.path.exists(f"{PATH}/combined.png"):
        os.remove(f"{PATH}/combined.png")

    if os.path.exists(f"{PATH}/detected.png"):
        os.remove(f"{PATH}/detected.png")

    boxes = convert_to_region(boxes)

    return boxes

def convert_to_region(boxes):
    converted_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxvibe = \
            min(x1, x2) + abs(x2 - x1) / 2, min(y1, y2) + abs(y2 - y1) / 2 \
                            , abs(x2 - x1), abs(y2 - y1)
        converted_boxes.append(boxvibe)
    return converted_boxes

def regions_to_regfile(regions, name, path):
    file = open(f"{path}/{name}.reg", 'w')
    file.writelines(["cosmetic \n"])
    file.writelines(["three \n"])
    file.writelines(["lines \n"])
    for region in regions:
        file.writelines([f"box({region[0]}, {region[1]}, {region[2]}, {region[3]}, 0) \n"])
    file.close()

def region_to_string(region):
    return f"box({region[0]}, {region[1]}, {region[2]}, {region[3]}, 0)"