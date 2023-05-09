
def cloudMassCalc(catalog):
    import numpy as np
    import random
    
    ppp = 0.2 # parsec per pixel
    pc_to_cm = 9.521*1e36 # parsec^2 to cm^2

        # plot cloud nr 4357
    index = random.randint(0, len(catalog)-1) #4357 
    padding = 10
    row = catalog.iloc[index]

    nearby = catalog[(catalog["x_center"] > row["x_center"]-padding) & (catalog["x_center"] < row["x_center"]+padding) & (catalog["y_center"] > row["y_center"]-padding) & (catalog["y_center"] < row["y_center"]+padding)]
    nearby = nearby[nearby["x_center"] != row["x_center"]] # remove self

    rand_mask = np.load("../data/masks/"+row["mask_file"])

    data_box_Ioff = np.array(imageHDU[0].data[row["box_y1"]:row["box_y2"], row["box_x1"]: row["box_x2"]], dtype=np.float32) # type: ignore
    data_box_Iobs = np.array(imageHDU[0].data[row["box_y1"]:row["box_y2"], row["box_x1"]: row["box_x2"]], dtype=np.float32) # type: ignore

    rows, cols = rand_mask.shape
    for r in range(rows):
        for c in range(cols):
            if rand_mask[r,c] == True:
                data_box_Ioff[r,c] = None
            else:
                data_box_Iobs[r,c] = None

    flat_data_box_Iobs = np.sort(data_box_Iobs.flatten())
    I_fg = np.nanmean(flat_data_box_Iobs[:10])
    I_obs = np.nanmean(flat_data_box_Iobs)
    I_off = np.nanmean(data_box_Ioff.flatten())

    tau8 = -np.log((I_obs - I_fg)/(I_off - I_fg))
    kappa8 = 7.5 #cm^2 /g
    sigma8 = tau8/kappa8
    approx_size_of_cloud = row[2]

    

    return approx_size_of_cloud*ppp**2*pc_to_cm*sigma8/1000
