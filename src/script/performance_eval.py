def compare_dir_with_regfile(dir, regfile, TRUE_POSITIVE):
    """
    Compare a directory of region files with the catalog, the box shift is werid right now,
    and I think we should restructure and move some of this codes into different functions.
    Maybe... Depends.
    """
    import script.cloud_detection as cd
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import re
    catalog_path = regfile
    calculated = cd.read_regfile(catalog_path)
    
    found = []
    false_positives = 0
    annotated_boxes = []
    annotated_regions = []
    region_locations = []
    # läs in alla filer
    for file in os.listdir(dir):
        annotated_regions += [cd.stringbox_to_box(file)]
        annotated_boxes += [shift_to_box(box, annotated_regions[-1]) for box in cd.read_regfile(dir +'/'+ file)]
        match = re.search(r'box\((\d+\.\d+), (\d+\.\d+)', file)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            region_locations.append({
        "box_x1": round(x-400),
        "box_x2": round(x+400),
        "box_y1": round(y-400),
        "box_y2": round(y+400),
    })
    ann_amount = len(annotated_boxes)
    #return region_locations
    detections_in_annotated_region = 0
    for calculated_box in calculated:
        
        for k, annotated_region in enumerate(annotated_regions):
            if intersection_over_union(region_locations[k], calculated_box) != 0:
                detections_in_annotated_region += 1
            overlapping_boxes = []
            if(fraction_contained(annotated_region, calculated_box) > 0.5): # ändra.1till en gräns du vill ha
                for annotated_box in annotated_boxes:
                    overlap = boxes_overlap(calculated_box, annotated_box)
                    if overlap > TRUE_POSITIVE: # ändra if overlap till en gräns du vill ha
                        overlapping_boxes.append([annotated_box, overlap])
                if overlapping_boxes:
                    best_box = max(overlapping_boxes, key=lambda x: x[1])
                    found.append([annotated_box, best_box[1]])
                    annotated_boxes.remove(best_box[0])
                else:
                    # Vi är i ett annoterat område men inga annoterade överlappar med gränsen
                    # har denna annnoterade boxen redan hittats? Annars är det en false positive
                    # ifall den redan har hittats så kollar vi om den har en högre överlappning
                    multi_overlap = False
                    for i in range(len(found)):
                        overlap = boxes_overlap(calculated_box, found[i][0])
                        if overlap:
                            multi_overlap = True
                            if overlap > found[i][1]:
                                found[i][1] = overlap
                    if not multi_overlap:
                        false_positives += 1
    #return detections_in_annotated_region
    accuracy = (len(found)/ann_amount)*100
    if len(found) == 0:
        intersection_over_union_average = 0
    else:
        intersection_over_union_average = sum([x[1] for x in found])/len(found)

    fpr = 1 - ann_amount/false_positives
    fpr2 = false_positives/detections_in_annotated_region
    return accuracy, intersection_over_union_average, found, fpr, fpr2


def intersection_over_union(boxA, boxB):

    xA = max(boxA["box_x1"], boxB["box_x1"])
    yA = max(boxA["box_y1"], boxB["box_y1"])
    xB = min(boxA["box_x2"], boxB["box_x2"])
    yB = min(boxA["box_y2"], boxB["box_y2"])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both bounding boxes
    boxAArea = (boxA["box_x2"] - boxA["box_x1"] + 1) * (boxA["box_y2"] - boxA["box_y1"] + 1)
    boxBArea = (boxB["box_x2"] - boxB["box_x1"] + 1) * (boxB["box_y2"] - boxB["box_y1"] + 1)

    try:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    except ZeroDivisionError:
        iou = 0.0

    return iou

def fraction_contained(boxA, boxB):

    # Determine the intersection box
    xA = max(boxA["box_x1"], boxB["box_x1"])
    yA = max(boxA["box_y1"], boxB["box_y1"])
    xB = min(boxA["box_x2"], boxB["box_x2"])
    yB = min(boxA["box_y2"], boxB["box_y2"])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of smaller and larger bounding boxes
    boxAArea = (boxA["box_x2"] - boxA["box_x1"] + 1) * (boxA["box_y2"] - boxA["box_y1"] + 1)
    boxBArea = (boxB["box_x2"] - boxB["box_x1"] + 1) * (boxB["box_y2"] - boxB["box_y1"] + 1)

    # Compute the fraction of smaller box that is contained within the larger box
    if boxAArea < boxBArea:
        fraction = interArea / float(boxAArea)
    else:
        fraction = interArea / float(boxBArea)

    return fraction

def box_contained_in_box(box1, box2):
    if (
        box1["box_x1"] > box2["box_x1"]
        and box1["box_x2"] < box2["box_x2"]
        and box1["box_y1"] > box2["box_y1"]
        and box1["box_y2"] < box2["box_y2"]
    ):
        return True
    else:
        return False

def boxes_overlap(box1, box2):
    if (
        box1["box_x1"] < box2["box_x2"]
        and box1["box_x2"] > box2["box_x1"]
        and box1["box_y1"] < box2["box_y2"]
        and box1["box_y2"] > box2["box_y1"]
    ):
        return intersection_over_union(box1, box2)
    else:
        return 0.0
    
def shift_to_box(box1, box2):
    newbox = {}
    newbox["box_x1"] = box1["box_x1"] + box2["box_x1"]
    newbox["box_x2"] = box1["box_x2"] + box2["box_x1"]
    newbox["box_y1"] = box1["box_y1"] + box2["box_y1"]
    newbox["box_y2"] = box1["box_y2"] + box2["box_y1"]
    return newbox
