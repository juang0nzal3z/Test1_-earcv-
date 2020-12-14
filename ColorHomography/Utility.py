""" --------------------------------------------------------------------------------------------------
 This file has all the utility(helper) functions defined
 ----------------------------------------------------------------------------------------------------"""

import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

''' --------------------------------------------------------------------------------------------------
 Order box coordinates from top left, going clockwise, to bottom left
 Sort w.r.t x-coordinate first, retrieve top left & bottom right from it
 Then do similar with the y-coordinate
 ----------------------------------------------------------------------------------------------------'''
def order_points_new(pts):
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left = x_sorted[:2, :]
    right = x_sorted[2:, :]
    left = left[np.argsort(left[:, 1]), :]
    (top_left, bottom_left) = left
    right = right[np.argsort(right[:, 1]), :]
    (top_right, bottom_right) = right
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


''' --------------------------------------------------------------------------------------------------
 Calculate euclidean distance between two point coordinates
 ----------------------------------------------------------------------------------------------------'''
def calc_distance(x, y):
    dis = round(math.sqrt(math.pow(x[0]-y[0], 2) + math.pow(x[1]-y[1], 2)))
    return dis


''' --------------------------------------------------------------------------------------------------
 Calculate the correctly oriented coordinates of checkerboard
 ----------------------------------------------------------------------------------------------------'''
def get_dest_coord(boxs):
    fsc = []
    sc = []
    tc = []
    frc = []
    if calc_distance(boxs[0], boxs[3]) < calc_distance(boxs[0], boxs[1]):
        fsc.append(boxs[3][0])
        fsc.append(boxs[3][1])
        sc.append(boxs[3][0])
        sc.append(boxs[3][1] - calc_distance(boxs[0], boxs[1]))
        tc.append(boxs[3][0] + calc_distance(boxs[0], boxs[3]))
        tc.append(boxs[3][1] - calc_distance(boxs[0], boxs[1]))
        frc.append(boxs[3][0] + calc_distance(boxs[0], boxs[3]))
        frc.append(boxs[3][1])
    else:
        fsc.append(boxs[0][0])
        fsc.append(boxs[0][1])
        sc.append(boxs[0][0] + calc_distance(boxs[0], boxs[1]))
        sc.append(boxs[0][1])
        tc.append(boxs[0][0] + calc_distance(boxs[0], boxs[1]))
        tc.append(boxs[0][1] + calc_distance(boxs[0], boxs[3]))
        frc.append(boxs[0][0])
        frc.append(boxs[0][1] + calc_distance(boxs[0], boxs[3]))
    return np.array([fsc, sc, tc, frc])


''' --------------------------------------------------------------------------------------------------
 Calculate euclidean distance between two RGB colors
 ----------------------------------------------------------------------------------------------------'''
def dist_rgb(mat1, mat2):
    dis = round(math.sqrt(math.pow(mat1[0]-mat2[0], 2) + math.pow(mat1[1]-mat2[1], 2)) + math.pow(mat1[2]-mat2[2], 2))
    return dis


''' --------------------------------------------------------------------------------------------------
 Extract the checkerboard from the image after orienting it in the standard way
 ----------------------------------------------------------------------------------------------------'''
def clr_chk(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)  # Split into it channel constituents
    clr_ck = cv2.threshold(v, 65, 256, cv2.THRESH_BINARY)[1]  # Threshold
    clr_ck = cv2.morphologyEx(clr_ck, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=6)
    '''cv2.namedWindow('Reff', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Reff', 1000, 1000)
    cv2.imshow('Reff', clr_ck);
    cv2.waitKey(10000);'''
    cnts = cv2.findContours(clr_ck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE);
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    clr_ck = np.zeros_like(s)
    for c in cnts:
        area_tip = cv2.contourArea(c, oriented=0)
        if 2000 < area_tip < ((img.shape[0] * img.shape[1]) * 0.05):
            rects = cv2.minAreaRect(c)
            boxs = cv2.boxPoints(rects)
            boxs = np.array(boxs, dtype="int")
            width_i = int(rects[1][0])
            height_i = int(rects[1][1])
            if height_i > width_i:
                rat = round(width_i / height_i, 2)
            else:
                rat = round(height_i / width_i, 2)
            if rat > 0.96:
                cv2.drawContours(clr_ck, [c], -1, (255), -1)
    clr_ck = cv2.morphologyEx(clr_ck, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
                              iterations=10)
    _,cnts,_ = cv2.findContours(clr_ck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE);
    for c in cnts:
        rects = cv2.minAreaRect(c)
        boxs = cv2.boxPoints(rects)
        boxs = np.array(boxs, dtype="int")
        boxs = order_points_new(boxs)
        pts_dst = get_dest_coord(boxs)
        h, status = cv2.findHomography(boxs, pts_dst)
        img = cv2.warpPerspective(img, h, (img.shape[1], img.shape[0]))
    # repeat
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)  # Split into it channel constituents
    clr_ck = cv2.threshold(s, 65, 256, cv2.THRESH_BINARY)[1]  # Threshold
    clr_ck = cv2.morphologyEx(clr_ck, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=6)
    cnts = cv2.findContours(clr_ck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE);
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    clr_ck = np.zeros_like(s)
    for c in cnts:
        area_tip = cv2.contourArea(c, oriented=0)
        if 2000 < area_tip < ((img.shape[0] * img.shape[1]) * 0.05):
            rects = cv2.minAreaRect(c)
            width_i = int(rects[1][0])
            height_i = int(rects[1][1])
            if height_i > width_i:
                rat = round(width_i / height_i, 2)
            else:
                rat = round(height_i / width_i, 2)
            if rat > 0.94:
                cv2.drawContours(clr_ck, [c], -1, (255), -1)
    # cv2.namedWindow('Reff', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Reff', 1000, 1000)
    # cv2.imshow('Reff', mask); cv2.waitKey(5000); cv2.destroyAllWindows()
    clr_ck = cv2.morphologyEx(clr_ck, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
                              iterations=10)
    # take a binary image and run a connected component analysis
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(clr_ck, connectivity=8)
    # extracts sizes vector for each connected component
    sizes = stats[:, -1]
    # initiate counters
    max_label = 1
    max_size = sizes[1]
    # loop through and fine the largest connected component
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    # create an empty array and fill only with the largest the connected component
    clr_ck = np.zeros(clr_ck.shape, np.uint8)
    clr_ck[output == max_label] = 255
    # return a binary image with only the largest connected component
    cnts = cv2.findContours(clr_ck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE);
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    clr_ck = np.zeros_like(s)
    for c in cnts:
        rects = cv2.minAreaRect(c)
        boxs = cv2.boxPoints(rects)
        boxs = np.array(boxs, dtype="int")
        start_point = (boxs[1][0], boxs[1][1])
        end_point = (boxs[3][0] + 180, (boxs[3][1]))
        clr_ck = cv2.rectangle(clr_ck, start_point, end_point, (255), -1)
    clr_ck = cv2.dilate(clr_ck, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)), iterations=2)
    img_chk = img.copy()
    img_chk[clr_ck == 0] = 0
    return [img,img_chk]


""" -----------------------------------------------------------------------------------------------------
 Write the corrected images to the output directory
 -----------------------------------------------------------------------------------------------------"""
def write_corrected_images(outDir, targetFileName, targetImg, correctedImg):
    destin = outDir + "/" + targetFileName.split('./Images/')[1]
    im_a = cv2.hconcat([targetImg, correctedImg])
    cv2.imwrite(destin, im_a, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


""" -----------------------------------------------------------------------------------------------------
 Plot images before and after correction next to each other along with reference
 -----------------------------------------------------------------------------------------------------"""
def plot_images(src, tar, corrected):
    plt.subplot(2, 2, 1)
    plt.imshow(src)
    plt.title("Reference color checker")
    plt.subplot(2, 2, 3)
    plt.imshow(tar)
    plt.title("Before Correction")
    plt.subplot(2, 2, 4)
    plt.imshow(corrected)
    plt.title("After Correction")
    plt.show()