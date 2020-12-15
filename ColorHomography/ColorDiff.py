""" --------------------------------------------------------------------------------------------------
 This file calculates the RMS difference between the mean RGB values of source and target color checker chip
 It also writes the overall % improvement in reduction of RMS difference as well as for each of the 24 color
  chip to a csv file
 ----------------------------------------------------------------------------------------------------"""

from plantcv import plantcv as pcv
import math
import numpy as np
import os
import csv


def calculate_color_diff(target_im, src_matrix, tar_matrix, transfer_chk):
    # RGB value based difference between source and target image
    dist = []
    dist_r = []
    dist_g = []
    dist_b = []
    avg_src_error = 0.0
    for r in range(0, np.ma.size(tar_matrix, 0)):
        for i in range(0, np.ma.size(src_matrix, 0)):
            if tar_matrix[r][0] == src_matrix[i][0]:
                r_mean = math.pow((tar_matrix[r][1] - src_matrix[i][1]), 2)
                g_mean = math.pow((tar_matrix[r][2] - src_matrix[i][2]), 2)
                b_mean = math.pow((tar_matrix[r][3] - src_matrix[i][3]), 2)
                dist_r.append(math.sqrt(r_mean))
                dist_g.append(math.sqrt(g_mean))
                dist_b.append(math.sqrt(b_mean))
                temp = math.sqrt((r_mean + g_mean + b_mean)/3)
                avg_src_error = avg_src_error + temp
                dist.append(temp)
    avg_src_error /= np.ma.size(tar_matrix, 0)

    # Corrected image
    # Extract the color chip mask and RGB color matrix from transfer color checker image
    dataframe1, start, space = pcv.transform.find_color_card(rgb_img=transfer_chk, background='dark')
    transfer_mask = pcv.transform.create_color_card_mask(rgb_img=transfer_chk, radius=5, start_coord=start,
                                                         spacing=space,
                                                         ncols=4, nrows=6)
    transfer_head, transfer_matrix = pcv.transform.get_color_matrix(transfer_chk, transfer_mask)

    # RGB value based difference between source and transfer image
    dist1 = []
    dist1_r = []
    dist1_g = []
    dist1_b = []
    csv_field = [target_im]
    avg_transfer_error = 0.0
    for r in range(0, np.ma.size(transfer_matrix, 0)):
        for i in range(0, np.ma.size(src_matrix, 0)):
            if transfer_matrix[r][0] == src_matrix[i][0]:
                r1_mean = math.pow((transfer_matrix[r][1] - src_matrix[i][1]), 2)
                g1_mean = math.pow((transfer_matrix[r][2] - src_matrix[i][2]), 2)
                b1_mean = math.pow((transfer_matrix[r][3] - src_matrix[i][3]), 2)
                dist1_r.append(math.sqrt(r1_mean))
                dist1_g.append(math.sqrt(g1_mean))
                dist1_b.append(math.sqrt(b1_mean))
                temp = math.sqrt((r1_mean + g1_mean + b1_mean)/3)
                avg_transfer_error = avg_transfer_error + temp
                dist1.append(temp)
                csv_field.append((dist[i]-temp)/float(dist[i])*100)
    avg_transfer_error /= np.ma.size(tar_matrix, 0)
    csv_field.insert(1, ((avg_src_error-avg_transfer_error)/float(avg_src_error))*100)
    csvName = "color_correction(% improvement).csv"
    file_exists = os.path.isfile(csvName)
    with open(csvName, 'a') as csvfile:
        headers = ['Image', 'Overall improvement', 'Square1', 'Square1', 'Square3', 'Square4', 'Square5', 'Square6',
                   'Square7', 'Square8', 'Square9', 'Square10', 'Square11', 'Square12', 'Square13',
                   'Square14', 'Square15', 'Square16', 'Square17', 'Square18', 'Square19', 'Square20',
                   'Square21', 'Square22', 'Square23', 'Square24']
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
        writer.writerow(
           {headers[i]: csv_field[i] for i in range(26)})
    return (avg_src_error, avg_transfer_error)
