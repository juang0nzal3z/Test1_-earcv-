import numpy as np
import cv2
from plantcv import plantcv as pcv
import argparse
import LearnHomography
import ColorDiff
import csv
import os
import glob
import Utility



# Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Command line arguments for color correction")
    parser.add_argument("-r", "--reff", help="Input refferenceimage file.", required=True)
    parser.add_argument("-t", "--tar", help="Target image folder to be corrected", required=True)
    parser.add_argument("-o", "--outdir", default="", help="Output directory to store corrected images", required=True)
    parser.add_argument("-c", "--csv", default="color_correction(% improvement).csv", help="Path to csv file to write the results", required=False)
    parser.add_argument("-p", "--plot", default="False", help="Displays the image before and after the correction if value is True", required=False)
    args = parser.parse_args()
    return args
args = options()
csvName = args.csv
file_exists = os.path.isfile(csvName)
with open (csvName, 'w') as csvfile:
    headers = ['Image', 'Overall improvement', 'Square1', 'Square1', 'Square3', 'Square4', 'Square5', 'Square6',
               'Square7', 'Square8', 'Square9', 'Square10', 'Square11', 'Square12', 'Square13',
               'Square14', 'Square15', 'Square16', 'Square17', 'Square18', 'Square19', 'Square20',
               'Square21', 'Square22', 'Square23', 'Square24']
    writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
    writer.writeheader()

# SOURCE IMAGE
source = args.reff
srcImg = cv2.imread(source)
# src_chk is the extracted source color checker image
src_chk = srcImg

# Extract the color chip mask and RGB color matrix from source color checker image
dataframe1, start, space = pcv.transform.find_color_card(rgb_img=src_chk, background='dark')
src_mask = pcv.transform.create_color_card_mask(rgb_img=src_chk, radius=5, start_coord=start, spacing=space,
                                            ncols=4, nrows=6)
src_head, src_matrix = pcv.transform.get_color_matrix(src_chk, src_mask)
S = np.zeros((np.shape(src_matrix)[0], 3))

for r in range(0, np.ma.size(src_matrix, 0)):
    S[r][0] = src_matrix[r][1]
    S[r][1] = src_matrix[r][2]
    S[r][2] = src_matrix[r][3]
S_reshaped = np.reshape(S,(6, 4, 3))

# TARGET IMAGE
target_fold = args.tar + "/*"
files = glob.glob(target_fold)
for target in files:
    tarImg = cv2.imread(target)
    # Extract the color chip mask and RGB color matrix from target color checker image
    y, tar_chk = Utility.clr_chk(tarImg)
    dataframe1, start, space = pcv.transform.find_color_card(rgb_img=tar_chk, background='dark')
    tar_mask = pcv.transform.create_color_card_mask(rgb_img=tar_chk, radius=5, start_coord=start, spacing=space,
                                                ncols=4, nrows=6)
    tar_head, tar_matrix = pcv.transform.get_color_matrix(tar_chk, tar_mask)
    T = np.zeros((np.shape(tar_matrix)[0], 3))
    for r in range(0, np.ma.size(tar_matrix, 0)):
        T[r][0] = tar_matrix[r][1]
        T[r][1] = tar_matrix[r][2]
        T[r][2] = tar_matrix[r][3]
    T_reshaped = np.reshape(T, (6, 4, 3))
    if Utility.dist_rgb(S_reshaped[0][3], T_reshaped[0][3]) > Utility.dist_rgb(S_reshaped[0][3], T_reshaped[5][0]):
        T_reshaped = np.rot90(T_reshaped, axes=(0, 1))
        T_reshaped = np.rot90(T_reshaped, axes=(0, 1))
        T = np.reshape(T_reshaped, (24, 3))
    # Call functions from ColorHomo to generate and apply the color homography matrix
    homography = LearnHomography.generate_homography(S, T)
    corr = LearnHomography.apply_homo(tar_chk, homography, False)
    corrected = LearnHomography.apply_homo(tarImg, homography, True)

    Utility.write_corrected_images(args.outdir, target, tarImg, corrected)

    (avg_tar_error, avg_trans_error) = ColorDiff.calculate_color_diff(target, src_matrix, tar_matrix, corr)
    print(target, ": Before correction - ", avg_tar_error, ". After correction - ", avg_trans_error)
    corrected = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
    tarImg = cv2.cvtColor(tarImg, cv2.COLOR_RGB2BGR)
    srcToShow = cv2.cvtColor(srcImg, cv2.COLOR_RGB2BGR)

    if args.plot == "True":
        Utility.plot_images(srcToShow, tarImg, corrected)