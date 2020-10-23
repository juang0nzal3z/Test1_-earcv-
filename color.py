#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
############################COMPUTER VISION FOR EAR ANALYSIS##############################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##################################BACKGROUND REMOVER######################################

####################################padding_version#######################################
#if your ears are not touching the edge then you dont need padding

###################################adaptive_version#######################################
#PARAMETER A
#close until the background have a convexity of < 0.4


#PARAMETER B
#open until all of the ears are disconnected due to silks connecting different ears

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
########################IMPORT PACKAGES AND FUNCTIONS#####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#!/usr/bin/python
from pyzbar.pyzbar import decode
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from statistics import stdev
from statistics import mean 
import sys, traceback, logging
import os
import re
import argparse
import string
from plantcv import plantcv as pcv
from plotnine import *


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
###################################FUNCTION ARGUMENTS#####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-r", "--reff", help="Input refferenceimage file.", required=True)
    parser.add_argument("-i", "--image", help="Input image file to be corrected", required=True)
    parser.add_argument("-c", "--color", default=False, action='store_true', help="Flag to do color checker analysis")
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=True)
    parser.add_argument("-D", "--debug", default=False, action='store_true', help="Turn on debug, prints intermediate images.")
    parser.add_argument("-s", "--save", default=True, action='store_false', help="Flag to stop saving proofs and ear ROIs")
    parser.add_argument("-p", "--proof", default=True, action='store_false', help="Flag to stop proof from showing final output")

    args = parser.parse_args()
    return args
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##################################### BUILD LOGGER #######################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def get_logger(logger_name):
	
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
	
	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.DEBUG) # better to have too much log than not enough
	logger.addHandler(console_handler)
	
	args = options()
	
	if args.outdir is not None:
		destin = "{}".format(args.outdir)
		if not os.path.exists(destin):
			try:
				os.mkdir(destin)
			except OSError as e:
				if e.errno != errno.EEXIST:
					raise
		LOG_FILE = ("{}ear_CV.log".format(args.outdir))
	else:
		LOG_FILE = "ear_CV.log"
		
	file_handler = logging.FileHandler(LOG_FILE)
	
	file_handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
	
	logger.addHandler(file_handler)
	# with this pattern, it's rarely necessary to propagate the error up to parent
	logger.propagate = False
	return logger
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##############################DEFINE HELPER FUNCTIONS#####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def clr_chk(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)	
	h,s,v = cv2.split(hsv)											#Split into it channel constituents

	#ret,_ = cv2.threshold(s, 0, 255, cv2.THRESH_OTSU) #cv2.imshow('img', thr); cv2.waitKey(5000); cv2.destroyAllWindows() 
	#print(ret)

	clr_ck = cv2.threshold(s,65,256, cv2.THRESH_BINARY)[1]		 		#Threshold

	clr_ck = cv2.morphologyEx(clr_ck, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations=2)

	#cv2.namedWindow('Reff', cv2.WINDOW_NORMAL)
	#cv2.resizeWindow('Reff', 1000, 1000)
	#cv2.imshow('Reff', bkgrnd); cv2.waitKey(5000); cv2.destroyAllWindows() 

	cnts = cv2.findContours(clr_ck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	clr_ck = np.zeros_like(s)
	for c in cnts:
		area_tip = cv2.contourArea(c)
		if 2000 < area_tip < ((img.shape[0]*img.shape[1])*0.05):
			rects = cv2.minAreaRect(c)
			width_i = int(rects[1][0])
			height_i = int(rects[1][1])
			if height_i > width_i:
				rat = round(width_i/height_i, 2)
			else:
				rat = round(height_i/width_i, 2)
			if rat > 0.94:
				#print(area_tip, rat)			
				cv2.drawContours(clr_ck, [c], -1, (255), -1)


	#cv2.namedWindow('Reff', cv2.WINDOW_NORMAL)
	#cv2.resizeWindow('Reff', 1000, 1000)
	#cv2.imshow('Reff', mask); cv2.waitKey(5000); cv2.destroyAllWindows() 

	clr_ck = cv2.morphologyEx(clr_ck, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25,25)), iterations=2)


	# take a binary image and run a connected component analysis
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(clr_ck, connectivity=8)
	# extracts sizes vector for each connected component
	sizes = stats[:, -1]
	#initiate counters
	max_label = 1
	max_size = sizes[1]
	#loop through and fine the largest connected component
	for i in range(2, nb_components):
		if sizes[i] > max_size:
			max_label = i
			max_size = sizes[i]
	#create an empty array and fill only with the largest the connected component
	clr_ck = np.zeros(clr_ck.shape, np.uint8)
	clr_ck[output == max_label] = 255
	#return a binary image with only the largest connected component
	
	#cv2.namedWindow('Reff', cv2.WINDOW_NORMAL)
	#cv2.resizeWindow('Reff', 1000, 1000)
	#cv2.imshow('Reff', cnct); cv2.waitKey(2000); cv2.destroyAllWindows() 

	cnts = cv2.findContours(clr_ck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	clr_ck = np.zeros_like(s)
	for c in cnts:
			rects = cv2.minAreaRect(c)
			boxs = cv2.boxPoints(rects)
			boxs = np.array(boxs, dtype="int")
			start_point = (boxs[1][0], boxs[1][1]) 
			end_point = (boxs[3][0]+180, (boxs[3][1]))
			clr_ck = cv2.rectangle(clr_ck, start_point, end_point, (255), -1)

	clr_ck = cv2.dilate(clr_ck, cv2.getStructuringElement(cv2.MORPH_RECT, (25,25)), iterations = 2)
	img_chk = img.copy()
	img_chk[clr_ck == 0] = 0

	#cv2.namedWindow('Reff', cv2.WINDOW_NORMAL)
	#cv2.resizeWindow('Reff', 1000, 1000)
	#cv2.imshow('Reff', img_chk); cv2.waitKey(5000); cv2.destroyAllWindows() 
	return img_chk

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
	w_min = min(im.shape[1] for im in im_list)
	im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
				for im in im_list]
	return cv2.vconcat(im_list_resize)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################DEFINE MAIN FUNCTION#####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Main workflow
def main():

	args = options()											# Get options
	log = get_logger("logger")									# Create logger
	log.info(args)												# Print expanded arguments


# Read image
#img, path, filename consider splitting into 
	reff = args.reff
	reff_i = cv2.imread(reff)										#Read file in
	root_ext = os.path.splitext(reff) 
	ext = root_ext[1]										#File  ID
	reff = root_ext[0]										#File  ID
	root = reff[:reff.rindex('/')+1]
	reff = reff[reff.rindex('/')+1:]
	print("Refference: {}".format(reff))

	reff_i_chk = clr_chk(reff_i)

	
# Read image
#img, path, filename consider splitting into 
	filename = args.image
	img=cv2.imread(filename)										#Read file in
	root_ext = os.path.splitext(filename) 
	ext = root_ext[1]										#File  ID
	filename = root_ext[0]										#File  ID
	root = filename[:filename.rindex('/')+1]
	filename = filename[filename.rindex('/')+1:]
	print("Processing: {}".format(filename))


	img_chk = clr_chk(img)



#Debug	
	if args.debug is True:

		cv2.namedWindow('Reff', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Reff', 1000, 1000)
		cv2.imshow('Reff', reff_i); cv2.waitKey(2000); cv2.destroyAllWindows() 

		cv2.namedWindow('Reff', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Reff', 1000, 1000)
		cv2.imshow('Reff', reff_i_chk); cv2.waitKey(2000); cv2.destroyAllWindows() 


		cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Original', 1000, 1000)
		cv2.imshow('Original', img); cv2.waitKey(2000); cv2.destroyAllWindows() 

		cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Original', 1000, 1000)
		cv2.imshow('Original', img_chk); cv2.waitKey(2000); cv2.destroyAllWindows() 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	if args.color is True:
		dataframe1, start, space = pcv.transform.find_color_card(rgb_img=img_chk, background='dark')
	# Use these outputs to create a labeled color card mask
		print(dataframe1)
		mask = pcv.transform.create_color_card_mask(rgb_img=img_chk, radius=5, start_coord=start, spacing=space, ncols=4, nrows=6)

		mask_i = cv2.threshold(mask,1,256, cv2.THRESH_BINARY)[1]		 		#Threshold
		proof = img.copy()
		proof[mask_i == 255] = (0,0,255)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#		
		reffdataframe1, start, space = pcv.transform.find_color_card(rgb_img=reff_i_chk, background='dark')
	# Use these outputs to create a labeled color card mask
		print(reffdataframe1)
		reffmask = pcv.transform.create_color_card_mask(rgb_img=reff_i_chk, radius=5, start_coord=start, spacing=space, ncols=4, nrows=6)
		mask_i = cv2.threshold(reffmask,1,256, cv2.THRESH_BINARY)[1]		 		#Threshold
		
		proof1 = reff_i.copy()
		
		proof1[mask_i == 255] = (0,0,255)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#		
		tm, sm, transformation_matrix, corrected_img = pcv.transform.correct_color(target_img=img, target_mask=mask, source_img=reff_i, source_mask=reffmask, output_directory="./")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		#cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
		#cv2.resizeWindow('Original', 1000, 1000)
		#cv2.imshow('Original', corrected_img); cv2.waitKey(2000); cv2.destroyAllWindows() 

		im_a = vconcat_resize_min([mask, reffmask])	#
		im_b = vconcat_resize_min([proof, proof1])#
		im_c = vconcat_resize_min([reff_i, img, corrected_img])	
		#im_a = vconcat_resize_min([mask, reffmask])
		#im_b = vconcat_resize_min([b_chnnl2, b_chnnl3, b_chnnl4])
		#im_all = cv2.vconcat([im_a, im_b])	
		
		cv2.namedWindow('HSV', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('HSV', 1000, 1000)
		cv2.imshow('HSV', im_a); cv2.waitKey(2000); cv2.destroyAllWindows() 

		cv2.namedWindow('HSV', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('HSV', 1000, 1000)
		cv2.imshow('HSV', im_b); cv2.waitKey(3000); cv2.destroyAllWindows() 
		
		cv2.namedWindow('HSV', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('HSV', 1000, 1000)
		cv2.imshow('HSV', im_c); cv2.waitKey(5000); cv2.destroyAllWindows() 


	#pcv.transform.save_matrix(matrix=tm, filename='target_matrix.npz')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
########################################OUTPUT############################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Rotate the image in case it is saved vertically  
	#if img.shape[0] > img.shape[1]:
	#	img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

	#img[0:700,:] = 0
	#img[:,900:img.shape[1]] = 0
	#img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE);

	#reff_i[0:700,:] = 0
	#reff_i[:,900:reff_i.shape[1]] = 0
	#reff_i = cv2.rotate(reff_i, cv2.ROTATE_90_COUNTERCLOCKWISE);


	#if args.proof is True:
		#cv2.namedWindow('Found Ears', cv2.WINDOW_NORMAL)
		#cv2.resizeWindow('Found Ears', 1000, 1000)
		#cv2.imshow('Found Ears', img); cv2.waitKey(3000); cv2.destroyAllWindows() 


#Save
	if args.save is True:
		#destin = "{}".format(args.outdir) + "01_Ear_PROOFs/"
		#os.mkdir(destin)
		#destin = "{}".format(args.outdir) + "/01_Ear_PROOFs/" + filename + "proof.jpeg"
		#print(destin)
		destin = "./" + filename + "proof.jpeg"
		cv2.imwrite(destin, img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
main()
