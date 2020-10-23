#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
######################### Computer Vision for Maize Ear Analysis  ########################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
######################### By: Juan M. Gonzalez, University of Florida  ###################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
###################################  IMPORT PACKAGES  ####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import sys, traceback, os, re, logging
import argparse
import string
from pyzbar.pyzbar import decode
import numpy as np
import cv2
import csv
from statistics import stdev
from statistics import mean 
from scipy.spatial import distance as dist
#from plantcv import plantcv as pcv

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##############################  PARSE FUNCTION ARGUMENTS  ################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def options():
	parser = argparse.ArgumentParser(description="Fulll pipeline for Maize ear analysis")

#Required input
	parser.add_argument("-i", "--image",  help="Path to input image file", required=True)

#Optional main
	parser.add_argument("-o", "--outdir", help="Provide directory to saves proofs, logfile, and output CSVs. Default: Will save in current directory if not provided.")
	parser.add_argument("-ns", "--no_save", default=False, action='store_true', help="Default saves proofs and output CSVs. Raise flag to stop saving.")
	parser.add_argument("-np", "--no_proof", default=False, action='store_true', help="Default prints proofs on screen. Raise flag to stop printing proofs.")
	parser.add_argument("-D", "--debug", default=False, action='store_true', help="Raise flag to print intermediate images throughout analysis. Useful for troubleshooting.")	

#QRcode options
	parser.add_argument("-qr", "--qrcode", default=False, action='store_true', help="Scan image for QRcode. Raise flag to start QRcode analysis.")	
	parser.add_argument("-r", "--rename", default=True, action='store_false', help="Default renames images with found QRcode. Raise flag to stop renaming images with found QRcode.")	
	parser.add_argument("-blk", "--block", default=True, action='store_false', help="Default breaks image into 10 overlapping blocks and looks for QRcode in each block. Raise flag to increase efficiency or to troubleshoot QRcode scanner.") 

#Color Checker options
	#parser.add_argument("-ppm", "--pixelspermetric", nargs=1, type=float, help="Calculate pixels per metric using either a color checker or the largest uniform color square. Provide width of refference.")

#Pixels Per Metric options
	parser.add_argument("-ppm", "--pixelspermetric", metavar=("[Refference Length]"), nargs=1, type=float, help="Calculate pixels per metric using either a color checker or the largest uniform color square. Provide refference length.")

#Find Ears options
	parser.add_argument("-bkgrnd", "--adv_background", default=False, action='store_true', help="adv. segmentation from background.") 
	parser.add_argument("-ear", "--ear_segmentation", metavar=("[Min Area]", "[Max Aspect Ratio]", "[Max Solidity]"), nargs=3, type=float, help="Ear segmentation filter. Default: Min Area--1 percent, Max Aspect Ratio: x < 0.6,  Max Solidity: 0.98. Flag with three arguments to customize ear filter.")
	parser.add_argument("-clnup", "--ear_cleanup", metavar=("[Max COV]", "[Max iterations]"), help="Ear clean-up module. Default: Max Convexity Coefficient of Variation threshold: 0.2, Max number of iterations: 10. Flag with two arguments to customize clean up module.", nargs=2, type=float, required=False)
	parser.add_argument("-rot", "--rotate", default=False, action='store_true', help="After segmenting ear use widths along the ear to determine the orientation of the ear and rotate accordingly.") 

#Analyze ear options
	parser.add_argument("-reff", "--reff_length", metavar=("[Refference Length]"), nargs=1, type=float, help="If pixel per metric is not used, then provide known refference to estimate relative measurements")
	parser.add_argument("-slk", "--silk_cleanup", metavar=("[Min delta COV change]", "[Max iterations]"), nargs=2, type=float, help="Silk decontamination module. Default: Min change in covexity coeffficient of variance: 0.04, Max number of iterations: 10. Flag with two arguments to customize silk clean up module")
	parser.add_argument("-t", "--tip", nargs=4, metavar=("[Tip percent]", "[Contrast]", "[Threshold]", "[Close]"), type=float, help="Tip segmentation module. Tip percent, Contrast, Threshold, Close. Flag with four arguments to customize tip segmentation module. Turn of module by providing '0' for all arguments")
	parser.add_argument("-b", "--bottom", nargs=4, metavar=("[Bottom percent]", "[Contrast]", "[Threshold]", "[Close]"), type=float, help="Bottom segmentation module. Bottom percent, Contrast, Threshold, Close. Flag with four arguments to customize tip segmentation module. Turn of module by providing '0' for all arguments")

	args = parser.parse_args()
	return args
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
############################  DEFINE HELPER FUNCTIONS  ###################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def start_points(size, split_size, overlap=10):
	points = [0]
	stride = int(split_size * (1-overlap))
	counter = 1
	while True:
		pt = stride * counter
		if pt + split_size >= size:
			points.append(size - split_size)
			break
		else:
			points.append(pt)
		counter += 1
	return points				# Defines matrix boundaries for blocking the image into overlapping sections
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def img_parse(fullpath):
	fullpath = fullpath
	root_ext = os.path.splitext(fullpath) 
	ext = root_ext[1]											
	filename = root_ext[0]										#File  ID
	try:
		root = filename[:filename.rindex('/')+1]
	except:
		root = "./"
	try:
		filename = filename[filename.rindex('/')+1:]
	except:
		filename = filename
	return fullpath, root, filename, ext										# Parses input path into root, filename, and extension
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def max_cnct(binary):
# take a binary image and run a connected component analysis
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
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
	cnct = np.zeros(binary.shape, np.uint8)
	cnct[output == max_label] = 255
#return a binary image with only the largest connected component
	return cnct										# Returns largest connected component
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")										# Orders connected components from left ot right
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def ranges(N, nb):
	step = N / nb
	return ["{},{}".format(round(step*i), round(step*(i+1))) for i in range(nb)]											# Cut image into N slices
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def get_logger(logger_name):
	args = options()
	
	if args.outdir is not None:
		out = args.outdir
	else:
		out = "./"

	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.DEBUG) # better to have too much log than not enough
	logger.addHandler(console_handler)
	
	destin = "{}".format(out)
	if not os.path.exists(destin):
		try:
			os.mkdir(destin)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise
		LOG_FILE = ("{}ear_CV.log".format(out))
	else:
		LOG_FILE = ("{}ear_CV.log".format(out))
		
	file_handler = logging.FileHandler(LOG_FILE)
	
	file_handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
	
	logger.addHandler(file_handler)
	# with this pattern, it's rarely necessary to propagate the error up to parent
	logger.propagate = False
	return logger									# Defines data logger 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def cnctfill(binary):
	# take a binary image and run a connected component analysis
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
	# extracts sizes vector for each connected component
	sizes = stats[:, -1]
	#initiate counters
	max_label = 1
	if len(sizes) > 1:
		max_size = sizes[1]
	#loop through and fine the largest connected component
		for i in range(2, nb_components):
			if sizes[i] > max_size:
				max_label = i
				max_size = sizes[i]
	#create an empty array and fill only with the largest the connected component
		cnct = np.zeros(binary.shape, np.uint8)
		cnct[output == max_label] = 255
	#take that connected component and invert it
		nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(cnct), connectivity=8)
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
		filld = np.zeros(binary.shape, np.uint8)
		filld[output == max_label] = 255
		filld = cv2.bitwise_not(filld)
	else:
		filld = binary
	#return a binary image with only the largest connected component, filled
	return filld											# Fill in largest connected component
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
 
 	if brightness != 0:
 		if brightness > 0:
 			shadow = brightness
 			highlight = 255
 		else:
 			shadow = 0
 			highlight = 255 + brightness
 		alpha_b = (highlight - shadow)/255
 		gamma_b = shadow
 		
 		buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
 	else:
 		buf = input_img.copy()
 	
 	if contrast != 0:
 		f = 131*(contrast + 127)/(127*(131-contrast))
 		alpha_c = f
 		gamma_c = 127*(1-f)
 		
 		buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
 		
 	return buf # Change contrast of image
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
###############################  DEFINE MAIN FUNCTION  ###################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def main():
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Preprocessing
	args = options()											# Get options
	log = get_logger("logger")									# Create logger
	log.info(args)												# Print expanded arguments
	fullpath, root, filename, ext = img_parse(args.image)		# Parse provided path
	if args.outdir is not None:									# If out dir is provided
		out = args.outdir
	else:
		out = "./"
	img=cv2.imread(fullpath)									# Read file in
	log.info("[START]--{}--Starting analysis pipeline..".format(filename)) # Log


##YOU HAVE TO CHANGE THIS IF YOU ARE RUNINING THE THING ON THE WHOLE IMAGE WITH MULTIUPLE EARS
	if img.shape[1] > img.shape[0]:								# Rotate the image in case it is saved vertically  
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

	proof = img.copy()
	ears = img.copy()

	if args.debug is True:										# Debug
		cv2.namedWindow('[DEBUG] Original', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('[DEBUG] Original', 1000, 1000)
		cv2.imshow('[DEBUG] Original', img); cv2.waitKey(3000); cv2.destroyAllWindows() 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
####################################  QR CODE MODULE  ####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Set default variables
	if args.qrcode is True:
		id = []
		QRcodeData = None
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Break into subimages for QRCODE Scan
		if args.block is True:
			log.info("[QR]--{}--Divide image into overlapping subsections".format(filename))
			proof_h, proof_w, _ = proof.shape
			split_width = int(proof.shape[0]/2)
			split_height = int(proof.shape[1]/2)
			X_points = start_points(proof_w, split_width, 0.5)
			Y_points = start_points(proof_h, split_height, 0.5)
			count = 0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Scan each subimages for QRCODE
			for i in Y_points:
				for j in X_points:
					split = proof[i:i+split_height, j:j+split_width]
					count += 1
					mask = cv2.inRange(split,(0,0,0),(200,200,200))
					thresholded = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
					inverted = 255-thresholded # black-in-white

					if args.debug is True: 							# Debug
						cv2.namedWindow('[QR][DEBUG] Searching...QR', cv2.WINDOW_NORMAL)
						cv2.resizeWindow('[QR][DEBUG] Searching...QR', 1000, 1000)
						cv2.imshow('[QR][DEBUG] Searching...QR', inverted); cv2.waitKey(1500); cv2.destroyAllWindows()

					log.info("[QR]--{}--Searching for QRCODE: {}th iteration".format(filename, count))
					id = decode(inverted)
					if id != []:				
						for QRcode in decode(inverted):
							id= QRcode.data.decode()
							(x, y, w, h) = QRcode.rect
							cv2.rectangle(split, (x, y), (x + w, y + h), (0, 0, 255), 15)
							cv2.rectangle(ears, (x-40, y-40), (x + w + 320, y + h + 40), (0, 0, 0), -1)
						# the QRcode data is a bytes object so if we want to draw it on
							QRcodeData = QRcode.data.decode("utf-8")
							QRcodeType = QRcode.type
						# draw the QRcode data and QRcode type on the image
							text = "Found {}: {}".format(QRcodeType, QRcodeData)
							cv2.putText(split, text, (int(50), int(800)), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 255), 8)
						# print the QRcode type and data to the terminal
							log.info("[QR]--{}--Found {}: {} on the {}th iteration".format(filename, QRcodeType, QRcodeData, count))					

							if args.debug is True: 							# Debug
								cv2.namedWindow('[QR][DEBUG] QR Subimage', cv2.WINDOW_NORMAL)
								cv2.resizeWindow('[QR][DEBUG] QR Subimage', 1000, 1000)
								cv2.imshow('[QR][DEBUG] QR Subimage', split); cv2.waitKey(3000); cv2.destroyAllWindows()						

						break
					if QRcodeData is not None:
						proof = split
				else:		
					continue
				break
		else:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Scan entire image for QRcode
			log.info("[QR]--{}--Scanning entire image for QRcode...".format(filename))
			mask = cv2.inRange(proof,(0,0,0),(200,200,200))
			thresholded = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
			inverted = 255-thresholded # black-in-white

			if args.debug is True:								# Debug
				cv2.namedWindow('[DEBUG] QR', cv2.WINDOW_NORMAL)
				cv2.resizeWindow('[DEBUG] QR', 1000, 1000)
				cv2.imshow('[DEBUG] QR', inverted); cv2.waitKey(3000); cv2.destroyAllWindows()						
		
			id = decode(inverted)
			if id != []:
				for QRcode in decode(inverted):
					id= QRcode.data.decode()
					(x, y, w, h) = QRcode.rect
					cv2.rectangle(proof, (x, y), (x + w, y + h), (0, 0, 255), 20)
					# the QRcode data is a bytes object so if we want to draw it on
					QRcodeData = QRcode.data.decode("utf-8")
					QRcodeType = QRcode.type
					# draw the QRcode data and QRcode type on the image
					text = "Found {}: {}".format(QRcodeType, QRcodeData)
					cv2.putText(proof, text, (int(proof.shape[0]/3), int(proof.shape[1]/3)), cv2.FONT_HERSHEY_SIMPLEX,7, (0, 0, 255), 15)
				# print the QRcode type and data to the terminal
					log.info("[QR]--{}--Found {}: {}".format(filename, QRcodeType, QRcodeData))							
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~QRCODE output
		if QRcodeData is None:										# Print error if no QRcode found
			log.warning("[QR]--{}--Error: QRcode not found".format(filename))
		else:
			if args.no_proof is False:								# Print proof with QR code
				cv2.namedWindow('[QR][PROOF] Found QRcode', cv2.WINDOW_NORMAL)
				cv2.resizeWindow('[QR][PROOF] Found QRcode', 1000, 1000)
				cv2.imshow('[QR][PROOF] Found QRcode', proof); cv2.waitKey(3000); cv2.destroyAllWindows()

			if args.rename is True:									# Rename image with QR code
				os.rename(args.image, root + QRcodeData + ext)
				filename = QRcodeData
				log.info("[QR]--{}--Renamed with QRCODE info: {}".format(filename, filename, QRcodeData))

			if args.no_save is False:								
				csvname = out + 'QRcodes' +'.csv'			# Create CSV and store barcode info
				file_exists = os.path.isfile(csvname)
				with open (csvname, 'a') as csvfile:
					headers = ['Filename', 'QR Code']
					writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
					if not file_exists:
						writer.writeheader()
					writer.writerow({'Filename': filename, 'QR Code': QRcodeData})
				log.info("[QR]--{}--Saved filename and QRcode info to: {}QRcodes.csv".format(filename, out))
			
				destin = "{}".format(out) + "01_Proofs/" 	# Create proof folder and save proof
				if not os.path.exists(destin):
					try:
						os.mkdir(destin)
					except OSError as e:
						if e.errno != errno.EEXIST:
							raise
				destin = "{}".format(out) + "01_Proofs/" + QRcodeData + "_proof.jpeg"
				log.info("[QR]--{}--Proof saved to: {}".format(filename, destin))
				cv2.imwrite(destin, proof, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
	else:
		log.info("[QR]--{}--QR module turned off".format(filename))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##############################  COLOR CORRECTION MODULE  #################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#NEED TO MAKE THIS PERFECT....HAVE DRAFT CODE UNDER clrchkr.py
	#img_chk = clr_chk(img) 
	#if img_chk is not None:
	#	img[img_chk != 0] = 0		
	#	if args.debug is True:				
	#		cv2.namedWindow('FOUND: Color Checker', cv2.WINDOW_NORMAL)
	#		cv2.resizeWindow('FOUND: Color Checker', 1000, 1000)
	#		cv2.imshow('FOUND: Color Checker', img_chk); cv2.waitKey(5000); cv2.destroyAllWindows() 
#	if args.color_checker is not None:
#	else:
#		log.info("[CLR] Color Checker module turned off")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##############################  PIXELS PER METRIC MODULE  ################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Calculating pixels per metric
	if args.pixelspermetric is not None:
		log.info("[PPM]--{}--Calculating pixels per metric".format(filename))
		PixelsPerMetric = None
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Find any objects with high saturation
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		_,_,v = cv2.split(hsv)
		_, sqr = cv2.threshold(v, 0, 255, cv2.THRESH_OTSU)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Find any roughly square objects
		cnts = cv2.findContours(sqr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
		mask = np.zeros_like(sqr)
		for c in cnts:
			area_sqr = cv2.contourArea(c)
			if area_sqr > 3000:
				rects = cv2.minAreaRect(c)
				width_i = int(rects[1][0])
				height_i = int(rects[1][1])
				if height_i > width_i:
					rat = round(width_i/height_i, 2)
				else:
					rat = round(height_i/width_i, 2)
				if 0.95 <  rat < 1.1: 
					cv2.drawContours(mask, [c], -1, (255), -1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Find and measure width largest square
		if cv2.countNonZero(mask) != 0:
			mask = max_cnct(mask)
			cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
			for cs in cnts:			
				proof =cv2.drawContours(proof, [cs], -1, (53, 57, 250), 5)
				areas = cv2.contourArea(cs)
				rects = cv2.minAreaRect(cs)
				boxs = cv2.boxPoints(rects)
				boxs = np.array(boxs, dtype="int")			
				boxs1 = order_points(boxs)
				proof =cv2.drawContours(proof, [boxs.astype(int)], -1, (0, 255, 255), 10)
				(tls, trs, brs, bls) = boxs
				(tltrXs, tltrYs) = ((tls[0] + trs[0]) * 0.5, (tls[1] + trs[1]) * 0.5)
				(blbrXs, blbrYs) = ((bls[0] + brs[0]) * 0.5, (bls[1] + brs[1]) * 0.5)
				(tlblXs, tlblYs) = ((tls[0] + bls[0]) * 0.5, (tls[1] + bls[1]) * 0.5)
				(trbrXs, trbrYs) = ((trs[0] + brs[0]) * 0.5, (trs[1] + brs[1]) * 0.5)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~compute the Euclidean distance between the midpoints
				dBs = dist.euclidean((tlblXs, tlblYs), (trbrXs, trbrYs)) #pixel width
				PixelsPerMetric = dBs / args.pixelspermetric[0]
				cv2.putText(proof, "{:.1f} Pixels per Metric".format(PixelsPerMetric),
						(int(trbrXs), int(trbrYs - 180)), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 15)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~draw midpoints and lines on proof
				cv2.line(proof, (int(tlblXs), int(tlblYs)), (int(trbrXs), int(trbrYs)), (255, 0, 255), 20) #width
				cv2.circle(proof, (int(tlblXs), int(tlblYs)), 23, (255, 0, 255), -1) #left midpoint
				cv2.circle(proof, (int(trbrXs), int(trbrYs)), 23, (255, 0, 255), -1) #right midpoint

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PPM module Output
		if PixelsPerMetric is not None:
			log.info("[PPM]--{}--Found {} pixels per metric".format(filename, PixelsPerMetric))
			
			if args.no_proof is False:
				cv2.namedWindow('Pixels Per Metric: FOUND', cv2.WINDOW_NORMAL)
				cv2.resizeWindow('Pixels Per Metric: FOUND', 1000, 1000)
				cv2.imshow('Pixels Per Metric: FOUND', proof); cv2.waitKey(2000); cv2.destroyAllWindows()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~If size reference found then remove it
			ears[mask != 0] = 0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Save proof and pixels per metric csv			
			if args.no_save is False:		
				csvname = out + 'pixelspermetric' +'.csv'
				file_exists = os.path.isfile(csvname)
				with open (csvname, 'a') as csvfile:
					headers = ['Filename', 'Pixels Per Metric']
					writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
					if not file_exists:
						writer.writeheader()  # file doesn't exist yet, write a header	
					writer.writerow({'Filename': filename, 'Pixels Per Metric': PixelsPerMetric})
				log.info("[PPM]--{}--Saved pixels per metric to: {}pixelspermetric.csv".format(filename, out))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Save proofs
				destin = "{}".format(out) + "01_Proofs/"
				if not os.path.exists(destin):
					try:
						os.mkdir(destin)
					except OSError as e:
						if e.errno != errno.EEXIST:
							raise
				destin = "{}".format(out) + "01_Proofs/" + filename + "_proof.jpeg"
				log.info("[PPM]--{}--Proof saved to: {}".format(filename, destin))
				cv2.imwrite(destin, proof, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
		else:
			log.warning("[PPM]--{}--No size refference found for pixel per metric calculation".format(filename))
	else:
		log.info("[PPM]--{}--Pixels per Metric module turned off".format(filename))
		PixelsPerMetric = None
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##################################  FIND EARS MODULE  ###################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	log.info("[EARS]--{}--Looking for ears...".format(filename))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Set default variables
	if args.ear_segmentation is not None:
		log.info("[EARS]--{}--Ear Segmentation with cutom settings: Area: {}% ".format(filename, args.ear_segmentation[0]) + " 0.19 < Aspect Ratio < {}".format(format(args.ear_segmentation[1])) + " Solidity < {}".format(args.ear_segmentation[2]))
		area_var = ((ears.shape[0]*ears.shape[1])*((args.ear_segmentation[0])/100))
		rat_var = args.ear_segmentation[1]
		solidity_var = args.ear_segmentation[2]
	else:
		area_var = ((ears.shape[0]*ears.shape[1])*0.010)
		rat_var = 0.6
		solidity_var = 0.983
		log.info("[EARS]--{}--Segmenting ear with default settings: Area {}% ".format(filename, float(area_var/(ears.shape[0]*ears.shape[1])*100)) + " 0.19 < Aspect Ratio < {}".format(format(rat_var)) + " Solidity < {}".format(solidity_var))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Use K means to threshold ears in lab channel
	if args.adv_background is True:
		lab = cv2.cvtColor(ears, cv2.COLOR_BGR2LAB)
		vectorized = lab.reshape((-1,3))
		vectorized = np.float32(vectorized)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		K = 2
		attempts = 3
		ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
		center = np.uint8(center)
		res = center[label.flatten()]
		img_sgmnt = res.reshape((img.shape))
		_,_,gray = cv2.split(img_sgmnt)		
		_, bkgrnd = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
	else:
		#Threshold in the red channel
		_,_,r = cv2.split(ears)											#Get the red channel
		bkgrnd = cv2.threshold(r,50,256, cv2.THRESH_BINARY)[1]		 		#Threshold
	if args.debug is True:
		cv2.namedWindow('[EARS][DEBUG] Segmentation before Filter', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('[EARS][DEBUG] Segmentation before Filter', 1000, 1000)
		cv2.imshow('[EARS][DEBUG] Segmentation before Filter', bkgrnd); cv2.waitKey(2000); cv2.destroyAllWindows()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~Filter connected components with area, aspect:ratio, and solidity	
	cnts = cv2.findContours(bkgrnd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	mask = np.zeros_like(bkgrnd)
	i = 1
	for c in cnts:
		area_tip = cv2.contourArea(c)
		if area_tip > area_var:
			perimeters = cv2.arcLength(c,True); hulls = cv2.convexHull(c); hull_areas = cv2.contourArea(hulls); hullperimeters = cv2.arcLength(hulls,True)
			solidity_tip = float(area_tip)/hull_areas
			rects = cv2.minAreaRect(c)
			width_i = int(rects[1][0])
			height_i = int(rects[1][1])
			if height_i > width_i:
				rat = round(width_i/height_i, 2)
			else:
				rat = round(height_i/width_i, 2)
			#log.warning("{},{}".format(rat, solidity_tip))
			if 0.19 < rat < rat_var and solidity_tip < solidity_var: 
				log.info("[EARS]--{}--Ear #{}: Min Area: {}% Aspect Ratio: {} Solidity score: {}".format(filename, i, round((area_tip/(img.shape[0]*img.shape[1]))*100, 3), rat, round(solidity_tip, 3)))
				cv2.drawContours(mask, [c], -1, (255), -1)
				i = i+1

	if args.debug is True:												# Debug
		cv2.namedWindow('[EARS][DEBUG] Segmentation after Filter', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('[EARS][DEBUG] Segmentation after Filter', 1000, 1000)
		cv2.imshow('[EARS][DEBUG] Segmentation after Filter', mask); cv2.waitKey(2000); cv2.destroyAllWindows()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~Silk Contamination
	cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	areas = [cv2.contourArea(c) for c in cnts]
	if len(areas) > 1:
		cov = (stdev(areas)/mean(areas))
		log.info("[CLNUP]--{}--Coefficent of Variance: {}".format(filename, cov))
		
		if cov > 0.35:
			log.warning("[CLNUP]--{}--COV above 0.2 has triggered default Ear clean-up module".format(filename))
			i = 1
			b = 1
			cov_var = 0.35
			i_var = 10
			
		elif args.ear_cleanup is not None:
			log.info("[CLNUP]--{}--Ear clean-up module with cutom setings".format(filename))
			cov_var = args.ear_cleanup[0]	
			i_var = args.ear_cleanup[1]	
		
		if cov > 0.35 or args.ear_cleanup is not None:
			while cov > cov_var  and i <= i_var:
				log.info("[CLNUP]--{}--Ear clean-up module: Iterate up to {} times or until COV < {} Current COV: {} and iteration {}".format(filename, i_var, cov_var, round(cov, 3), i))
				mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=i)
				cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
				mask = np.zeros_like(mask)
				for c in cnts:
					area_tip = cv2.contourArea(c)
					perimeters = cv2.arcLength(c,True); hulls = cv2.convexHull(c); hull_areas = cv2.contourArea(hulls); hullperimeters = cv2.arcLength(hulls,True)
					solidity_tip = float(area_tip)/hull_areas
					rects = cv2.minAreaRect(c)
					width_i = int(rects[1][0])
					height_i = int(rects[1][1])
					if height_i > width_i:
						rat = round(width_i/height_i, 2)
					else:
						rat = round(height_i/width_i, 2)
					if area_tip > area_var and 0.17 < rat < rat_var  and solidity_tip < solidity_var: 
						log.info("[CLNUP]--{}--Ear #{}: Min Area--{}% Aspect Ratio--{} Solidity score--{}".format(filename, b, round((area_tip/(img.shape[0]*img.shape[1]))*100, 3), rat, round(solidity_tip, 3)))
						b = 1 + b
						cv2.drawContours(mask, [c], -1, (255), -1)

				cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
				areas = [cv2.contourArea(c) for c in cnts]
				cov = (stdev(areas)/mean(areas))
				log.info("[CLNUP]--{}--Coefficent of Variance: {}".format(filename, cov))
				i = i+1    #  update counter
				b = 1

				if args.debug is True:
					cv2.namedWindow('Silk Contamination', cv2.WINDOW_NORMAL)
					cv2.resizeWindow('Silk Contamination', 1000, 1000)
					cv2.imshow('Silk Contamination', mask); cv2.waitKey(2000); cv2.destroyAllWindows() 
	else:
		log.warning("[CLNUP]--{}--Cannot calculate Coefficent of Variance on single ear".format(filename))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
########################################Sort Ears#########################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	ears[mask != 255] = 0 
	cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	areas = [cv2.contourArea(c) for c in cnts]
	if len(areas) >1:
		cov = (stdev(areas)/mean(areas))
		log.info("[CLNUP]--{}--Final COV--{}".format(filename, cov))
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
#Sort left to right
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][0], reverse= False))
#Count the number of ears and number them on proof
	i = 0
	Temp = []
	for c in cnts:
		i = i+1
#Find centroid
		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
#Create ROI and find tip
		rects = cv2.minAreaRect(c)
		boxs = cv2.boxPoints(rects)
		boxs = np.array(boxs, dtype="int")
		width_i = int(rects[1][0])
		height_i = int(rects[1][1])
		src_pts_i = boxs.astype("float32")
		dst_pts_i = np.array([[0, height_i-1],[0, 0],[width_i-1, 0],[width_i-1, height_i-1]], dtype="float32")
		M_i = cv2.getPerspectiveTransform(src_pts_i, dst_pts_i)
		ear = cv2.warpPerspective(ears, M_i, (width_i, height_i))
		if ear.shape[1] > ear.shape[0]:
			ear = cv2.rotate(ear, cv2.ROTATE_90_COUNTERCLOCKWISE) 				#This rotates the image in case it is saved vertically
		ear = cv2.copyMakeBorder(ear, 30, 30, 30, 30, cv2.BORDER_CONSTANT)
#Draw the countour number on the image
		Temp.append(ear)
		cv2.drawContours(proof, [c], -1, (134,22,245), -1)
		cv2.putText(proof, "#{}".format(i), (cX - 80, cY), cv2.FONT_HERSHEY_SIMPLEX,4.0, (255, 255, 0), 10)
	
	cv2.putText(proof, "Found {} Ear(s)".format(i), (int((img.shape[0]/1.5)), img.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (200, 255, 255), 17)
	log.info("[EARS]--{}--Found {} Ear(s)".format(filename, i))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Save proof and pixels per metric csv			
	if args.no_save is False:		
		destin = "{}".format(out) + "01_Found_Proofs/"
		if not os.path.exists(destin):
			try:
				os.mkdir(destin)
			except OSError as e:
				if e.errno != errno.EEXIST:
					raise
		destin = "{}".format(out) + "01_Found_Proofs/" + filename + "_proof.jpeg"
		log.info("[EARS]--{}--Proof saved to: {}".format(filename, destin))
		cv2.imwrite(destin, proof, [int(cv2.IMWRITE_JPEG_QUALITY), 20])

	

	if args.no_proof is False:
		cv2.namedWindow('Found Ears', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Found Ears', 1000, 1000)
		cv2.imshow('Found Ears', proof); cv2.waitKey(3000); cv2.destroyAllWindows() 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
############################## OPERATIONS ON SINGLE EAR ##################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	if args.reff_length is not None:
		PixelsPerMetric = args.reff_length[0]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
################################## CLEAN UP EAR #########################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	n = 1 #Counter
	for r in range(i):
		ear = Temp[r]
		ymax = ear.shape[0]										#Make copy of original image
		_,_,r = cv2.split(ear)											#Split into it channel constituents
		_,r = cv2.threshold(r, 0, 255, cv2.THRESH_OTSU)
		r = cnctfill(r)
		ear[r == 0] = 0
		lab = cv2.cvtColor(ear, cv2.COLOR_BGR2LAB)
		lab[r == 0] = 0
		_,_,b_chnnl = cv2.split(lab)										#Split into it channel constituents
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Calculate convexity
		cntss = cv2.findContours(b_chnnl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		cntss = cntss[0] if len(cntss) == 2 else cntss[1]
		cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
		for cs in cntss:
			perimeters = cv2.arcLength(cs,True); hulls = cv2.convexHull(cs); hull_areas = cv2.contourArea(hulls); hullperimeters = cv2.arcLength(hulls,True)
			Convexity = hullperimeters/perimeters
			log.info("[SILK]--{}--Convexity: {}".format(filename, round(Convexity, 3)))

		if Convexity < 0.87:
			i = 1
			conv_var = 0.04
			i_var = 10	
			log.warning("[SILK]--{}--Convexity under 0.87 has triggered default ear clean-up module".format(filename))
	
		elif args.silk_cleanup is not None:
			log.info("[SILK]--{}--Convexity clean up module with custom settings")
			conv_var = args.silk_cleanup[0]	
			i_var = args.silk_cleanup[1]	
		
		if 	Convexity < 0.87 or args.silk_cleanup is not None:
			delta_conv = 0.001
			log.info("[SILK]--{}--Min delta convexity: {}, Max interations: {}".format(filename, round(conv_var, 3), i_var))
			while delta_conv < conv_var  and i <= i_var:
				b_chnnl = cv2.morphologyEx(b_chnnl, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2+i,2+i)    ), iterations=1+i) #Open to get rid of the noise
				cntss = cv2.findContours(b_chnnl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
				cntss = cntss[0] if len(cntss) == 2 else cntss[1]
				cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
				for cs in cntss:
					perimeters = cv2.arcLength(cs,True); hulls = cv2.convexHull(cs); hull_areas = cv2.contourArea(hulls); hullperimeters = cv2.arcLength(hulls,True)
					Convexity2 = hullperimeters/perimeters
					delta_conv = Convexity2-Convexity
			
				log.info("[SILK]--{}--Convexity: {}, delta convexity: {}, iteration: {}".format(filename, round(Convexity2, 3), round(delta_conv, 3), i))
				i = i + 1

		ear[b_chnnl == 0] = 0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Calculate widths to orient ears
		if args.rotate is True:
			sect = ranges(ear.shape[0], 3)
			ori_width = []
			for i in range(3):
				see = sect[i].split (",")
				wid = r.copy()
				wid2 = r.copy()
				wid[int(see[0]):int(see[1]), :] = 0
				wid2[wid != 0] = 0
				wid2 = cnctfill(wid2)
				if cv2.countNonZero(mask) != 0:
					cntss = cv2.findContours(wid2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
					cntss = cntss[0] if len(cntss) == 2 else cntss[1]
					cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
					for cs in cntss:
						rects = cv2.minAreaRect(cs)
						boxs = cv2.boxPoints(rects)
						boxs = np.array(boxs, dtype="int")			
						boxs = order_points(boxs)
						(tls, trs, brs, bls) = boxs
						(tlblXs, tlblYs) = ((tls[0] + bls[0]) * 0.5, (tls[1] + bls[1]) * 0.5)
						(trbrXs, trbrYs) = ((trs[0] + brs[0]) * 0.5, (trs[1] + brs[1]) * 0.5)
						thmp = dist.euclidean((tlblXs, tlblYs), (trbrXs, trbrYs)) #pixel width
				else:
					thmp = 0
				ori_width.append(thmp)
			if ori_width[2] < ori_width[0]:
				log.warning('[EAR]--{}--Ear rotated'.format(filename))
				ear = cv2.rotate(ear, cv2.ROTATE_180)		

		_,_,r = cv2.split(ear)											#Split into it channel constituents
		_,r = cv2.threshold(r, 0, 255, cv2.THRESH_OTSU)
		r = cnctfill(r)
		ear[r == 0] = 0

		if args.no_save is False:
			destin = "{}".format(out) + "02_Ear_ROIs/"
			if not os.path.exists(destin):
				try:
					os.mkdir(destin)
				except OSError as e:
					if e.errno != errno.EEXIST:
						raise			
			destin = "{}02_Ear_ROIs/{}_ear_{}".format(out, filename, n) + ".jpeg"
			log.info("[EAR]--{}--Ear {} ROI saved to: {}".format(filename, n, destin))			
			cv2.imwrite(destin, ear, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

		#if print proof

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
############################# BASIC FULL EAR FEATURES ####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		Ear_Area = Convexity = Solidity = Ellipse = Ear_Box_Width = Ear_Box_Length = Ear_Box_Area = Ear_Extreme_Length = Ear_Area_DP = Solidity_PolyDP = Solidity_Box = Taper_PolyDP = Taper_Box = Widths = Widths_Sdev = Cents_Sdev = Ear_area = Tip_Area = Bottom_Area = Krnl_Area = Tip_Fill = Blue = Green = Red = Hue = Sat = Vol = Light = A_chnnl = B_chnnl = second_width = mom1 = None
		_,_,r = cv2.split(ear)											#Split into it channel constituents
		_,r = cv2.threshold(r, 0, 255, cv2.THRESH_OTSU)
		lab = cv2.cvtColor(ear, cv2.COLOR_BGR2LAB)
		lab[r == 0] = 0
		_,_,b_chnnl = cv2.split(lab)										#Split into it channel constituents
		hsv = cv2.cvtColor(ear, cv2.COLOR_BGR2HSV)						#Convert into HSV color Space	
		hsv[r == 0] = 0
		_,s,_ = cv2.split(hsv)											#Split into it channel constituents	
		ear_proof = ear.copy()

		cntss = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		cntss = cntss[0] if len(cntss) == 2 else cntss[1]
		cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
		for cs in cntss:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
####################### Area, Convexity, Solidity, fitEllipse ############################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
			Ear_Area = cv2.contourArea(cs)
			perimeters = cv2.arcLength(cs,True); hulls = cv2.convexHull(cs); hull_areas = cv2.contourArea(hulls); hullperimeters = cv2.arcLength(hulls,True)
			Convexity = hullperimeters/perimeters
			Solidity = float(Ear_Area)/hull_areas
			Ellipse = cv2.fitEllipse(cs)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##################################### EAR BOX ############################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
			rects = cv2.minAreaRect(cs)
			boxs = cv2.boxPoints(rects)
			boxs = np.array(boxs, dtype="int")			
			boxs1 = order_points(boxs)
# loop over the original points and draw them
# unpack the ordered bounding box, then compute the midpoint
			(tls, trs, brs, bls) = boxs
			(tltrXs, tltrYs) = ((tls[0] + trs[0]) * 0.5, (tls[1] + trs[1]) * 0.5)
			(blbrXs, blbrYs) = ((bls[0] + brs[0]) * 0.5, (bls[1] + brs[1]) * 0.5)
			(tlblXs, tlblYs) = ((tls[0] + bls[0]) * 0.5, (tls[1] + bls[1]) * 0.5)
			(trbrXs, trbrYs) = ((trs[0] + brs[0]) * 0.5, (trs[1] + brs[1]) * 0.5)
# compute the Euclidean distance between the midpoints
			Ear_Box_Width = dist.euclidean((tltrXs, tltrYs), (blbrXs, blbrYs)) #pixel length
			Ear_Box_Length = dist.euclidean((tlblXs, tlblYs), (trbrXs, trbrYs)) #pixel width
			if Ear_Box_Width > Ear_Box_Length:
				Ear_Box_Width = dist.euclidean((tlblXs, tlblYs), (trbrXs, trbrYs)) #pixel width
				Ear_Box_Length = dist.euclidean((tltrXs, tltrYs), (blbrXs, blbrYs)) #pixel length
				cv2.line(ear_proof, (int(tltrXs), int(tltrYs)), (int(blbrXs), int(blbrYs)), (165, 105, 189), 7) #length
				cv2.circle(ear_proof, (int(tltrXs), int(tltrYs)), 15, (165, 105, 189), -1) #left midpoint
				cv2.circle(ear_proof, (int(blbrXs), int(blbrYs)), 15, (165, 105, 189), -1) #right midpoint
			else:
				cv2.line(ear_proof, (int(tlblXs), int(tlblYs)), (int(trbrXs), int(trbrYs)), (165, 105, 189), 7) #length
				cv2.circle(ear_proof, (int(tlblXs), int(tlblYs)), 15, (165, 105, 189), -1) #left midpoint
				cv2.circle(ear_proof, (int(trbrXs), int(trbrYs)), 15, (165, 105, 189), -1) #right midpoint
			if PixelsPerMetric is not None:
				Ear_Box_Length = Ear_Box_Length/ (PixelsPerMetric)
				Ear_Box_Width = Ear_Box_Width/ (PixelsPerMetric)
			Ear_Box_Area = float(Ear_Box_Length*Ear_Box_Width)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### EXTREME POINTS ######################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
			extTops = tuple(cs[cs[:, :, 1].argmin()][0])
			extBots = tuple(cs[cs[:, :, 1].argmax()][0])
			Ear_Extreme_Length = dist.euclidean(extTops, extBots)
			cv2.circle(ear_proof, extTops, 15, (255, 255, 204), -1)
			#cv2.circle(ear_proof, extBots, 30, (156, 144, 120), -1)
			#cv2.line(ear_proof, extTops, extBots, (0, 0, 155), 10)
			if PixelsPerMetric is not None:
				Ear_Extreme_Length = Ear_Extreme_Length / (PixelsPerMetric)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### POLY DP ######################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		cntss = cv2.findContours(b_chnnl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		cntss = cntss[0] if len(cntss) == 2 else cntss[1]
		cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
		for cs in cntss:
			arclen = cv2.arcLength(cs, True)
	# do approx
			eps = 0.001
			epsilon = arclen * eps
			approx = cv2.approxPolyDP(cs, epsilon, True)
# draw the result
			canvas = np.zeros_like(b_chnnl)
			cv2.drawContours(canvas, [approx], -1, (255), 2, cv2.LINE_AA)

		cntss = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		cntss = cntss[0] if len(cntss) == 2 else cntss[1]
		cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
		for cs in cntss:
			Ear_Area_DP = cv2.contourArea(cs)
			perimeters = cv2.arcLength(cs,True); hulls = cv2.convexHull(cs); hull_areas = cv2.contourArea(hulls); hullperimeters = cv2.arcLength(hulls,True)
			if perimeters != 0:
				Convexity_polyDP = hullperimeters/perimeters
			if hull_areas != 0:
				Solidity_PolyDP = float(Ear_Area_DP)/hull_areas
				Solidity_Box = float(Ear_Area_DP)/ Ear_Box_Area

		canvas[int(1-(ymax/2)): ymax, :] = 0 #FOR THE TIP
		cntss = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		cntss = cntss[0] if len(cntss) == 2 else cntss[1]
		cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
		for cs in cntss:
			Taper_Area_DP = cv2.contourArea(cs)
			perimeters = cv2.arcLength(cs,True); hulls = cv2.convexHull(cs); hull_areas = cv2.contourArea(hulls); hullperimeters = cv2.arcLength(hulls,True)
			if hull_areas != 0:
				Taper_PolyDP = float(Taper_Area_DP)/hull_areas
				Taper_Box = float(Taper_Area_DP)/ Ear_Box_Area
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
######################################### WIDTHS #########################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		Cents = []
		Widths = []
		wid_proof = ear.copy()	
		sect = ranges(ear.shape[0], 20)
	
		for i in range(20):
			see = sect[i].split (",")
			wid = r.copy()
			wid2 = r.copy()
			wid[int(see[0]):int(see[1]), :] = 0
			wid2[wid != 0] = 0
			wid2 = cnctfill(wid2)
			if cv2.countNonZero(mask) != 0:
				cntss = cv2.findContours(wid2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
				cntss = cntss[0] if len(cntss) == 2 else cntss[1]
				cntss = sorted(cntss, key=cv2.contourArea, reverse=False)[:len(cntss)]
				M1 = cv2.moments(wid2)
				if M1["m00"] != 0:
					cX = int(M1["m10"] / M1["m00"])
					Cents.append(cX)
					cY = int(M1["m01"] / M1["m00"])
					for cs in cntss:
						rects = cv2.minAreaRect(cs)
						boxs = cv2.boxPoints(rects)
						boxs = np.array(boxs, dtype="int")			
						boxs = order_points(boxs)
						(tls, trs, brs, bls) = boxs
						(tlblXs, tlblYs) = ((tls[0] + bls[0]) * 0.5, (tls[1] + bls[1]) * 0.5)
						(trbrXs, trbrYs) = ((trs[0] + brs[0]) * 0.5, (trs[1] + brs[1]) * 0.5)
						Ear_Width_B = dist.euclidean((tlblXs, tlblYs), (trbrXs, trbrYs)) #pixel width
						if PixelsPerMetric is not None:
							Ear_Width_B = Ear_Width_B/ (PixelsPerMetric)
						Widths.append(Ear_Width_B)
						cv2.line(wid_proof, (int(tlblXs), int(tlblYs)), (int(trbrXs), int(trbrYs)), (33, 43, 156), 7) #width
					cv2.circle(wid_proof, (cX, cY), 20, (176, 201, 72), -1)
			
		Widths_Sdev = stdev(Widths)		
		Cents_Sdev = stdev(Cents) 
		log.info("[EAR]--{}--Collected basic ear features".format(filename))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################### ADV FEAUTURES #######################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~Dominant Color
		pixels = np.float32(ear[r != 0].reshape(-1, 3))
		n_colors = 2
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
		flags = cv2.KMEANS_RANDOM_CENTERS
		_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
		_, counts = np.unique(labels, return_counts=True)
		dominant = palette[np.argmax(counts)]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##########################################TIP#############################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#	
		bw = r.copy()
		s_p = ear.copy()
		ymax = s.shape[0]
		s = cv2.cvtColor(s,cv2.COLOR_GRAY2RGB)
		chnnl = s
		mskd,_ = cv2.threshold(chnnl[chnnl !=  0],1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU); 
#~~~~~~~~~~~~~~~~~~~~~~~~~~Tip segmentation 
		if args.tip is not None:
			if args.tip[0] == 0 and args.tip[1] == 0 and args.tip[2] == 0 and args.tip[3] == 0:
				log.info("[TIP]--{}--Tip segmentation module off".format(filename))
				chnnl = apply_brightness_contrast(chnnl, brightness = 0, contrast = 0)	#Split into it channel constituents
				chnnl = cv2.cvtColor(chnnl,cv2.COLOR_RGB2GRAY)
				chnnl = cv2.threshold(chnnl, int(mskd-(mskd*0.99)),256, cv2.THRESH_BINARY_INV)[1]
				chnnl[int((ymax*(99/100))):ymax, :] = 0 #FOR THE TIP
				bw[chnnl == 0] = 0		
				nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=4)
				tip = np.zeros(chnnl.shape, np.uint8)	
				for i in range (1,10):
					tip[output == i] = 255
				tip = cv2.cvtColor(tip,cv2.COLOR_GRAY2RGB)
				s_p[tip == 255] = 0
				_,_,tip = cv2.split(s_p)											#Split into it channel constituents
				_,tip = cv2.threshold(tip, 0, 255, cv2.THRESH_OTSU)
				tip = cnctfill(tip)

			else:
				log.info("[TIP]--{}--Custom tip segmentation module".format(filename))
				chnnl = apply_brightness_contrast(chnnl, brightness = 0, contrast = args.tip[1])	#Split into it channel constituents
				chnnl = cv2.cvtColor(chnnl,cv2.COLOR_RGB2GRAY)
				chnnl = cv2.threshold(chnnl, int(mskd-(mskd*args.tip[2])),256, cv2.THRESH_BINARY_INV)[1]
				chnnl[int((ymax*(int(args.tip[0])/100))):ymax, :] = 0 #FOR THE TIP
				bw[chnnl == 0] = 0		
				if args.tip[2] > 0:
					bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(args.tip[3]),int(args.tip[3]))), iterations=int(args.tip[3]))
				nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=4)
				tip = np.zeros(chnnl.shape, np.uint8)	
				for i in range (1,10):
					tip[output == i] = 255
				tip = cv2.cvtColor(tip,cv2.COLOR_GRAY2RGB)
				s_p[tip == 255] = 0
				_,_,tip = cv2.split(s_p)											#Split into it channel constituents
				_,tip = cv2.threshold(tip, 0, 255, cv2.THRESH_OTSU)
				tip = cnctfill(tip)
#~~~~~~~~~~~~~~~~~~~~~~~~~~Special case for white ears otherwise use k means
		else:
			if mskd < 80 and dominant[0] > 140:
				log.warning("[TIP]--{}--Special white kernel tip segmentation module triggered".format(filename))
				chnnl = apply_brightness_contrast(chnnl, brightness = 0, contrast = 10)	#Split into it channel constituents
				chnnl = cv2.cvtColor(chnnl,cv2.COLOR_RGB2GRAY)
				chnnl = cv2.threshold(chnnl, int(mskd-(mskd*0.95)),256, cv2.THRESH_BINARY_INV)[1]
				chnnl[int((ymax*(int(50)/100))):ymax, :] = 0 #FOR THE TIP
				bw[chnnl == 0] = 0		
				bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
				nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=4)
				tip = np.zeros(chnnl.shape, np.uint8)	
				for i in range (1,10):
					tip[output == i] = 255
				tip = cv2.cvtColor(tip,cv2.COLOR_GRAY2RGB)	
				s_p[tip == 255] = 0
				_,_,tip = cv2.split(s_p)											#Split into it channel constituents
				_,tip = cv2.threshold(tip, 0, 255, cv2.THRESH_OTSU)
				tip = cnctfill(tip)
			else:
				log.info("[TIP]--{}--Default tip segmentation module".format(filename))
				chnnl = apply_brightness_contrast(chnnl, brightness = 0, contrast = 15)	#Split into it channel constituents
				vectorized = chnnl.reshape((-1,3))
				vectorized = np.float32(vectorized)
				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
				K = 2
				attempts = 5
				ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
				center = np.uint8(center)
				res = center[label.flatten()]
				chnnl = res.reshape((chnnl.shape))
				chnnl = cv2.cvtColor(chnnl,cv2.COLOR_RGB2GRAY)
				_,chnnl = cv2.threshold(chnnl, 0, 255, cv2.THRESH_OTSU)	
				chnnl=cv2.bitwise_not(chnnl)
				chnnl[int((ymax*(int(50)/100))):ymax, :] = 0 #FOR THE TIP	
				bw[chnnl == 0] = 0
				nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=4)
				tip = np.zeros(chnnl.shape, np.uint8)
				for i in range (1,5):
					tip[output == i] = 255
				s_p[tip == 255] = 0
				_,_,tip = cv2.split(s_p)											#Split into it channel constituents
				_,tip = cv2.threshold(tip, 0, 255, cv2.THRESH_OTSU)
				tip = cnctfill(tip)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#####################################  BOTTOM  ###########################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#	
		bw = r.copy()
		s_p = ear.copy()
		hsv = cv2.cvtColor(ear, cv2.COLOR_BGR2HSV)						#Convert into HSV color Space	
		hsv[ear == 0] = 0
		_,s,_ = cv2.split(hsv)											#Split into it channel constituents	
		ymax = s.shape[0]
		s = cv2.cvtColor(s,cv2.COLOR_GRAY2RGB)
		chnnl = s
		mskd,_ = cv2.threshold(chnnl[chnnl !=  0],1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU); 
#~~~~~~~~~~~~~~~~~~~~~~~~~~Tip segmentation 
		if args.bottom is not None:
			if args.bottom[0] == 0 and args.bottom[1] == 0 and args.bottom[2] == 0 and args.bottom[3] == 0:
				log.info("[BTM]--{}--Bottom segmentation module off".format(filename))
				chnnl = apply_brightness_contrast(chnnl, brightness = 0, contrast = 0)	#Split into it channel constituents
				chnnl = cv2.cvtColor(chnnl,cv2.COLOR_RGB2GRAY)
				chnnl = cv2.threshold(chnnl, int(mskd-(mskd*0.7)),256, cv2.THRESH_BINARY_INV)[1]
				chnnl[0:int((ymax*(1-(int(99)/100)))), :] = 0 #FOR THE bottom
				bw[chnnl == 0] = 0		
				nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=4)
				bottom = np.zeros(chnnl.shape, np.uint8)			
				# take a binary image and run a connected component analysis
				nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
				# extracts sizes vector for each connected component
				sizes = stats[:, -1]
				if len(sizes) > 1:
					for i in range(2, nb_components):
						if sizes[i] > ((r.shape[0]*r.shape[1])*0.001):
							bottom[output == i] = 255
				bottom[output == len(centroids)] = 255
				bottom[output == int(len(centroids)-1)] = 255
				s_p[bottom == 255] = 0
				_,_,bottom = cv2.split(s_p)											#Split into it channel constituents
				_,bottom = cv2.threshold(bottom, 0, 255, cv2.THRESH_OTSU)
				bottom = cnctfill(bottom)			
			else:
				log.info("[BTM]--{}--Custom bottom segmentation module".format(filename))
				chnnl = apply_brightness_contrast(chnnl, brightness = 0, contrast = args.bottom[1])	#Split into it channel constituents
				chnnl = cv2.cvtColor(chnnl,cv2.COLOR_RGB2GRAY)
				chnnl = cv2.threshold(chnnl, int(mskd-(mskd*args.bottom[2])),256, cv2.THRESH_BINARY_INV)[1]
				chnnl[0:int((ymax*(1-int(args.bottom[0])/100))), :] = 0 #FOR THE bottom
				bw[chnnl == 0] = 0		
				if args.bottom[2] > 0:
					bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(args.bottom[3]),int(args.bottom[3]))), iterations=int(args.bottom[3]))
				nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=4)
				bottom = np.zeros(chnnl.shape, np.uint8)			
				# take a binary image and run a connected component analysis
				nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
				# extracts sizes vector for each connected component
				sizes = stats[:, -1]
				if len(sizes) > 1:
					for i in range(2, nb_components):
						if sizes[i] > ((r.shape[0]*r.shape[1])*0.001):
							bottom[output == i] = 255
				bottom[output == len(centroids)] = 255
				bottom[output == int(len(centroids)-1)] = 255
				s_p[bottom == 255] = 0
				_,_,bottom = cv2.split(s_p)											#Split into it channel constituents
				_,bottom = cv2.threshold(bottom, 0, 255, cv2.THRESH_OTSU)
				bottom = cnctfill(bottom)			
		else:
			if mskd < 80 and dominant[0] > 140:
				chnnl = apply_brightness_contrast(chnnl, brightness = 0, contrast = 5)	#Split into it channel constituents
				chnnl = cv2.cvtColor(chnnl,cv2.COLOR_RGB2GRAY)
				chnnl = cv2.threshold(chnnl, int(mskd-(mskd*0.85)),256, cv2.THRESH_BINARY_INV)[1]
				chnnl[0:int((ymax*(1-int(25)/100))), :] = 0 #FOR THE bottom
				bw[chnnl == 0] = 0		
				nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=4)
				bottom = np.zeros(chnnl.shape, np.uint8)			
				# take a binary image and run a connected component analysis
				nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
				# extracts sizes vector for each connected component
				sizes = stats[:, -1]
				if len(sizes) > 1:
					for i in range(2, nb_components):
						if sizes[i] > ((r.shape[0]*r.shape[1])*0.001):
							bottom[output == i] = 255
				bottom[output == len(centroids)] = 255
				bottom[output == int(len(centroids)-1)] = 255
				s_p[bottom == 255] = 0
				_,_,bottom = cv2.split(s_p)											#Split into it channel constituents
				_,bottom = cv2.threshold(bottom, 0, 255, cv2.THRESH_OTSU)
				bottom = cnctfill(bottom)
			else:
				vectorized = chnnl.reshape((-1,3))
				vectorized = np.float32(vectorized)
				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
				K = 2
				attempts = 5
				ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
				center = np.uint8(center)
				res = center[label.flatten()]
				chnnl = res.reshape((chnnl.shape))
				chnnl = cv2.cvtColor(chnnl,cv2.COLOR_RGB2GRAY)
				_,chnnl = cv2.threshold(chnnl, 0, 255, cv2.THRESH_OTSU)	
				chnnl=cv2.bitwise_not(chnnl)
				chnnl[0:int((ymax*(1-int(25)/100))), :] = 0 #FOR THE bottom
				bw[chnnl == 0] = 0
				bottom = np.zeros(chnnl.shape, np.uint8)			
				# take a binary image and run a connected component analysis
				nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
				# extracts sizes vector for each connected component
				sizes = stats[:, -1]
				if len(sizes) > 1:
					for i in range(2, nb_components):
						if sizes[i] > ((r.shape[0]*r.shape[1])*0.001):
							bottom[output == i] = 255
				bottom[output == len(centroids)] = 255
				bottom[output == int(len(centroids)-1)] = 255
				s_p[bottom == 255] = 0
				_,_,bottom = cv2.split(s_p)											#Split into it channel constituents
				_,bottom = cv2.threshold(bottom, 0, 255, cv2.THRESH_OTSU)
				bottom = cnctfill(bottom)

		tip = cv2.bitwise_not(tip)
		bottom = cv2.bitwise_not(bottom)
		tipbottom = tip + bottom
		tipbottom = cv2.bitwise_not(tipbottom)
		
#########weird buttip patch
	
		krnl = ear.copy()
		log.warning("{}".format(cv2.countNonZero(tipbottom)))
		if cv2.countNonZero(tipbottom) < 300000:
			tipbottom = r.copy()
		else:		
			krnl[tipbottom == 0] = 0
########

		
		full = ear.copy()
		full[int((ymax*(int(50)/100))):ymax, :] = [59,106,59] #FOR THE TIP
		full[0:int((ymax*(1-int(25)/100))), :] = [140,97,33] #FOR THE bottom
		full[tipbottom != 0] = ear[tipbottom != 0]

		if mskd < 80 and dominant[0] > 140:
			if args.tip is None or args.bottom is None:
				logo = np.zeros((64, 64, 3), dtype=np.uint8)
				CENTER = (32, 32)
				cv2.circle(logo, CENTER, 48, (100,0,167), -1)
				TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
				TEXT_SCALE = 1
				TEXT_THICKNESS = 2
				TEXT = "YT"
				text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
				text_origin = (CENTER[0] - text_size[0] / 2, CENTER[1] + text_size[1] / 2)
				cv2.putText(logo, TEXT, (int(text_origin[0]),int(text_origin[1])) , cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, (255,255,255), TEXT_THICKNESS, cv2.LINE_AA)
				x_offset= int(full.shape[1]/2)-20
				y_offset= int(full.shape[0]/2)
				full[y_offset:y_offset+logo.shape[0], x_offset:x_offset+logo.shape[1]] = logo
		if args.tip is not None:
			if args.tip[0] == 0 and args.tip[1] == 0 and args.tip[2] == 0:
				tip = 0
				tipbottom = r.copy()
				full = ear.copy()
			else:
				logo = np.zeros((96, 96, 3), dtype=np.uint8)
				CENTER = (49, 48)
				cv2.circle(logo, CENTER, 300, (100,0,167), -1)
				TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
				TEXT_SCALE = .5
				TEXT_THICKNESS = 1
				TEXT = "Cust Tip"
				text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
				text_origin = (CENTER[0] - text_size[0] / 2, CENTER[1] + text_size[1] / 2)
				cv2.putText(logo, TEXT, (int(text_origin[0]),int(text_origin[1])) , cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, (255,255,255), TEXT_THICKNESS, cv2.LINE_AA)
				x_offset= int(full.shape[1]/2)-20
				y_offset= int(full.shape[0]/2)
				full[y_offset:y_offset+logo.shape[0], x_offset:x_offset+logo.shape[1]] = logo

		if args.bottom is not None:
			if args.bottom[0] == 0 and args.bottom[1] == 0 and args.bottom[2] == 0:
				bottom = 0
				tipbottom = r.copy()
				full = ear.copy()
			else:
				logo = np.zeros((96, 96, 3), dtype=np.uint8)
				CENTER = (48, 48)
				cv2.circle(logo, CENTER, 300, (100,0,167), -1)
				TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
				TEXT_SCALE = .5
				TEXT_THICKNESS = 1
				TEXT = "Cust bottom"
				text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
				text_origin = (CENTER[0] - text_size[0] / 2, CENTER[1] + text_size[1] / 2)
				cv2.putText(logo, TEXT, (int(text_origin[0]),int(text_origin[1])) , cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, (255,255,255), TEXT_THICKNESS, cv2.LINE_AA)
				x_offset= int(full.shape[1]/2)-20
				y_offset= int(full.shape[0]/2)+100
				full[y_offset:y_offset+logo.shape[0], x_offset:x_offset+logo.shape[1]] = logo

		_,tipbottom = cv2.threshold(tipbottom, 0, 255, cv2.THRESH_OTSU)
		Ear_area = cv2.countNonZero(r)
		Tip_Area = cv2.countNonZero(tip)
		Bottom_Area = cv2.countNonZero(bottom)
		Krnl_Area = cv2.countNonZero(tipbottom)
		Tip_Fill = (Ear_area-Tip_Area)/Ear_area 
		
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
################################## KERNEL FEATS ##########################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		_, cntss, _ = cv2.findContours(tipbottom, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		for cs in cntss:
			ear_proof =cv2.drawContours(ear_proof, cs, -1, (244, 143, 177), 7)
			rects = cv2.minAreaRect(cs)
			boxs = cv2.boxPoints(rects)
			boxs = np.array(boxs, dtype="int")			
			boxs1 = order_points(boxs)
# loop over the original points and draw them
# unpack the ordered bounding box, then compute the midpoint
			(tls, trs, brs, bls) = boxs1
			(tltrXs, tltrYs) = ((tls[0] + trs[0]) * 0.5, (tls[1] + trs[1]) * 0.5)
			(blbrXs, blbrYs) = ((bls[0] + brs[0]) * 0.5, (bls[1] + brs[1]) * 0.5)
# compute the Euclidean distance between the midpoints
			Kernel_Length = dist.euclidean((tltrXs, tltrYs), (blbrXs, blbrYs)) #pixel length
			if PixelsPerMetric is not None:
				Kernel_Length = Kernel_Length/ (PixelsPerMetric)

#~~~~~~~~~~~~~~~~~~~~~~~~~Dominant Color
		pixels = np.float32(krnl[tipbottom != 0].reshape(-1, 3))
		n_colors = 2
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
		flags = cv2.KMEANS_RANDOM_CENTERS
		_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
		_, counts = np.unique(labels, return_counts=True)
		dominant = palette[np.argmax(counts)]
		Blue = dominant[0]
		Red = dominant[1]
		Green = dominant[2]
#~~~~~~~~~~~~~~~~~~~~~~~~~Dominant HSV Color
		hsv = cv2.cvtColor(krnl, cv2.COLOR_BGR2HSV)						#Convert into HSV color Space	
		pixels = np.float32(hsv[tipbottom != 0].reshape(-1, 3))
		n_colors = 2
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
		flags = cv2.KMEANS_RANDOM_CENTERS
		_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
		_, counts = np.unique(labels, return_counts=True)
		dominant = palette[np.argmax(counts)]
		Hue = dominant[0]
		Sat = dominant[1]
		Vol = dominant[2]

#~~~~~~~~~~~~~~~~~~~~~~~~~Dominant LAB Color
		lab = cv2.cvtColor(krnl, cv2.COLOR_BGR2LAB)						#Convert into HSV color Space	
		pixels = np.float32(lab[tipbottom != 0].reshape(-1, 3))
		n_colors = 2
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
		flags = cv2.KMEANS_RANDOM_CENTERS
		_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
		_, counts = np.unique(labels, return_counts=True)
		dominant = palette[np.argmax(counts)]
		Light = dominant[0]
		A_chnnl = dominant[1]
		B_chnnl = dominant[2]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
################################## KERNEL SEGEMENTATION ##################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CROP IMAGE
		krnl = ear.copy()
		ymax = krnl.shape[0]
		krnl[int((ymax*0.55)):ymax, :] = 0 #FOR THE TIP
		krnl[0:int((ymax*(1-int(50)/100))), :] = 0 #FOR THE bottom

		gray = cv2.cvtColor(krnl, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

		cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if len(cnts) == 2 else cnts[1]
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
		for c in cnts:
			x,y,w,h = cv2.boundingRect(c)
			krnl = krnl[y:y+h, x:x+w]
			break
	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~KRN Dataset #1	
		_,g_krnl,_ = cv2.split(krnl)
		hsv = cv2.cvtColor(krnl, cv2.COLOR_BGR2HSV)
		hsv[krnl == 0] = 0
		_,s_krnl,_ = cv2.split(hsv)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Resize image
		dim = (350, 70)
		s_mom = cv2.moments(s_krnl)
		s_krnl = cv2.resize(s_krnl, dim, interpolation = cv2.INTER_AREA)
		s_krnl_l = s_krnl.tolist()
		s_krnl_list = []
		for sublist in s_krnl_l:
			for item in sublist:
				s_krnl_list.append(item)


		g_mom = cv2.moments(g_krnl)
		g_krnl = cv2.resize(g_krnl, dim, interpolation = cv2.INTER_AREA)
		g_krnl_l = g_krnl.tolist()
		g_krnl_list = []
		for sublist in g_krnl_l:
			for item in sublist:
				g_krnl_list.append(item)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		#if Cents_Sdev > 35:
		second_width = mean(Widths[7:11])
		#else:
		#	second_width = Ear_Box_Width
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#		

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
########################################OUTPUT############################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Create CSV
		if args.no_save is False:			
			#csvname = out + 'features' +'.csv'
			#file_exists = os.path.isfile(csvname)
			#with open (csvname, 'a') as csvfile:
		#		headers = ['Filename', 'Ear Number','Ear_Area', 'Convexity', 'Solidity', 'Ellipse', 'Ear_Box_Width', 'Ear_Box_Length', 'Ear_Box_Area', 'Ear_Extreme_Length', 'Ear_Area_DP', 'Solidity_PolyDP', 'Solidity_Box', 'Taper_PolyDP', 'Taper_Box', 'Widths', 'Widths_Sdev', 'Cents_Sdev', 'Ear_area', 'Tip_Area', 'Bottom_Area', 'Kernel_Area', 'Tip_Fill', 'Blue', 'Green', 'Red', 'Hue', 'Saturation', 'Volume', 'Light', 'A_Channel', 'B_Channel', 'second_width']  
		#		writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
		#		if not file_exists:
		#			writer.writeheader()  # file doesn't exist yet, write a header	
	
		#		writer.writerow({'Filename': filename,'Ear Number': n, 'Ear_Area': Ear_Area, 'Convexity': Convexity, 'Solidity': Solidity, 'Ellipse': Ellipse, 'Ear_Box_Width': Ear_Box_Width, 'Ear_Box_Length': Ear_Box_Length, 'Ear_Box_Area': Ear_Box_Area, 'Ear_Extreme_Length': Ear_Extreme_Length, 'Ear_Area_DP': Ear_Area_DP, 'Solidity_PolyDP': Solidity_PolyDP, 'Solidity_Box': Solidity_Box, 'Taper_PolyDP': Taper_PolyDP, 'Taper_Box': Taper_Box, 'Widths': Widths, 'Widths_Sdev': Widths_Sdev, 'Cents_Sdev': Cents_Sdev, 'Ear_area': Ear_area, 'Tip_Area': Tip_Area, 'Bottom_Area': Bottom_Area, 'Kernel_Area': Krnl_Area, 'Tip_Fill': Tip_Fill, 'Blue': Blue, 'Green': Green, 'Red': Red, 'Hue': Hue, 'Saturation': Sat, 'Volume': Vol, 'Light': Light, 'A_Channel': A_chnnl, 'B_Channel': B_chnnl, 'second_width': second_width})
		
		#	log.info("[EAR]--{}--Saved features to: {}features.csv".format(filename, out))


			csvname = out + 'krnl_moments' +'.csv'
			file_exists = os.path.isfile(csvname)
			with open (csvname, 'a') as csvfile:
				headers = ['Filename', 'ear_mom', 'green_mom', 'sat_mom']  
				writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
				if not file_exists:
					writer.writeheader()  # file doesn't exist yet, write a header	
	
				writer.writerow({'Filename': filename, 'ear_mom': M, 'green_mom': g_mom, 'sat_mom': s_mom})
		
			log.info("[EAR]--{}--Saved features to: {}krnl_moments.csv".format(filename, out))


			csvname = out + 'krnls_s' +'.csv'
			f = open(csvname, 'a')
			with f:
				writer = csv.writer(f)
				writer.writerow(s_krnl_list)		
			log.info("[EAR]--{}--Saved krnls_saturation to: {}".format(filename, csvname))

			csvname = out + 'krnls_g' +'.csv'
			f = open(csvname, 'a')
			with f:
				writer = csv.writer(f)
				writer.writerow(g_krnl_list)		
			log.info("[EAR]--{}--Saved krnls_green to: {}".format(filename, csvname))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Save proofs
			destin = "{}".format(out) + "03_Ear_Proofs/"
			if not os.path.exists(destin):
				try:
					os.mkdir(destin)
				except OSError as e:
					if e.errno != errno.EEXIST:
						raise

			im_a = cv2.hconcat([ear, full, ear_proof, wid_proof])

			destin = "{}03_Ear_Proofs/{}_proof_ear_{}".format(out, filename, n) + ".jpeg"
			log.info("[DONE]--{}--Ear {} proof saved to: {}".format(filename, n, destin))			
			#cv2.imwrite(destin, im_a, [int(cv2.IMWRITE_JPEG_QUALITY), 20])

			if args.no_proof is False:
				cv2.namedWindow('Found Ears', cv2.WINDOW_NORMAL)
				cv2.resizeWindow('Found Ears', 800, 800)
				cv2.imshow('Found Ears', im_a); cv2.waitKey(5000); cv2.destroyAllWindows() 

			destin = "{}".format(out) + "04_Krnl_Proofs/"
			if not os.path.exists(destin):
				try:
					os.mkdir(destin)
				except OSError as e:
					if e.errno != errno.EEXIST:
						raise
			
			#destin = "{}04_Krnl_Proofs/sat_{}_ear_{}".format(out, filename, n) + ".jpeg"
			#log.info("[DONE]--{}--Ear {} proof saved to: {}".format(filename, n, destin))			
			#cv2.imwrite(destin, s_krnl, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

			#destin = "{}04_Krnl_Proofs/green_{}_ear_{}".format(out, filename, n) + ".jpeg"
			#log.info("[DONE]--{}--Ear {} proof saved to: {}".format(filename, n, destin))			
			#cv2.imwrite(destin, g_krnl, [int(cv2.IMWRITE_JPEG_QUALITY), 70])


		n = n+1

	log.info("[DONE]--{}--Finished analysis".format(filename))# Log
	log.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	log.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
main()