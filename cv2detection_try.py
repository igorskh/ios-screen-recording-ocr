#!/usr/bin/python3
__author__ = "Igor Kim"
__credits__ = ["Igor Kim"]
__maintainer__ = "Igor Kim"
__email__ = "igor.skh@gmail.com"
__status__ = "Development"
__date__ = "05/2019"
__license__ = "MIT"

import numpy as np
import cv2
import argparse, time
import imutils

from imutils.object_detection import non_max_suppression
from imutils import contours

def detect_text(image, east_model_path, layers=["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]):
    size_reversed = (image.shape[:2][1], image.shape[:2][0])
    net = cv2.dnn.readNet(east_model_path)
    blob = cv2.dnn.blobFromImage(image, 1.0, size_reversed, (123.68, 116.78, 103.94), swapRB = True, crop = False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layers)
    end = time.time()
    print("[INFO] text detection took {:.4f} seconds".format(end-start))
    return scores, geometry

def show_boxes(image, boxes):
    output = image.copy()
    for ((startX, startY, endX, endY), text) in boxes:
        cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.imshow("Text Detection", output)
    cv2.waitKey(0)

def save_with_boxes(image, boxes, output_path="output/test.png"):
    output = image.copy()
    for ((startX, startY, endX, endY), text) in boxes:
        cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imwrite(output_path,output)

def resize_image(image, target_size): 
    ratio_height = image.shape[:2][0] / float(target_size[0])
    ratio_width = image.shape[:2][1] / float(target_size[1])
    image_new = image.copy()
    image_new = cv2.resize(image_new, (target_size[1], target_size[0]))
    return image_new, (ratio_height, ratio_width)

def preprocess_image(image, brightness=0, contrast=2000):
    new_image = image.copy()
    new_image = np.int16(new_image)
    new_image = new_image * (contrast/127+1) - contrast + brightness
    new_image = np.clip(new_image, 0, 255)
    _, new_image = cv2.threshold(new_image, 0, 255, cv2.THRESH_BINARY_INV)
    # new_image = cv2.Canny(img_gray,100,200)
    # new_image = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    kernel = np.ones((2,2),np.uint8)
    new_image = cv2.dilate(new_image,kernel,iterations = 2)
    return np.uint8(new_image)

def open_image(image_path):
    return cv2.imread(image_path)

def decode_predictions(scores, geometry, min_confidence=0.5):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue
            # compute the offset factor as our resulting feature maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle) 
            # use the geometry volume to derive the width and height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    return rects, confidences

def get_actual_boxes(boxes, resized_ratio, original_size, padding=0.0):
    (original_height, original_width) = original_size
    (ratio_height, ratio_width) = resized_ratio
    results = []
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box
        startX = int(startX * ratio_width)
        startY = int(startY * ratio_height)
        endX = int(endX * ratio_width)
        endY = int(endY * ratio_height)
        # calc padding
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)
        # apply padding
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(original_width, endX + (dX * 2))
        endY = min(original_height, endY + (dY * 2))
        # extract the actual padded ROI
        # roi = self.original_image[startY:endY, startX:endX]
        results.append(((startX, startY, endX, endY), ""))
    return sorted(results, key=lambda r:r[0][1])

def find_countours(image):
    ref = image.copy()
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts, method="left-to-right")[0]
    return cnts

def save_with_countrous(image, cnts, output_path="output/test.png"):
    clone = image.copy()
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(output_path, clone)

ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", default="assets/frozen_east_text_detection.pb", type = str, help = "path to input EAST Detector")
ap.add_argument("-w", "--width", type = int,
	default = 800, help = "resized image width(should be multiple of 32)")
ap.add_argument("-e", "--height", type = int,
	default = 800, help = "resized image height(should be multiple of 32)")
ap.add_argument("-c", "--min-confidence", type = float,
	default = .5, help = "minimum probability required to inspect a region")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
ap.add_argument("-i", "--input", type=str, help="Input image path")
ap.add_argument("-o", "--output", type=str, help="Output image path")
args = vars(ap.parse_args())

def process_one_image(input_path, output_path):
    target_size = (args["height"], args["width"])

    original_image = open_image(input_path)
    original_size = original_image.shape[:2]
    processed_image = preprocess_image(original_image)
    image, resized_ratio = resize_image(processed_image, target_size)
    scores, geometry = detect_text(image, args["east"])
    rects, confidences = decode_predictions(scores, geometry, args["min_confidence"])
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    results = get_actual_boxes(boxes, resized_ratio, original_size, padding=args["padding"])
    save_with_boxes(original_image, results, output_path)

def process_one_image_1(input_path, output_path):
    original_image = open_image(input_path)
    processed_image = preprocess_image(original_image)
    cnts = find_countours(processed_image)
    save_with_countrous(original_image, cnts, output_path=output_path)

# for i in range(138):
process_one_image("build/images/test3/frame0.png", "screenshots/test2.png")