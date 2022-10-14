# OCR Program utilizing Tesseract OCR Engine and OpenCV
# Comparitively, this program is outperformed by AWS Rekognition's text extraction
# service - the Tesseract OCR engine is free, open-source and constantly under development 
# and will likely improve over time.

import os
import time
import cv2
import numpy as np
from PIL import Image
from imutils.object_detection import non_max_suppression
import pytesseract
import enchant

from elasticsearch import Elasticsearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import json

valid = enchant.Dict("en_US")

def predictions(scores, geometry):

    # grab the number of rows and columns from the scores volume
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores, followed by the geometry coord
        # data used to calculate potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        # loop over the number of columns
        for x in range(0, numCols):
            # if score has low probability, ignore it
            if scoresData[x] < 0.5:
                continue
            # compute the offset factor
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract prediction rotation angle
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # geometry volume - derive the width and height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    return (rects, confidences)

def processing(image):
    # load the input image
    image = cv2.imread(image)
    orig = image.copy()
    (H, W) = image.shape[:2]

    newH = round((H/32)) * 32 # keep close to orig size but make multiple of 32
    newW = round((W/32)) * 32
    rW = W / float(newW)
    rH = H / float(newH)
   
    # resize the image and grab new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # download into working directory
    net = cv2.dnn.readNet("./frozen_east_text_detection.pb")
    # construct a blob from  image and perform a forward pass of
    # the model to get two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (rects, confidences) = predictions(scores, geometry)
    end = time.time()
    print("[Processing] text detection took {:.6f} seconds".format(end - start))
    # suppress weak, overlapping bounding boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    results = []
    for (startX, startY, endX, endY) in boxes:
        # scale bounding box coords
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        padding = 0.05 # padding around boundary box can increase accuracy
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)

        # adjust coords to reflect padding
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(W, endX + (dX * 2))
        endY = min(H, endY + (dY * 2))

        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]
        
        # lang = english, use LSTM model, treat ROI as raw text
        # look at pytesseract documentation to get other options
        config = ("-l eng --oem 1 --psm 7")

        # pre-processing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        # threshold the image, roughly setting text to black on white background
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # automatic deskewing
        # define all pixels that are black
        coords = np.column_stack(np.where(thresh < 255))
        angle = cv2.minAreaRect(coords)[-1]
        angle *= -1
        if angle < -45:
	        angle = -(90 + angle)
        # otherwise, just take the inverse of the angle to make positive
        else:
            angle = -angle
        # rotate image to deskew based on angle
        (h, w) = thresh.shape[:2]
        center = (w // 2, h // 2) # find center pixel of image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(thresh, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # show the output image
        #print("[INFO] angle: {:.3f}".format(angle))
        #cv2.imshow("Rotated", rotated)
        #cv2.waitKey(0)

        # run Tesseract engine
        text = pytesseract.image_to_string(thresh, config=config)
        results.append(((startX, startY, endX, endY), text))

        # draw bounding box on image
        #cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # show the output image
    #cv2.imshow("Text Detection", orig)
    #cv2.waitKey(0)

    # sort the results bounding box coordinates from image top to bottom
    results = sorted(results, key=lambda r:r[0][1])
    # loop over the results
    whole = ""
    for ((startX, startY, endX, endY), text) in results:
        # display the text
        if not valid.check(text):
            opts = valid.suggest(text)
            text = opts[0] # assume first suggestion is best
        whole += ("{}\n".format(text))
    return whole

host = $host
region = $region

service = 'es'
credentials = boto3.Session(profile_name="lambda").get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

es = Elasticsearch(
    hosts = [{'host': host, 'port': 443}],
    http_auth = awsauth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)

directory = "./images" # keyframes pulled from S3 bucket
for file in os.listdir(directory): # iterate over files in images directory
    filename = directory + '/' + file
    print(filename)
    text = processing(filename)
    document = {
        "filename": filename,
        "ocr": text
    }
    #print(document)
    # write data to elasticsearch index
    response = es.index(index="ocr", doc_type="_doc", body=document)
