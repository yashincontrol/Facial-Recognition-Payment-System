# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 12:35:28 2018

@author: bunte
"""

import os

import numpy as np

import cv2

from PIL import Image # For face recognition we will the the LBPH Face Recognizer 

#recognizer = cv2.createLBPHFaceRecognizer();
recognizer = cv2.face.LBPHFaceRecognizer_create()

path="G:\\Yash_CS\\Inceptra\\Faces_Data\\"

def getImagesWithID(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]   

 # print image_path   

 #getImagesWithID(path)

    faces = []

    IDs = []

    for imagePath in imagePaths:      

  # Read the image and convert to grayscale

        facesImg = Image.open(imagePath).convert('L')

        faceNP = np.array(facesImg, 'uint8')

        # Get the label of the image

        ID= int(os.path.split(imagePath)[-1].split(".")[1])

         # Detect the face in the image

        faces.append(faceNP)

        IDs.append(ID)

        cv2.imshow("Adding faces for traning",faceNP)

        cv2.waitKey(10)

    return np.array(IDs), faces

Ids,faces  = getImagesWithID(path)

recognizer.train(faces,Ids)

recognizer.save("G:\\Yash_CS\\Inceptra\\Training Data.yml")

cv2.destroyAllWindows()