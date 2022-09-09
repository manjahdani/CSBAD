# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 10:36:56 2022

@author: grotsartdehe
"""
import matplotlib.pyplot as plt 
import numpy as np 
from skimage.measure import shannon_entropy as entropy
import cv2
import math 


def snr(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd).tolist()

#compute the individual entropy for each frame
def indEntropy(imageList):
    individualEntropy = []
    for image in imageList:
        individualEntropy.append(entropy(image))
        
    meanEntropy = np.mean(individualEntropy)
    resultList = individualEntropy#[meanEntropy]
    return resultList

def indSNR(imageList):
    individualSNR = []
    for image in imageList:
        individualSNR.append(snr(image,axis=None))
    
    meanSNR = np.mean(individualSNR)
    resultList = individualSNR#[meanSNR]
    return resultList

#return the metric's score for each subsampling method
def evaluation(imageFolder,subsetIndices,metricsChoice):
    scoreList = []
    subsetSize = len(subsetIndices[0])#suppose a constant number of subset frames
    for idxList in subsetIndices:#loop for each subsampling method
        #open images and compute the background
        imageList = []
        background = np.zeros((1920,2560))
        for idx_img in idxList:
            frame_folder = "frame_" + str(idx_img) + ".png"
            im = plt.imread(imageFolder + frame_folder)
            im = np.floor((0.2126*im[:,:,0]+0.7152*im[:,:,1]+0.0722*im[:,:,2])*255)
            background += im
            imageList.append(im)
        background /= subsetSize
        
        individualScore = [indEntropy(imageList), indSNR(imageList)]
        scoreList.append(individualScore)
        
        image1 = imageList[0]
        image2 = imageList[1::]
        
        averagingMethod(image1,image2)
        
        #remove background for each frame and then, compute the metric's score
        for j in range(subsetSize):
            imageList[j] = (imageList[j] + 255 - background)//2
        
        individualScore_bg = [indEntropy(imageList), indSNR(imageList)]
        scoreList.append(individualScore_bg)
    return scoreList

