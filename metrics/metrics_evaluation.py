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
from sklearn.metrics import mutual_info_score


def temporalAverage(subsetIndices,imageList,shape):
    matrix = np.zeros((len(subsetIndices),len(subsetIndices)))
    for i in range(len(subsetIndices)):
        for j in range(len(subsetIndices)):
            if i < j:
                matrix[i][j] = abs(subsetIndices[i]-subsetIndices[j])
                
    timeInterval = []
    for i in range(len(subsetIndices)-1):
        if i < len(subsetIndices):
            dt = subsetIndices[i+1]-subsetIndices[i]
            timeInterval.append(dt)

    mu = np.mean(timeInterval)  
    listNeighbours = []
    for i in range(len(subsetIndices)):
        li = []
        for j in range(len(subsetIndices)):
            if i < j and matrix[i][j] <= mu:
                if li == []:
                    li = [subsetIndices[i]]
                li.append(subsetIndices[j])
        listNeighbours.append(li)
                
    l = listNeighbours
    l2 = l[:]
    for m in l:
        for n in l:
            
            if set(m).issubset(set(n)) and m != n:
                l2.remove(m)
                break
            
    l2 = list(set(map(tuple,l2)))
    averagedListImages = []
    for indices in l2:
        meanImage = np.zeros(shape)
        for k in indices:
            meanImage += imageList[subsetIndices.index(k)]
        
        meanImage /= len(indices)
        averagedListImages.append(meanImage)
    return averagedListImages

def calc_MI(image1, image2, bins):
    c_xy = np.histogram2d(image1, image2, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi
    
def snr(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.mean(np.where(sd == 0, 0, m/sd).tolist())


#return the metric's score for each subsampling method
def evaluationIntraFrame(imageFolder,subsetIndices,metricsChoice,shape):
    individualEntropy = []
    individualSNR = []
    individualEntropy_bg = []
    individualSNR_bg = []
    subsetSize = len(subsetIndices[0])#suppose a constant number of subset frames
    for idxList in subsetIndices:#loop for each subsampling method
        #open images and compute the background
        background = np.zeros(shape)
        for idx_img in idxList:
            frame_folder = "frame_" + str(idx_img).zfill(4) + ".png"
            im = plt.imread(imageFolder + frame_folder)
            im = np.floor((0.2126*im[:,:,0]+0.7152*im[:,:,1]+0.0722*im[:,:,2])*255)
            background += im
            individualEntropy.append(entropy(im))
            individualSNR.append(snr(im))
            #imageList.append(im)
        background /= subsetSize
        
        for idx_img in idxList:
            frame_folder = "frame_" + str(idx_img).zfill(4) + ".png"
            im = plt.imread(imageFolder + frame_folder)
            im = np.floor((0.2126*im[:,:,0]+0.7152*im[:,:,1]+0.0722*im[:,:,2])*255)
            im = (im + 255 - background)//2
            individualEntropy_bg.append(entropy(im))
            individualSNR_bg.append(snr(im))
    return individualEntropy,individualEntropy_bg,individualSNR,individualSNR_bg

def evaluationInterFrame(imageFolder,subsetIndices,metricsChoice,shape):
    cst = math.log(math.exp(1),2)
    miScore = []
    psnrScore = []
    subsetSize = len(subsetIndices[0])#suppose a constant number of subset frames
    for idxList in subsetIndices:#loop for each subsampling method
        #open images and compute the background
        imageList = []
        background = np.zeros(shape)
        for idx_img in idxList:
            frame_folder = "frame_" + str(idx_img).zfill(4) + ".png"
            im = plt.imread(imageFolder + frame_folder)
            im = np.floor((0.2126*im[:,:,0]+0.7152*im[:,:,1]+0.0722*im[:,:,2])*255)
            background += im
            imageList.append(im)
        background /= subsetSize
        
        temporalAvImages = temporalAverage(idxList,imageList,shape)
        for i in range(len(temporalAvImages)):
            image1 = temporalAvImages[i]
            image2 = temporalAvImages[0:i] + temporalAvImages[i+1::]
            meanImage = np.zeros(shape)
            for j in range(len(image2)):
                meanImage += image2[j]
            meanImage /= len(image2)
            mi = calc_MI(image1.flatten(), meanImage.flatten(), 256)
            psnr = cv2.PSNR(image1, meanImage)
            miScore.append(mi*cst)
            psnrScore.append(psnr)
        """
        #remove background for each frame and then, compute the metric's score
        for j in range(subsetSize):
            imageList[j] = (imageList[j] + 255 - background)//2
        """
        
    return miScore,psnrScore