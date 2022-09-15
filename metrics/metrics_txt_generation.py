# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:39:43 2022

@author: grotsartdehe
"""

#for the script
import matplotlib.pyplot as plt 
import random
import time 
import numpy as np
import os
from TrailRQ2 import evaluationIntraFrame,evaluationInterFrame

def writeCSV(filename,cameraFolder,metric):
    with open("E:/Data_trail/metrics/"+cameraFolder+filename, 'w') as file_handler:
        for item in metric:
            file_handler.write("{}\n".format(item))
def main():
    startTime = time.time()
    cameraFolder = 'C024/'
    imageFolder = "E:/Data_trail/S05c024/bank/images/"
    frame_folder = "frame_" + str(0).zfill(4) + ".png"
    im = plt.imread(imageFolder + frame_folder)
    H,W,_ = im.shape
    intraframe = False
    if intraframe:
        lst = os.listdir(imageFolder) # your directory path
        number_files = len(lst)
        idxList = [i for i in range(0,number_files)]#[i for i in range(3730, 3730+30)]
        indexList = [idxList]#len corresponds to number of strategies
        
        metricsChoice = ['Entropy']
        
        individualEntropy,individualEntropy_bg,individualSNR,individualSNR_bg = evaluationIntraFrame(imageFolder,indexList,metricsChoice,(H,W))
        writeCSV("entropy.txt",cameraFolder, individualEntropy)
        writeCSV("entropy_bg.txt",cameraFolder, individualEntropy_bg)
        writeCSV("snr.txt",cameraFolder, individualSNR)
        writeCSV("snr_bg.txt",cameraFolder, individualSNR_bg)
    else:
        LogFolder = "E:/Data_trail/runs/"+cameraFolder
        LogFolder_list = os.listdir(LogFolder)
        
        for i in range(0,len(LogFolder_list)):
            print(LogFolder_list[i])
            with open(LogFolder + LogFolder_list[i],'r') as f:
                mylist = f.readlines()
            idxList = []
            log = LogFolder_list[i].split('.')[0]
            log_split = log.split('-')
            nbrFrames = log_split[2]
            strat = log_split[4]
            for j in range(len(mylist)):
                line = mylist[j].split('/')
                line = line[-1].split('.')
                line = line[0].split('_')
                idx = line[-1]
                idxList.append(int(idx))
    
            idxList.sort()   
            indexList = [idxList]
            metricsChoice = ['Entropy']
            miScore,psnrScore = evaluationInterFrame(imageFolder,indexList,metricsChoice,(H,W))
            writeCSV(LogFolder_list[i].split('.')[0]+'_mutual_information.txt',cameraFolder, miScore)
            writeCSV(LogFolder_list[i].split('.')[0]+'_psnr.txt',cameraFolder, psnrScore)
    
    endTime= time.time()
    print("Execution time: ", endTime - startTime)
    #return scores,miScore,psnrScore#indEntropy_mean,indSNR_mean,indEntropy_bg_mean,indSNR_bg_mean, mi_mean,psnr_mean
"""
nbrFrames = [10,25,50,75,100]
indEntropy_mean,indSNR_mean,indEntropy_bg_mean,indSNR_bg_mean,mi_mean,psnr_mean = main()
plt.figure()
plt.scatter(nbrFrames,indEntropy_mean)
plt.title("Entropy with background")

plt.figure()
plt.scatter(nbrFrames,indEntropy_bg_mean)
plt.title("Entropy without background")

plt.figure()
plt.scatter(nbrFrames,indSNR_mean)
plt.title("SNR with background")

plt.figure()
plt.scatter(nbrFrames,indSNR_bg_mean)
plt.title("SNR without background")

plt.figure()
plt.scatter(nbrFrames,mi_mean)
plt.title("Mutual information")

plt.figure()
plt.scatter(nbrFrames,psnr_mean)
plt.title("psnr")
"""
"""
scores,miScore,psnrScore = main()
withBG = scores[0]
indEntropy = withBG[0]
snr = withBG[1]
withoutBG = scores[1]
indEntropy_bg = withoutBG[0]
snr_bg = withoutBG[1]

plt.figure()
plt.plot(indEntropy)
plt.plot(indEntropy_bg)
plt.title("Entropy")
plt.legend(["With background", "Without background"])

plt.figure()
plt.plot(snr)
plt.plot(snr_bg)
plt.title("SNR")
plt.legend(["With background", "Without background"])

plt.figure()
plt.plot(miScore)
plt.title("Mutual information")

plt.figure()
plt.plot(psnrScore)
plt.title("Psnr")
"""
main()