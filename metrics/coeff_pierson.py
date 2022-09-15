# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:01:45 2022

@author: grotsartdehe
"""
import numpy as np 
import pandas as pd 
from more_itertools import locate
from scipy.stats import pearsonr

def coeff(metricCam,metricName,score,map_score):
    pos = list(locate(metricCam, lambda x: x == metricName))
    scoreList = []
    map_scoreList = []
    for pos_id in pos:
        scoreList.append(score[pos_id])
        map_scoreList.append(map_score[pos_id])
    return pearsonr(scoreList,map_scoreList)[0]

data = pd.read_csv("E:/toGauthier.csv",sep=';',decimal=',')
metricList = ['entropy','entropy_bg','snr','snr_bg','mutual_information','psnr']
mean = list(map(float,data["Mean"]))
median = list(map(float,data["Median"]))
metric = list(map(str,data["RQ2_met"]))
cameraID = list(data["video"])
map05 = list(map(float,data["mAP 0.5 (640x640)"]))
map95 = list(map(float,data["mAP 0.5:95 (640x640)"]))
cam_ID = []

for i in range(len(cameraID)):
    cam_ID.append(int(cameraID[i].split('c')[1]))
camList = [16,17,19]
coeffList = []
for cam in camList:
    nbDataCam = cam_ID.count(cam)
    dataCam = mean[0:nbDataCam]
    metricCam = metric[0:nbDataCam] 
    for m in metricList:
        coeffList.append(coeff(metricCam,m,dataCam,map95))
    del mean[0:nbDataCam]
    del metric[0:nbDataCam] 
    del map95[0:nbDataCam]
coeffPierson = np.reshape(coeffList,(len(camList),len(metricList)))



    

    


