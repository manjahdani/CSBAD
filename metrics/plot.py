# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 18:48:18 2022

@author: grotsartdehe
"""
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import os

cameraFolder = 'C024/'
intraFrameFolder = ['entropy','entropy_bg','snr','snr_bg']
dataFolder = 'E:/Data_trail/metrics/' + cameraFolder
intraFrameFigure = False

if intraFrameFigure:
    for intraFrame in intraFrameFolder:
        dataFolder += 'intraFrame/' + intraFrame + '.txt'
        data = pd.read_csv(dataFolder,header=None)[0].tolist()
        time = np.arange(0,len(data))
        
        strategyFolder = os.listdir('E:/Data_trail/plot/'+cameraFolder)
        nbFrame = ["/25/","/50/","/75/","/100/"]
        color = ['b','g','tab:orange','r','tab:pink']
        color = color[0:len(strategyFolder)]
        results = np.zeros((len(strategyFolder),len(nbFrame)))
        for nbFrame_idx in range(len(nbFrame)):
            plt.figure(figsize=(15,10))
            for strategy in range(len(strategyFolder)):
                LogFolder = "E:/Data_trail/plot/" + cameraFolder + strategyFolder[strategy] + nbFrame[nbFrame_idx]
                LogFolder_list = os.listdir(LogFolder)
                for i in range(0,len(LogFolder_list)):
                    print(LogFolder_list[i])
                    cond = [False for i in range(len(data))]
                    value = []
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
                    for j in range(len(idxList)):
                        cond[idxList[j]] = True
                        value.append(data[idxList[j]])
                    print(np.median(value))
                
                    median = np.median(value)
                    results[strategy,nbFrame_idx] = median
                    col = np.where(cond, color[strategy],'k')
                    s = np.where(cond,50,1)
                    plt.scatter(time,data,c=col,s=s)
                    plt.title(intraFrame+" with " + nbFrame[nbFrame_idx].split('/')[1] + " frames")
                    plt.legend(strategyFolder)
                    a = 6
                    
                    plt.text(500, a-0.1*strategy, strategyFolder[strategy].split('/')[0] + ": " + str(median))
        
            plt.savefig('E:/Data_trail/figure/'+cameraFolder+intraFrame+'/FrameChosen_'+nbFrame[nbFrame_idx].split('/')[1]+'.png')
        frames = [25,50,75,100]
        plt.figure(figsize=(15,10))
        for i in range(len(strategyFolder)):
            plt.scatter(frames, (results[i,:]))
        plt.title('Strategy comparison for '+ intraFrame)
        plt.legend(strategyFolder)
        plt.savefig('E:/Data_trail/figure/'+cameraFolder+intraFrame+'/StrategyComparison_'+nbFrame[nbFrame_idx].split('/')[1]+'.png')
else:
    dataFolder += 'interFrame/'
    interFrameFolder = os.listdir(dataFolder)
    nbFrames = [25,50,75,100]
    strategyFolder = ["fixed_interval","flow_diff","flow_interval_mix","n_first"]#os.listdir('E:/Data_trail/plot/'+cameraFolder)
    res_mi = np.zeros((len(strategyFolder),len(nbFrames)))
    res_psnr = np.zeros((len(strategyFolder),len(nbFrames)))
    for file in interFrameFolder:
        file_split = (file.split('.')[0]).split('-')
        nb_frame = int(file_split[2])
        strategy = file_split[4]
        metric = file_split[-1]#le nom de la metric est a la fin du fichier
        with open('E:/Data_trail/metrics/'+cameraFolder+'interFrame/' + file,'r') as f:
            data = list(map(float, f.readlines()))
            data = np.mean(data)
        if metric == 'mutual_information':
            res_mi[strategyFolder.index(strategy),nbFrames.index(nb_frame)] = data
        else:
            res_psnr[strategyFolder.index(strategy),nbFrames.index(nb_frame)] = data
    plt.figure()   
    for i in range(len(strategyFolder)):
        plt.scatter(nbFrames, (res_mi[i,:]))
    plt.title('Mutual information')
    plt.legend(strategyFolder)
    plt.savefig('E:/Data_trail/figure/'+cameraFolder)
    plt.figure()   
    for i in range(len(strategyFolder)):
        plt.scatter(nbFrames, (res_psnr[i,:]))
    plt.title('PSNR')
    plt.legend(strategyFolder)
    plt.savefig('E:/Data_trail/figure/'+cameraFolder)
     