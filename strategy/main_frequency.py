import argparse
import os
from subsampling import *
import utils 
from strategyFrequency import *
import pandas as pd 

if __name__ == "__main__":
    outfolder = "E:/data/WALT-challenge/cam2/week5/bank/images/"
    os.chdir(outfolder)
    dic = {}
    count1 = 0
    for file in glob.glob("*.jpg"):
        print(str(count1)+  "- Processing image " + str(file) + outfolder)
        sumFrequency_original, sumFrequency_removed = Frequency(image_path=outfolder+file)
        dic[str(file).replace(".jpg","")] = [np.absolute(sumFrequency_original), np.absolute(sumFrequency_removed)]
        count1=count1+1
    sumFrequencyDataset = pd.DataFrame.from_dict(dic,orient='index', columns=['frequency','frequency_filtered'])
    sumFrequencyDataset.to_csv('../frequencies.txt', sep='\t')
    #outFolder = "E:/data/frequency/images/"
    #imgToAnalyse = '254_1552506849.jpg'
    #Frequency(image_path = outFolder+imgToAnalyse, plotFig= True)
    #os.ch
    #sum2 = {}
    #count1 = 0
    #for file in glob.glob("*.jpg"):
        #if count1 < 3:
            #print("Processing image "+str(count1))
            #sumFrequency_original, sumFrequency_removed = Frequency(image_path=outfolder+file)
            #count1=count1+1
            #sum1[file] = sumFrequency_original
            #sum2[file] = sumFrequency_removed
    
    #sortedOriginal = sorted(sum1)
    #print(sortedOriginal)
