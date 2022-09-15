import pandas as pd
import os
import copy

FrameFolder = "E:/RQ2/S05c016/Frames/"
FrameFolder_list = os.listdir(FrameFolder)

inter_MetricFolder = "E:/RQ2/S05c016/Metrics/Interframe/"
inter_MetricFolder_list = os.listdir(inter_MetricFolder)


inter_Metric_list = []

for i in range(0,len(inter_MetricFolder_list)):
    with open(inter_MetricFolder + inter_MetricFolder_list[i],'r') as f:
        inter_Metric = f.readlines()
        inter_Metric = [float(k) for k in inter_Metric]
    inter_Metric_list.append(inter_Metric)


inter_MetricFolder_list = [x[:-4] for x in inter_MetricFolder_list]


col0 = 'Frame #'
frame_num = list(range(1, len(FrameFolder_list)+1))



#chipotage
for i in range(0,len(inter_MetricFolder_list)):
    if(inter_MetricFolder_list[i][-19:]=='-mutual_information'):
        inter_MetricFolder_list[i]='mutual_information-'+inter_MetricFolder_list[i][:-19]
    if(inter_MetricFolder_list[i][-5:]=='-psnr'):
        inter_MetricFolder_list[i]='psnr-'+inter_MetricFolder_list[i][:-5]

                
        
        

df = pd.read_excel('RQ2 - 16.xlsx')
df[col0] = frame_num
for i in range(0,len(inter_MetricFolder_list)):
    df[inter_MetricFolder_list[i]] =  pd.Series(inter_Metric_list[i]) 


        
df.to_excel('RQ2 - 16.xlsx', index = False)


