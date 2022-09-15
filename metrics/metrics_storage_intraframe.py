import pandas as pd
import os
import copy
import string


FrameFolder = "E:/RQ2/S05c016/Frames/"
FrameFolder_list = os.listdir(FrameFolder)

intra_MetricFolder = "E:/RQ2/S05c016/Metrics/Intraframe/"
intra_MetricFolder_list = os.listdir(intra_MetricFolder)

intra_Metric_list = []

for i in range(0,len(intra_MetricFolder_list)):
    with open(intra_MetricFolder + intra_MetricFolder_list[i],'r') as f:
        intra_Metric = f.readlines()
        intra_Metric = [float(k) for k in intra_Metric]
    intra_Metric_list.append(intra_Metric)


LogFolder = "E:/RQ2/S05c016/Log/"
LogFolder_list = os.listdir(LogFolder)

LogFolder_idxlist_list = []

for i in range(0,len(LogFolder_list)):
    with open(LogFolder + LogFolder_list[i],'r') as f:
        mylist = f.readlines()
    idxList = []
    for i in range(len(mylist)):
        line = mylist[i].split('/')
        line = line[-1].split('.')
        line = line[0].split('_')
        idx = line[-1]
        idxList.append(int(idx))
    
    idxList.sort()    
    LogFolder_idxlist_list.append(idxList)


intra_Metric_subsample_list = copy.deepcopy(LogFolder_idxlist_list)
intra_Metric_subsample_list_all = copy.deepcopy(intra_MetricFolder_list)

for m in range(0, len(intra_MetricFolder_list)):  
    for i in range(0, len(LogFolder_idxlist_list)):
        for j in range(0, len(LogFolder_idxlist_list[i])):
            intra_Metric_subsample_list[i][j]= copy.deepcopy(intra_Metric_list[m][LogFolder_idxlist_list[i][j]])
                
    intra_Metric_subsample_list_all[m]=copy.deepcopy(intra_Metric_subsample_list) 

    
intra_MetricFolder_list = [x[:-4] for x in intra_MetricFolder_list]

LogFolder_list_new = [x[:-4] for x in LogFolder_list]
LogFolder_list_extended = []
for i in range(0,len(intra_MetricFolder_list)):
    for j in range(0,len(LogFolder_list_new)):
        LogFolder_list_extended.append(LogFolder_list_new[j]+intra_MetricFolder_list[i])
    
col0 = 'Frame #'
frame_num = list(range(1, len(FrameFolder_list)+1)) 



df = pd.read_excel('RQ2 - 16.xlsx')
df[col0] = frame_num
for i in range(0,len(intra_MetricFolder_list)):
    df[intra_MetricFolder_list[i]] =  pd.Series(intra_Metric_list[i])  


#chipotage
for i in range(0,len(LogFolder_list_extended)):
    if(LogFolder_list_extended[i][-4:]=='-snr'):
        LogFolder_list_extended[i]='snr-'+LogFolder_list_extended[i][:-4]
    if(LogFolder_list_extended[i][-8:]=='-entropy'):
        LogFolder_list_extended[i]='entropy-'+LogFolder_list_extended[i][:-8]
    if(LogFolder_list_extended[i][-7:]=='-snr_bg'):
        LogFolder_list_extended[i]='snr_bg-'+LogFolder_list_extended[i][:-7]
    if(LogFolder_list_extended[i][-11:]=='-entropy_bg'):
        LogFolder_list_extended[i]='entropy_bg-'+LogFolder_list_extended[i][:-11]
        
        


for i in range(0,len(intra_MetricFolder_list)):
    for j in range(0, len(LogFolder_idxlist_list)):
        df[LogFolder_list_extended[len(LogFolder_idxlist_list)*i+j]] =  pd.Series(intra_Metric_subsample_list_all[i][j])  

# asci_string = string.ascii_uppercase
# asci = list(string.ascii_uppercase)
# asci_excel = copy.deepcopy(asci)
# for i in range(0,len(asci)):
#     for j in range(0,len(asci)):
#         asci_excel.append(asci[i]+asci[j])

# row_to_add=[]
    
# for i in range(0,len(asci_excel)):
#     row_to_add.append('=AVERAGE(' + asci_excel[i] + '2:' + asci_excel[i] + str(len(frame_num))+')')


# for i in range(0, len(row_to_add)):
#     df[asci_excel[i]+str(len(frame_num)+2)] = pd.Series(row_to_add[i])


        
df.to_excel('RQ2 - 16.xlsx', index = False)


