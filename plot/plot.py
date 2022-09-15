import os
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : SMALL_SIZE}

ticks_x = [0, 25, 50, 75, 100]
ticks_y = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
hline_x = [-10,110]
interval_to_show_x = [-2,102]

rc('font', **font)
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

if __name__ == "__main__":
    
    CSV_path = 'E:/CSV/results_all.csv'
    export_path = 'E:/Figures/'
    
    selected_cameras = ['S05c016', 'S05c017', 'S05c019', 'S05c024']
    strategies = ['n_first', 'fixed_interval', 'flow_diff', 'flow_interval_mix']
    strategies_clean_name = ['N first', 'Fixed intervals', 'Movement detection', 'Movement detection + fixed intervals','yolov5n pretrained on COCO only','yolov5x pretrained on COCO only']
    
    Ns = ['25', '50', '75', '100']
    df = pd.read_csv(CSV_path, sep=',')
    df = df.loc[df['camera'].isin(selected_cameras)]
    dpi = 1200
    
    
    df_avg_camera_05 = pd.DataFrame(data = np.zeros((len(strategies), len(Ns)+1)),
                                    columns = ['0', '25', '50', '75', '100'],
                                    index = strategies)

    df_avg_camera_0595 = pd.DataFrame(data = np.zeros((len(strategies), len(Ns)+1)),
                                      columns = ['0', '25', '50', '75', '100'],
                                      index = strategies)
    avg_teacher_05 = 0
    avg_teacher_0595 = 0
    for camera in selected_cameras:
        df_camera = df.loc[df['camera'] == camera]
        df_camera = df_camera.set_index('name')
        plt.figure(f'{camera}_mAP05', dpi=dpi)
        plt.figure(f'{camera}_mAP0595', dpi=dpi)
        
        # Extract teacher for horizontal line in the plots
        teacher_05 = df_camera.loc['yolov5x', 'mAP 0.5 (640x640)']
        teacher_0595 = df_camera.loc['yolov5x', 'mAP 0.5:95 (640x640)']
        avg_teacher_05 += teacher_05
        avg_teacher_0595 += teacher_0595

        for strategy in strategies:
            # Add n = 0 in the table, i.e. YOLOv5n trained on COCO dataset
            x = [0]
            val_n0_05 = df_camera.loc['yolov5n', 'mAP 0.5 (640x640)']
            val_n0_0595 = df_camera.loc['yolov5n', 'mAP 0.5:95 (640x640)']
            
            y_05 = [val_n0_05]
            y_0595 = [val_n0_0595]
            df_avg_camera_05.at[strategy, '0'] += val_n0_05
            df_avg_camera_0595.at[strategy, '0'] += val_n0_0595
            
            for n in Ns:
                names = df_camera.index
                for name in names:
                    words = name.split('-')
                    if strategy in words and n in words:
                        x.append(int(n))
                        
                        val_05 = df_camera.loc[name, 'mAP 0.5 (640x640)']
                        val_0595 = df_camera.loc[name, 'mAP 0.5:95 (640x640)']
                        
                        y_05.append(val_05)
                        y_0595.append(val_0595)
                        
                        df_avg_camera_05.at[strategy, n] += val_05
                        df_avg_camera_0595.at[strategy, n] += val_0595

            plt.figure(f'{camera}_mAP05')
            plt.plot(np.asarray(x),np.asarray(y_05),'--x',linewidth = 1)
            
            plt.figure(f'{camera}_mAP0595')
            plt.plot(np.asarray(x),np.asarray(y_0595),'--x',linewidth = 1)
            

        # end for strategy

        plt.figure(f'{camera}_mAP05')
        plt.title(f'Strategies comparison for {camera} (mAP @0.5)', weight = 'bold')        
        plt.plot(x[0],y_05[0],'*k',linewidth=2, markersize=12)
        plt.plot(hline_x,[teacher_05,teacher_05],'--k')
        plt.xlabel('Number of selected frames')
        plt.ylabel('mAP @ 0.5')
        plt.legend(strategies_clean_name,loc = 'lower right')
        plt.xticks(ticks = ticks_x)
        plt.yticks(ticks = ticks_y)
        plt.xlim(interval_to_show_x)
        plt.grid()

        plt.figure(f'{camera}_mAP0595')
        plt.title(f'Strategies comparison for {camera} (mAP @0.5:95)', weight = 'bold')            
        plt.plot(x[0],y_0595[0],'*k',linewidth=2, markersize=12)
        plt.plot(hline_x,[teacher_0595,teacher_0595],'--k')
        plt.xlabel('Number of selected frames')
        plt.ylabel('mAP @ 0.5:95')
        plt.legend(strategies_clean_name,loc = 'lower right')
        plt.xticks(ticks = ticks_x)
        plt.yticks(ticks = ticks_y)
        plt.xlim(interval_to_show_x)
        plt.grid()

    # end for camera

    df_avg_camera_05 /= 4
    df_avg_camera_0595 /= 4
    avg_teacher_05 /= 4
    avg_teacher_0595 /= 4
    
    plt.figure('avg_mAP05', dpi=dpi)
    plt.figure('avg_mAP0595', dpi=dpi)
    
    for strategy in strategies:
        y_05 = df_avg_camera_05.loc[strategy,:].values
        y_0595 = df_avg_camera_0595.loc[strategy,:].values
        # print(y_05)
        # print(y_0595)
        plt.figure('avg_mAP05')
        plt.plot(np.asarray(ticks_x),np.asarray(y_05),'--x',linewidth = 1)
        
        plt.figure('avg_mAP0595')
        plt.plot(np.asarray(ticks_x),np.asarray(y_0595),'--x',linewidth = 1)
        
    
    plt.figure('avg_mAP05')
    plt.title('Strategies comparison averaged across cameras (mAP @0.5)', weight = 'bold')        
    plt.plot(x[0],y_05[0],'*k',linewidth=2, markersize=12)
    plt.plot(hline_x,[avg_teacher_05,avg_teacher_05],'--k')
    plt.xlabel('Number of selected frames')
    plt.ylabel('mAP @0.5')
    plt.legend(strategies_clean_name,loc = 'lower right')
    plt.xticks(ticks = ticks_x)
    plt.yticks(ticks = ticks_y)
    plt.xlim(interval_to_show_x)
    plt.grid()

    plt.figure('avg_mAP0595')
    plt.title('Strategies comparison averaged across cameras (mAP @0.5:95)', weight = 'bold')            
    plt.plot(x[0],y_0595[0],'*k',linewidth=2, markersize=12)
    plt.plot(hline_x,[avg_teacher_0595,avg_teacher_0595],'--k')
    plt.xlabel('Number of selected frames')
    plt.ylabel('mAP @0.5:95')
    plt.legend(strategies_clean_name,loc = 'lower right')
    plt.xticks(ticks = ticks_x)
    plt.yticks(ticks = ticks_y)
    plt.xlim(interval_to_show_x)
    plt.grid()
    
    for camera in selected_cameras:
        plt.figure(f'{camera}_mAP05')
        plt.savefig(fname = os.path.join(export_path, f'{camera}_mAP05' + '.png'))
        
        plt.figure(f'{camera}_mAP0595')
        plt.savefig(fname = os.path.join(export_path, f'{camera}_mAP0595' + '.png'))
        
    plt.figure('avg_mAP05')
    plt.savefig(fname = os.path.join(export_path, 'avg_mAP05' + '.png'))
    
    plt.figure('avg_mAP0595')
    plt.savefig(fname = os.path.join(export_path, 'avg_mAP0595' + '.png'))
