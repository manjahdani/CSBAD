"""

DEPRECATED FILE ???
same as subsampling/frequency_utils.py

"""

from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import copy

def highpassFilter(dataFourier, n):
    # high pass filter
    # we remove the 2*n x 2*n low frequencies
    # we remove the n first positive frequencies and the n first negative frequencies
    (W, H) = dataFourier.shape
    dataFourier[W//2 - n:W//2 + n + 1,H//2 - n:H//2 + n + 1] = 0
    return dataFourier

def removeRegion(dataFourier, R):
    (W, H) = dataFourier.shape
    regionList = []
    sumFrequency = []
    for i in range(R//2):
        for j in range(R//2):
            block = dataFourier[2*W*i//R:2*W*(i+1)//R, 2*H*j//R:2*H*(j+1)//R]
            regionList.append(block)
            sumFrequency.append(np.abs(np.real(block)).sum())

    regionList_removed = copy.deepcopy(regionList)#just to be able to show the split of the image
    regionList_removed[np.argmax(sumFrequency)] = np.zeros(block.shape)
    return regionList, regionList_removed
def Frequency(image_path, plotFig=False):
    data = imread(image_path) #Read one image
    data = rgb2gray(data)   #Convert to gray 
    (W, H) = data.shape #Retrieve the width and the height of the cameras
    data_fourier = np.fft.fftshift(np.fft.fft2(data)) #Computation of the fourier transform.
    data_fourier_copy = copy.deepcopy(data_fourier)
    data_fourier_filtered = highpassFilter(data_fourier_copy, n=25)
    data_filtered = np.fft.ifft2(np.fft.ifftshift(data_fourier_filtered)).real

    R = 4
    region_image, _ = removeRegion(data_filtered, R=R)#just to check if everything is OK in the method
    _, regionList_removed = removeRegion(data_fourier_filtered, R=4)#to remove the region
    data_fourier_removed = copy.deepcopy(data_fourier_filtered)
    count = 0
    for i in range(R // 2):
        for j in range(R // 2):
            data_fourier_removed[2 * W * i // R:2 * W * (i + 1) // R, 2 * H * j // R:2 * H * (j + 1) // R] = regionList_removed[count]
            count += 1
    image_removed = np.fft.ifft2(np.fft.ifftshift(data_fourier_removed)).real
    sumFrequency_original = data_fourier.sum()
    sumFrequency_removed = data_fourier_removed.sum()
    
    #print("Frequency original f = %2.3f" % np.absolute(sumFrequency_original))
    #print("Frequency removed  S = %2.3f" % np.absolute(sumFrequency_removed))
    if plotFig:
        fig, ax = plt.subplots(nrows=3, ncols=4)
        ax[0,0].imshow(data, cmap='gray')
        ax[0,0].set_title("Original image")

        ax[0,1].imshow(np.log(abs(data_fourier)), cmap='gray')
        ax[0,1].set_title("Fourier transform")

        ax[0,2].imshow((20*np.log10( 0.1 + data_fourier_filtered)).astype(int), cmap='gray')
        ax[0,2].set_title("Fourier transform after high pass filtered")

        ax[0,3].imshow(data_filtered, cmap='gray')
        ax[0,3].set_title("Image after high pass filter")

        ax[1,0].imshow(region_image[0], cmap='gray')
        ax[1,0].set_title('Region 1')

        ax[1,1].imshow(region_image[1], cmap='gray')
        ax[1,1].set_title('Region 2')

        ax[1,2].imshow(region_image[2], cmap='gray')
        ax[1,2].set_title('Region 3')

        ax[1,3].imshow(region_image[3], cmap='gray')
        ax[1,3].set_title('Region 4')

        ax[2,0].imshow((20*np.log10( 0.1 + data_fourier_removed)).astype(int), cmap='gray')
        ax[2,0].set_title("Fourier transform removed")

        ax[2,1].imshow(image_removed, cmap='gray')
        ax[2,1].set_title("Image removed")

        plt.show()
        
    return sumFrequency_original, sumFrequency_removed


