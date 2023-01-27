from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt

def stategyFrequency(image_path):
    image = Image.open(image_path)
    data = asarray(image)
    print("type", type(data))

    t = np.arange(256)
    sp = np.fft.fft(np.sin(t))
    freq = np.fft.fftfreq(t.shape[-1])
    plt.plot(freq, sp.real, freq, sp.imag)



