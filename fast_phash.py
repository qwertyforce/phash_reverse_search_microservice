import scipy.fft
import cv2
import numpy as np
from numba import jit

@jit(nopython=True)
def diff(dct, hash_size):
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff.flatten()

def fast_phash(image, hash_size=16, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)  #cv2.INTER_AREA
    dct = scipy.fft.dct(scipy.fft.dct(image, axis=0), axis=1)
    return diff(dct, hash_size)

def bool_list_to_bin_string(arr):
    return '0b' + ''.join(['1' if x else '0' for x in arr])

# query_image=cv2.imread("457.jpeg")
# query_image=cv2.cvtColor(query_image,cv2.COLOR_BGR2RGB)
# x=fast_phash(query_image)
# print(int(bool_list_to_bin_string(x),2))
