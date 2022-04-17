# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data


def main():
    pass


def fast_glcm(img, vmin=0, vmax=255, levels=8, kernel_size=5, distance=1.0, angle=0.0):
    '''
    Parameters
    ----------
    img: array_like, shape=(h,w), dtype=np.uint8
        input image
    vmin: int
        minimum value of input image
    vmax: int
        maximum value of input image
    levels: int
        number of grey-levels of GLCM
    kernel_size: int
        Patch size to calculate GLCM around the target pixel
    distance: float
        pixel pair distance offsets [pixel] (1.0, 2.0, and etc.)
    angle: float
        pixel pair angles [degree] (0.0, 30.0, 45.0, 90.0, and etc.)

    Returns
    -------
    Grey-level co-occurrence matrix for each pixels
    shape = (levels, levels, h, w)
    '''

    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, levels+1)
    gl1 = np.digitize(img, bins) - 1

    # make shifted image
    dx = distance*np.cos(np.deg2rad(angle))
    dy = distance*np.sin(np.deg2rad(-angle))
    mat = np.array([[1.0,0.0,-dx], [0.0,1.0,-dy]], dtype=np.float32)
    gl2 = cv2.warpAffine(gl1, mat, (w,h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REPLICATE)

    # make glcm
    glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm


def fast_glcm_mean(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm mean
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            mean += glcm[i,j] * i / (levels)**2

    return mean


def fast_glcm_std(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm std
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            mean += glcm[i,j] * i / (levels)**2

    std2 = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            std2 += (glcm[i,j] * i - mean)**2

    std = np.sqrt(std2)
    return std


def fast_glcm_contrast(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm contrast
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    cont = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            cont += glcm[i,j] * (i-j)**2

    return cont


def fast_glcm_dissimilarity(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm dissimilarity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    diss = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            diss += glcm[i,j] * np.abs(i-j)

    return diss


def fast_glcm_homogeneity(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm homogeneity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    homo = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            homo += glcm[i,j] / (1.+(i-j)**2)

    return homo


def fast_glcm_ASM(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm asm, energy
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            asm  += glcm[i,j]**2

    ene = np.sqrt(asm)
    return asm, ene


def fast_glcm_max(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm max
    '''
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    max_  = np.max(glcm, axis=(0,1))
    return max_


def fast_glcm_entropy(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    return ent


if __name__ == '__main__':
    main()

    levels = 8
    ks = 5
    mi, ma = 0, 255

    img = data.camera()
    h,w = img.shape

    img[:,:w//2] = img[:,:w//2]//2+127
    glcm_mean = fast_glcm_mean(img, mi, ma, levels, ks)
