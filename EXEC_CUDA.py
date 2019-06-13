from multiprocessing import Pool, cpu_count
import time
import numpy as np
import cv2

""" GENERATING FEATURES FOR GLCM """
from MachineLearn.Classes.Extractors.GLCM import GLCM


def partial_lbp(data):
    atts = np.zeros(256)
    img = data[0]
    ini = data[1]
    fim = data[2]
    if (ini == 0): ini = 1
    for i in range(ini, fim):
        for j in range(1, img.shape[1] - 1):
            central = img[i, j]
            p11 = int(img[i - 1, j - 1] < central)
            p12 = int(img[i - 1, j] < central)
            p13 = int(img[i - 1, j + 1] < central)
            p21 = int(img[i, j - 1] < central)
            p23 = int(img[i, j + 1] < central)
            p31 = int(img[i + 1, j - 1] < central)
            p32 = int(img[i + 1, j] < central)
            p33 = int(img[i + 1, j + 1] < central)
            output = p11 + p21 * 2 + p31 * 4 + p32 * 8 + p33 * 16 + p23 * 32 + p13 * 64 + p12 * 128
            atts[output] += 1
    return atts


def lbp(img):
    atts = np.zeros(256)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            central = img[i, j]
            p11 = int(img[i - 1, j - 1] < central)
            p12 = int(img[i - 1, j] < central)
            p13 = int(img[i - 1, j + 1] < central)
            p21 = int(img[i, j - 1] < central)
            p23 = int(img[i, j + 1] < central)
            p31 = int(img[i + 1, j - 1] < central)
            p32 = int(img[i + 1, j] < central)
            p33 = int(img[i + 1, j + 1] < central)
            output = p11 + p21 * 2 + p31 * 4 + p32 * 8 + p33 * 16 + p23 * 32 + p13 * 64 + p12 * 128
            atts[output] += 1
    return atts


def glcm(core):
    img = cv2.imread("lena.jpg", 0)
    imgs = []
    pool = Pool(processes=core)
    print(core)
    inicio = time.time()
    for i in range(0, core):
        imgs.append([img, int((img.shape[0] - 1) / core * i), int((img.shape[0] - 1) / core * (i + 1))])
    result = np.sum(np.array(pool.map(partial_lbp, imgs)), axis=0)
    # result2 = lbp(img)
    pool.terminate()
    fim = time.time()

    print("total", fim - inicio)


def n_glcm():
    img = cv2.imread("lena.jpg", 0)
    inicio = time.time()
    result = lbp(img)
    fim = time.time()

    print("total", fim - inicio)

if __name__ == '__main__':
    import timeit

    print(timeit.timeit(n_glcm, number=10, ))
