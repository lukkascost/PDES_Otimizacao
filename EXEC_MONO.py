from multiprocessing import Pool, cpu_count
import time
import numpy as np
import cv2

""" GENERATING FEATURES FOR GLCM """
from MachineLearn.Classes.Extractors.GLCM import GLCM


def normalize_CC(coOccurenceMatrix, total):
    for i in range(coOccurenceMatrix.shape[0]):
        for j in range(coOccurenceMatrix.shape[1]):
            coOccurenceMatrix[i, j] = coOccurenceMatrix[i, j] / total
    return coOccurenceMatrix


def partial_CC(data):
    img = data[0]
    ini = data[1]
    fim = data[2]
    coOccurenceMatrix = np.zeros((256, 256))
    for i in range(ini, fim):
        for j in range(0, img.shape[1] - 1):
            coOccurenceMatrix[int(img[i, j]), int(img[i, j + 1])] += 1
    return coOccurenceMatrix


def partial_ATT(data):
    coOccurenceNormalized = data[0]
    ini = data[1]
    fim = data[2]
    glcm_atributes = np.zeros(25)
    for i in range(ini, fim):
        for j in range(coOccurenceNormalized.shape[1]):
            ij = coOccurenceNormalized[i, j]
            glcm_atributes[1] += ij * ij
            glcm_atributes[2] += ((i - j) * (i - j) * (ij))
            glcm_atributes[5] += (ij) / (1 + ((i - j) * (i - j)))
            glcm_atributes[9] += ij * np.log10(ij + 1e-30)
            glcm_atributes[15] += (ij) / (1 + abs(i - j))
            glcm_atributes[16] += ij * (i + j)
            glcm_atributes[21] += ij * abs(i - j)
            glcm_atributes[22] += ij * (i - j)
            glcm_atributes[23] += ij * i * j
    glcm_atributes[17] = np.amax(coOccurenceNormalized)
    glcm_atributes[16] /= 2
    glcm_atributes[22] /= 2
    glcm_atributes[9] *= -1
    return glcm_atributes


def glcm(core):
    img = cv2.imread("lena.jpg", 0)
    imgs = []
    ccs = []
    pool = Pool(processes=core)
    print(core)
    inicio = time.time()
    for i in range(0, core):
        imgs.append([img, int((img.shape[0]) / core * i), int((img.shape[0]) / core * (i + 1))])
    coOccurenceMatrix = np.sum(np.array(pool.map(partial_CC, imgs)), axis=0)
    total = img.shape[0] * (img.shape[1] - 1)
    coOccurenceMatrix = normalize_CC(coOccurenceMatrix, total)
    for i in range(0, core):
        ccs.append([coOccurenceMatrix, int((coOccurenceMatrix.shape[0]) / core * i),
                    int((coOccurenceMatrix.shape[0]) / core * (i + 1))])
    atts = np.sum(np.array(pool.map(partial_ATT, ccs)), axis=0)
    atts[17] = np.amax(coOccurenceMatrix)
    fim = time.time()
    pool.terminate()

    print("total", fim - inicio)


def n_glcm():
    img = cv2.imread("lena.jpg", 0)
    inicio = time.time()
    coOccurenceMatrix = partial_CC([img,0, img.shape[0]])
    total = img.shape[0] * (img.shape[1] - 1)
    coOccurenceMatrix = normalize_CC(coOccurenceMatrix, total)
    atts = partial_ATT([coOccurenceMatrix, 0, coOccurenceMatrix.shape[1]])
    fim = time.time()
    print("total", fim - inicio)


if __name__ == '__main__':
    import timeit
    print(timeit.timeit(n_glcm, number=10, ))
