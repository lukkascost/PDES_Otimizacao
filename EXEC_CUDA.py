import multiprocessing
import time
import numpy as np
import cv2
""" GENERATING FEATURES FOR GLCM """
from MachineLearn.Classes.Extractors.GLCM import GLCM

def glcm():
    inicio = time.time()
    img = cv2.imread("lena.jpg", 0)
    imgs = np.split(img, 5)
    procs = []
    glcms = []
    for index, img_ in enumerate(imgs):
        glcm = GLCM(img_, 8)
        proc = multiprocessing.Process(target=glcm.generateCoOccurenceHorizontal, args=())
        procs.append(proc)
        proc.start()
        glcms.append(glcm)
    oGlcm = GLCM(img, 8)
    for i, proc in enumerate(procs):
        proc.join()
        oGlcm.coOccurenceMatrix += glcms[i].coOccurenceMatrix
    print(multiprocessing.cpu_count())
    oGlcm.generateCoOccurenceHorizontal()
    oGlcm.normalizeCoOccurence()
    oGlcm.calculateAttributes()
    fim = time.time()

    print("total", fim-inicio)

if __name__ == '__main__':
    import timeit
    print(timeit.timeit(glcm, number=2, ))

