import time

import cv2
""" GENERATING FEATURES FOR GLCM """
from MachineLearn.Classes.Extractors.GLCM import GLCM

def glcm():
    inicio = time.time()
    img = cv2.imread("lena.jpg", 0)

    oGlcm = GLCM(img, 8)
    oGlcm.generateCoOccurenceHorizontal()
    oGlcm.normalizeCoOccurence()
    oGlcm.calculateAttributes()
    fim = time.time()

    print("total", fim-inicio)

if __name__ == '__main__':
    import timeit
    print(timeit.timeit(glcm, number=2, ))