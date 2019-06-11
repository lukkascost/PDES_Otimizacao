from __future__ import print_function

import numpy as np

""" GENERATING FEATURES FOR GLCM """

ex_array = np.zeros((100,), dtype='uint8')
for index in range(0, 100):
    ex_array[index] = 1
    print('Yay! {} done!'.format(index))

if __name__ == '__main__':
    import pymp
    ex_array = pymp.shared.array((100,), dtype='uint8')
    with pymp.Parallel(4) as p:
        for index in p.range(0, 100):
            ex_array[index] = 1
            # The parallel print function takes care of asynchronous output.
            p.print('Yay! {} done!'.format(index))
