from __future__ import print_function
import multiprocessing
import numpy as np
import time


def hard_code(unused_input):
    n = 10
    for i in range(n):
        x = np.random.random(size=(1000000,))
        x.sort()
    return None


def single_thread(n_threads):
    for i in range(n_threads):
        hard_code(None)


def multi_thread(n_threads):
    p = multiprocessing.Pool(n_threads)
    p.map(hard_code, range(n_threads))


def test_both(n_threads):
    time0 = time.time()
    single_thread(n_threads)
    time1 = time.time()
    multi_thread(n_threads)
    time2 = time.time()

    print('Single thread took {} seconds'.format(time1 - time0))
    print('Multi thread took {} seconds'.format(time2 - time1))
