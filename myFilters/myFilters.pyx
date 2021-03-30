import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] max_blur(np.ndarray[DTYPE_t, ndim=2] image, int size, int mask_size):
    # set the variable extension types
    cdef int x, y, w, h
    cdef DTYPE_t max_value
    cdef int i, j

    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] destination = np.zeros([h, w], dtype=DTYPE)
    size = int(size/2)
    mask_size = int(mask_size/2)

    # loop over the image
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            max_value = 0
            for i in range(-size, size+1):
                for j in range(-size, size+1):
                    if -mask_size <= i <= mask_size and -mask_size <= j <= mask_size: continue
                    if y+i >= h or y+i < 0 or x+j >= w or x+j < 0: continue
                    if max_value < image[y+i, x+j]:
                        max_value = image[y+i, x+j]
            destination[y, x] = max_value
    
    return destination

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] max_blur_corners(np.ndarray[DTYPE_t, ndim=2] image, int size):
    # set the variable extension types
    cdef int x, y, w, h
    cdef DTYPE_t max_value
    cdef int i, j

    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] destination = np.zeros([h, w], dtype=DTYPE)
    size = int(size/2)

    # loop over the image
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            max_value = 0
            for i in [-size,size]:
                for j in [-size,size]:
                    if y+i >= h or y+i < 0 or x+j >= w or x+j < 0: continue
                    if max_value < image[y+i, x+j]:
                        max_value = image[y+i, x+j]
            destination[y, x] = max_value
    
    return destination

