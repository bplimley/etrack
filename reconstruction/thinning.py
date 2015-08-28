import numpy as np
import ipdb as pdb

def thin(image,n_iterations=np.inf):
    """
    Replicate MATLAB Image Processing Toolbox bwmorph(image, 'thin', inf).

    See codes/lawrencium/lrc_codes_may2013/f/bwmorph.m

    See paper: Guo 1989, "Parallel thinning with two-subiteration algorithms"
    """

    image = np.array(image)
    current_img = image
    new_img = current_img
    img_size = np.shape(image)

    n=0
    while n < n_iterations:
        new_img = thin_iteration(new_img, img_size)
        if np.array_equal(current_img, new_img):
            # nothing changed
            break
        current_img = new_img
        n+=1

    return new_img

def thin_iteration(image, img_size):
    """
    """

    def G1(img, x):
        """Condition G1 of Guo 1989 A2

        G1: X_H = 1
          where
        X_H = sum(i=1:4)(b_i)
        b_i = (not x_(2i-1)) and (x_(2i) or x_(2i+1))
        """

        X_H = np.zeros(np.shape(img))
        b = np.zeros(np.shape(img)+(5,))    # 1-indexed

        for i in xrange(1,5):
            b[...,i] = np.logical_and(
                np.logical_not(x[2*i-1]),
                np.logical_or(x[2*i], x[2*i+1]))
        X_H = np.sum(b,2)

        return X_H == 1

    def G2(img, x):
        """Condition G2 of Guo 1989 A2

        G2: 2 <= min(n1,n2) <= 3
          where
        n1 = sum(i=1:4)(x_(2i-1) || x_(2i))
        n2 = sum(i=1:4)(x_(2i) || x_(2i+1))

        """

        temp1 = np.zeros(np.shape(img) + (5,))  # 1-indexed
        temp2 = np.zeros(np.shape(img) + (5,))  # 1-indexed
        for i in xrange(1,5):
            temp1[...,i] = np.logical_or(x[2*i-1], x[2*i])
            temp2[...,i] = np.logical_or(x[2*i], x[2*i+1])
        n1 = np.sum(temp1,2)
        n2 = np.sum(temp2,2)

        return np.logical_and(
            np.minimum(n1,n2) >= 2,
            np.minimum(n1,n2) <= 3)

    def G3a(img, x):
        """Condition G3a of Guo 1989 A2

        G3a: (x_2 | x_3 | ~x_8) & x_1 == false
        """

        return np.logical_not(
            np.logical_and(
                np.logical_or(
                    np.logical_or(x[2],x[3]),
                    np.logical_not(x[8])),
                x[1]))

    def G3b(img,x):
        """Condition G3b of Guo 1989 A2

        G3b: (x_6 | x_7 | ~x_4) & x_5 == false
        """

        return np.logical_not(
            np.logical_and(
                np.logical_or(
                    np.logical_or(x[6],x[7]),
                    np.logical_not(x[4])),
                x[5]))

    # generate neighbor matrix, x, in order to perform all comparisons
    #   matrix-wide instead of looping through elements and doing i+1's
    #   everywhere.
    # (x is its name in MATLAB docs)
    x = [[] for __ in xrange(10)]
    # use 1-indexing to match MATLAB, also wrap around x[9]==x[1]
    #
    # These are defined as in MATLAB doc, different than Guo's paper.
    # www.mathworks.com/help/images/ref/bwmorph.html
    #    ?s_tid=srchtitle#bui7chk-1
    #
    # x1 = east neighbor
    # x2 = northeast neighbor, etc., CCW.
    #
    # North = negative x.
    # East = positive y.
    #   in other words, origin at top left, like MATLAB variable viewer.

    # buffer with false
    temp_image = np.zeros(np.array(img_size)+(2,2))
    temp_image[1:-1,1:-1] = image

    x[1] = temp_image[1:-1, 2:]     # E neighbor: shift in -y direction
    x[2] = temp_image[:-2,  2:]     # NE: shift +x, -y
    x[3] = temp_image[:-2,  1:-1]   # N
    x[4] = temp_image[:-2,  :-2]    # NW
    x[5] = temp_image[1:-1, :-2]    # W
    x[6] = temp_image[2:,   :-2]    # SW
    x[7] = temp_image[2:,   1:-1]   # S
    x[8] = temp_image[2:,   2:]     # SE
    x[9] = x[1]

    # pdb.set_trace()
    # first subiteration (uses G3a)
    delete_pixels = np.logical_and(
        G1(image, x), np.logical_and(G2(image, x), G3a(image, x)))
    image[delete_pixels] = 0

    # second subiteration (uses G3b)
    delete_pixels = np.logical_and(
        G1(image, x), np.logical_and(G2(image, x), G3b(image, x)))
    image[delete_pixels] = 0

    return image

if __name__ == "__main__":
    """
    Run test of thin, based on MATLAB's exact output.
    """

    # this is a track:
    # Mat_CCD_SPEC_500k_49_TRK_truncated.mat, Event 8, >0.5

    input_array = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,1,0],
        [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0],
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    # MATLAB>> matlab_output_array = bwmorph(input_array,'thin',inf);
    matlab_output_array = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0],
        [0,0,0,0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    python_output_array = thin(input_array,n_iterations=np.inf)

    if np.all(python_output_array == matlab_output_array):
        print("'thin' passes test")
    else:
        print("*** 'thin' fails test!")
