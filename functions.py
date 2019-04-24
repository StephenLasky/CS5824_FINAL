# imports here
import numpy as np
import math

# used for opening the cifar-10 data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# returns a numpy matrix with the first 'number' of images (CIFAR-10)
# currently only supports multiples of 10000
def open_data_ims(number):
    IMG_SIZE = 3072
    IMS_IN_BATCH = 10000
    batch_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    batch_files_directory = "./data/cifar-10-batches-py/"

    # will store ims such that each row represents one image
    ims = np.zeros((number, IMG_SIZE))

    im_num = 0
    for batch_file in batch_files:
        data_dict = unpickle(batch_files_directory + batch_file)
        im_batch = data_dict[b'data']

        if im_num >= number:
            break

        ims[im_num:im_num+IMS_IN_BATCH,:] = im_batch
        im_num += IMS_IN_BATCH

    return ims

# converts an image vector into a standard image
def im_vec_to_im_std(im_vec):
    IM_WIDTH = 32
    IM_HEIGHT = 32
    COLOR_OFFSET = IM_HEIGHT * IM_WIDTH

    im_std = np.zeros((IM_HEIGHT,IM_WIDTH,3))

    for x in range(0,IM_WIDTH):
        for y in range(0,IM_HEIGHT):
            # do for all 3 colors
            im_std[x, y, 0] = im_vec[x * IM_WIDTH + y]
            im_std[x, y, 1] = im_vec[x * IM_WIDTH + y + COLOR_OFFSET]
            im_std[x, y, 2] = im_vec[x * IM_WIDTH + y + 2 * COLOR_OFFSET]

    return im_std

def im_vec_to_im_std(im_vec, im_width, im_height):
    COLOR_OFFSET = im_height * im_width

    im_std = np.zeros((im_height, im_width, 3))

    for x in range(0, im_width):
        for y in range(0, im_height):
            # do for all 3 colors
            base = y * im_width + x
            im_std[y, x, 0] = im_vec[base]
            im_std[y, x, 1] = im_vec[base + COLOR_OFFSET]
            im_std[y, x, 2] = im_vec[base + 2 * COLOR_OFFSET]

    return im_std

# shows an image to the screen
def show_im_std(im):
    from matplotlib import pyplot as plt
    im /= 255

    plt.imshow(im, interpolation='nearest')
    plt.show()

def show_im_vec(im):
    im = im_vec_to_im_std(im)
    show_im_std(im)

# combines vec ims into a large combined std_im
def comb_ims(ims, im_width, im_height):
    num_ims = ims.shape[0]

    ims_per_side = int(math.floor( math.sqrt(num_ims) ))
    if num_ims - ims_per_side ** 2 > 0:
        y = ims_per_side + 1
        while y * ims_per_side < num_ims:
            y += 1
        big_im = np.zeros((im_height*y, im_width*ims_per_side,3))
    else:
        big_im = np.zeros((im_height*ims_per_side, im_width*ims_per_side,3))

    x, y, i, = 0, 0, 0
    ims_left = num_ims
    while ims_left > 0:
        big_im[y:y+im_height, x:x+im_width] = im_vec_to_im_std(ims[i], im_width, im_height)

        i += 1
        x = (i) % (ims_per_side) * im_width
        y = int(i / ims_per_side) * im_height
        ims_left -= 1

    return big_im

# TODO: write THIS function next!
# for now, we write this function so that it splits images perfectly horizontally!
# input expectation: image vector of dim [num_ims x 3072]
def seperate_xy_ims(ims):
    num_ims = ims.shape[0]

    OFFSET = 1024
    HALF_OFFSET = int(OFFSET / 2)

    x = np.zeros((num_ims, 3 * HALF_OFFSET))
    y = np.zeros((num_ims, 3 * HALF_OFFSET))

    for i in range(0,num_ims):

        for j in range(0,3):
            xs = j * OFFSET
            xe = j * OFFSET + HALF_OFFSET
            ys = j * OFFSET + HALF_OFFSET
            ye = j * OFFSET + OFFSET

            s = j * HALF_OFFSET
            e = (j+1) * HALF_OFFSET

            x[i, s:e] = ims[i, xs:xe]
            y[i, s:e] = ims[i, ys:ye]

    return x,y

# for now, this function takes the top part of an image (x) and merges it into the bottom image (y)
# these are taken and processed in vector form (im_vec)
def combine_xy_ims(x, y):
    assert x.shape[0] == y.shape[0] # ensure we have the same number of images

    num_ims = x.shape[0]
    x_dim = x.shape[1]
    y_dim = y.shape[1]
    f_dim = x_dim + y_dim

    x_color_dim = int(x_dim / 3)
    y_color_dim = int(y_dim / 3)
    f_color_dim = int(f_dim / 3)

    final_ims = np.zeros((num_ims, x_dim+y_dim))

    for i in range(0,num_ims):
        # for each color
        for j in range(0,3):
            xs = j * x_color_dim
            xe = (j+1) * x_color_dim
            ys = j * y_color_dim
            ye = (j+1) * y_color_dim

            fxs = j * f_color_dim
            fxe = fxs + x_color_dim
            fys = fxe
            fye = fxs + f_color_dim

            final_ims[i,fxs:fxe] = x[i,xs:xe]
            final_ims[i,fys:fye] = y[i,ys:ye]

    return final_ims