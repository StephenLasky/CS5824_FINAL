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
        big_im = np.zeros((32*y, 32*ims_per_side,3))
    else:
        big_im = np.zeros((32*ims_per_side, 32*ims_per_side,3))

    x, y, i, = 0, 0, 0
    ims_left = num_ims
    while ims_left > 0:
        big_im[y:y+32, x:x+32] = im_vec_to_im_std(ims[i])

        i += 1
        x = (i) % (ims_per_side) * 32
        y = int(i / ims_per_side) * 32
        ims_left -= 1

    return big_im

