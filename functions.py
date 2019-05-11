# imports here
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

# take a 3/4d numpy array and convert it to and save it as a video sequence
# we expect the following format: num_frames x height x width x color_depth
def save_video(frames, video_filename="video"):
    num_frames, height, width = frames.shape

    # forcibly expand save_video such that it contains three color channels
    new_v = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    for i in range(0,3): new_v[:,:,:,i] = frames
    frames = new_v

    # trial
    # frames[:,0,1:32,0] = 255

    # UPSCALE HERE
    f = 30
    new_w, new_h = width *f, height * f
    new_v = np.zeros((num_frames, new_h, new_w,3), dtype=np.uint8)
    for frame in range(0,num_frames):
        new_v[frame] = cv2.resize(frames[frame], dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)
    frames = new_v
    width = new_w
    height = new_h

    # new_v = np.zeros((num_frames,height, width), dtype=np.uint8)
    # new_v[:,:,:] = frames

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 30
    video_filename += '.avi'
    video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for frame in range(0,num_frames):
        video.write(frames[frame])

    video.release()

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
def show_im_std(im, title="No title"):
    im /= 255

    plt.imshow(im, interpolation='nearest')
    plt.title(label=title)
    plt.show()

def show_im_vec(im, title="No title"):
    im = im_vec_to_im_std(im)
    show_im_std(im, title)

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

# combines std ims into a large combined std_im
def comb_ims_std(ims):
    # ims /= 255.0    # regularize?

    num = ims.shape[0]
    h = ims.shape[1]
    w = ims.shape[2]

    num_rows = 1
    num_cols = num

    # determine the number of rows and columns to get approximately a square
    while num_rows * w < num_cols * h:
        num_rows += 1
        num_cols = num / num_rows

    num_cols = int(math.ceil(num_cols))

    # define the canvas that they will be painted onto
    big_im = np.zeros((num_cols * h, num_rows * w, 3), dtype = ims.dtype)

    # paint the canvas
    for i in range(0,num):
        ix = int(i/num_rows) * w
        iy = (i % num_cols) * h

        big_im[iy:iy+h, ix:ix+w] = ims[i]

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

# x: in STD form
# y: in STD form
def combine_xy_ims_std(x,y):

    num = x.shape[0]
    imhx = x.shape[1]
    imwx = x.shape[2]
    imhy = y.shape[1]
    imwy = y.shape[2]
    channels = x.shape[3]

    combined = np.zeros((num, imhx+imhy, imwx, channels), dtype=x.dtype)

    for i in range(0,num):
        combined[i,0:imhx] = x[i]
        combined[i,imhx:] = y[i]

    return combined



# this is used to generate the custom data
def generate_custom_data(num_x_rows, num_y_rows, data, width, height):
    num_ims = data.shape[0]
    num_data_per_im = (height - (num_x_rows+num_y_rows) + 1)
    num_data_generated = num_ims * num_data_per_im
    NUM_COLOR_CHANNELS = 3

    x = np.zeros((num_data_generated, width * num_x_rows * NUM_COLOR_CHANNELS))
    y = np.zeros((num_data_generated, width * num_y_rows * NUM_COLOR_CHANNELS))

    i = 0                       # used to keep track of which example we are currently on
    x_size = num_x_rows * width
    y_size = num_y_rows * width
    d_size = width * height
    for im in range(0,num_ims):                          # for each image
        for j in range(0, num_data_per_im ):             # for each new example
            r_offset = j * width
            xs = r_offset
            xe = r_offset+x_size
            ys = xe
            ye = ys + y_size

            for c in range(0,3): # DO THE COLORS
                x[i, c*x_size:(c+1)*x_size] = data[im, xs:xe]
                y[i, c*y_size:(c+1)*y_size] = data[im, ys:ye]

                xs += d_size
                xe += d_size
                ys += d_size
                ye += d_size

            i += 1                                      # increment the example that we are on

    assert i == num_data_generated

    return x,y

# extracts vector rows from a vector image and returns a vector image
# rs : row start
# re : row end (noninclusive)
# rw : row width
# im : vectorized image rows are being pulled from
def extract_vec_rows(rs, re, rw, im):
    COLOR_CHANNELS = 3
    # re-rs should be fine because re is exclusive
    ret_im = np.zeros(((re-rs)*rw*COLOR_CHANNELS), dtype=im.dtype)

    # compute column height
    ch = int(im.shape[0] / (COLOR_CHANNELS * rw))
    im_color_offset = rw * ch
    im_base_offset = rw * rs

    cp_width = (re-rs)*rw   # copy width
    for i in range(0,COLOR_CHANNELS):
        ims = im_base_offset + im_color_offset * i  # image start
        ime = ims + cp_width                        # image end

        ret_im[i*cp_width:(i+1)*cp_width] = im[ims:ime]

    return ret_im

# takes a vectorized image im, and pastes the vectorized rows from v over the image
# rs : row start
# re : row end (noninclusive)
# rw : row width
# im : vectorized image rows are being pasted to
# v  : image being pasted from
def paste_over_rows(rs, re, rw, im, v):
    COLOR_CHANNELS = 3

    # compute column height
    ch = int(im.shape[0] / (COLOR_CHANNELS * rw))
    im_color_offset = rw * ch
    im_base_offset = rw * rs

    cp_width = (re - rs) * rw  # copy width
    for i in range(0, COLOR_CHANNELS):
        ims = im_base_offset + im_color_offset * i  # image start
        ime = ims + cp_width  # image end
        vs = i * cp_width
        ve = (i+1) * cp_width

        im[ims:ime] = v[vs:ve]

    return im

# returns a basic dataset of for x,y
# the dataset is in "standard image form" (image stored as XxYx3
def load_basic_dataset_std(num):
    ims = open_data_ims(50000)
    x, y = seperate_xy_ims(ims[0:num])

    im_h = 16
    im_w = 32
    num_channels = 3

    new_x = np.zeros((num, im_h, im_w, num_channels))
    new_y = np.zeros((num, im_h, im_w, num_channels))

    for i in range(0,num):
        new_x[i] = im_vec_to_im_std(x[i], im_w, im_h)
        new_y[i] = im_vec_to_im_std(y[i], im_w, im_h)

    return new_x,new_y

# returns basic dataset for x,y in form CHANNELS by HEIGHT by WIDTH
def load_basic_dataset_chw(num):
    x,y = load_basic_dataset_std(num)

    channels = x.shape[3]
    imw = x.shape[2]
    imh = x.shape[1]

    x_new = np.zeros((num,channels, imh, imw), dtype=x.dtype)
    y_new = np.zeros((num,channels, imh, imw), dtype=x.dtype)

    for i in range(0,num):
        x_new[i] = std_im_to_chw(x[i])
        y_new[i] = std_im_to_chw(y[i])

    return x_new, y_new


# converts 1 single std_im to 1 single chw_im
# a chw image is an image that is stored in the form: CHANNEL x HEIGHT x WIDTH
def std_im_to_chw(im):
    channels = im.shape[2]
    imw = im.shape[1]
    imh = im.shape[0]

    im_new = np.zeros((channels, imh, imw), dtype=im.dtype)

    for c in range(0,channels):
        im_new[c] = im[:,:,c]

    return im_new

# reverses the effects of std_im_to_chw()
def chw_im_to_std(im):
    channels = im.shape[0]
    imh = im.shape[1]
    imw = im.shape[2]

    im_new = np.zeros((imh, imw, channels), dtype=im.dtype)

    for c in range(0,channels):
        im_new[:, :, c] = im[c]

    return im_new

# performs same function as chw_im_to_std, but on multiple at the same time
def chw_ims_to_std(ims):
    num = ims.shape[0]

    channels = ims.shape[1]
    imh = ims.shape[2]
    imw = ims.shape[3]

    ims_new = np.zeros((num,imh, imw, channels), dtype=ims.dtype)

    for i in range(0,num):
        ims_new[i] = chw_im_to_std(ims[i])

    return ims_new

def regularize_data(data):
    print("Regularize data: min:{} mean:{} max:{}".format(np.min(data), np.mean(data), np.max(data)))
    data /= 255.0
    return data
def deregularize_data(data):
    data *= 255.0
    return data

def regularize_0_1(data):
    return data-np.min(data)/(np.max(data)-np.min(data))
