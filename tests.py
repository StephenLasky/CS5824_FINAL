import numpy as np
import functions as f

def test_extract_vec_rows():
    # TEST 1: test single rows
    rw = 3
    ch = 3
    im1 = np.zeros((27), dtype=np.int)
    for i in range(0,27): im1[i] = i

    for i in range(0,3):
        res1 = np.asarray([0,1,2,9,10,11,18,19,20], dtype=np.int) + i * rw
        rs = i
        re = i + 1
        t1 = f.extract_vec_rows(rs,re,rw,im1)
        assert np.array_equal(res1, t1)

    # TEST 2: test double rows
    for i in range(0,2):
        res2 = np.asarray([0,1,2,3,4,5, 9,10,11,12,13,14, 18,19,20,21,22,23], dtype=np.int) + i * rw
        rs = i
        re = i + 2
        t2 = f.extract_vec_rows(rs,re,rw,im1)
        assert np.array_equal(res2, t2)

    # TEST 3: test entire image
    res3 = im1
    rs,re = 0,3
    t3 = f.extract_vec_rows(rs,re,rw,im1)
    assert np.array_equal(res3, t3)

    # TEST 4: test zero rows
    res4 = np.asarray([])
    rs,re = 0,0
    t4 = f.extract_vec_rows(rs,re,rw,im1)
    assert np.array_equal(res4, t4)

    # TEST 5: test im of unequal column / width sizes
    im2 = np.zeros((24))    # try row_width = 4, col_height = 2
    for i in range(0,24): im2[i] = i
    rw = 4
    for i in range(0,2):
        res5 = np.asarray([0,1,2,3, 8,9,10,11, 16,17,18,19]) + i * rw
        rs, re = i, i+1
        t5 = f.extract_vec_rows(rs,re,rw,im1)
        assert np.array_equal(res5, t5)

    # TEST 6: (more like 5.2)
    rw = 2 # try row_width = 2, col_height = 4
    for i in range(0,2):
        res6 = np.asarray([0,1, 8,9, 16,17]) + i * rw
        rs, re = i, i+1
        t6 = f.extract_vec_rows(rs,re,rw,im1)
        assert np.array_equal(res6, t6)

def test_paste_over_rows():
    rw = 3  # row width
    ch = 3  # col height

    # test first row - single row
    im1 = np.zeros((27)) # 3x3 image with 3 color channels, in vector form
    v1 = np.ones((9))    # one row of ones in vector form
    for i in range(0,9): v1[i] = i  # v1 now takes the form [0,1,2,3,4,5,6,7,8]

    rs, re = 0,1
    t1 = f.paste_over_rows(rs,re,rw,im1,v1)
    res1 = np.asarray([0,1,2,0,0,0,0,0,0, 3,4,5,0,0,0,0,0,0, 6,7,8,0,0,0,0,0,0])
    assert np.array_equal(t1, res1)

    # test second row - single row
    im1 = np.zeros((27))  # 3x3 image with 3 color channels, in vector form
    v1 = np.ones((9))  # one row of ones in vector form
    for i in range(0, 9): v1[i] = i  # v1 now takes the form [0,1,2,3,4,5,6,7,8]

    rs, re = 1,2
    t1 = f.paste_over_rows(rs, re, rw, im1, v1)
    res1 = np.asarray([0,0,0,0,1,2,0,0,0, 0,0,0,3,4,5,0,0,0, 0,0,0,6,7,8,0,0,0])
    assert np.array_equal(t1, res1)

    # test third (final) row - single row
    im1 = np.zeros((27))  # 3x3 image with 3 color channels, in vector form
    v1 = np.ones((9))  # one row of ones in vector form
    for i in range(0, 9): v1[i] = i  # v1 now takes the form [0,1,2,3,4,5,6,7,8]

    rs, re = 2, 3
    t1 = f.paste_over_rows(rs, re, rw, im1, v1)
    res1 = np.asarray([0,0,0,0,0,0,0,1,2, 0,0,0,0,0,0,3,4,5, 0,0,0,0,0,0,6,7,8])
    assert np.array_equal(t1, res1)

    # OKAY - SHIFT TO TESTING TWO ROWS ARE A TIME!!
    # test first two rows - two at a time
    im2 = np.zeros((27))    # 3x3 image with 3 color channels, in vector form
    v2 = np.ones((18))           # one row of ones in vector form
    for i in range(0, 18): v2[i] = i  # v1 now takes the form [0,1,2,3,4,5,6,7,8, ... 17]

    rs, re = 0, 2
    t2 = f.paste_over_rows(rs, re, rw, im2, v2)
    res2 = np.asarray([0,1,2,3,4,5,0,0,0, 6,7,8,9,10,11,0,0,0, 12,13,14,15,16,17,0,0,0])
    assert np.array_equal(t2, res2)

    # test final two rows - two at a time
    im2 = np.zeros((27))  # 3x3 image with 3 color channels, in vector form
    v2 = np.ones((18))  # one row of ones in vector form
    for i in range(0, 18): v2[i] = i  # v1 now takes the form [0,1,2,3,4,5,6,7,8, ... 17]

    rs, re = 1, 3
    t2 = f.paste_over_rows(rs, re, rw, im2, v2)
    res2 = np.asarray([0,0,0,0,1,2,3,4,5, 0,0,0,6,7,8,9,10,11, 0,0,0,12,13,14,15,16,17])
    assert np.array_equal(t2, res2)








