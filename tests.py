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


