import numpy as np
import cv2

def im2col(img,ksize,s=1,p=1):
    N,C,H,W = img.shape
    npad = ((0, 0), (0, 0), (p, p), (p, p))
    img = np.pad(img, pad_width=npad,
               mode='constant', constant_values=0)
    Ho = int(np.floor((H + 2 * p - ksize) / s) + 1)
    Wo = int(np.floor((W + 2 * p - ksize) / s) + 1)
    out = np.zeros((C*ksize*ksize, N*Ho*Wo))

    for c in range(C):
        for y in range(Ho):
            for x in range(Wo):
                for i in range(ksize):
                    for j in range(ksize):
                        row = c*ksize*ksize + i*ksize + j
                        for n in range(N):
                            col = y*Wo*N + x*N + n
                            out[row,col] = img[n,c,s*y+i,s*x+j]
    return out

def conv_as_matmul(x,w,b,s=1,p=1):
    N, C, H, W = x.shape
    o_maps_num, input_maps_2, Kh, Kw = w.shape
    # assert (W + 2 * p - Kw) % s == 0, 'width  not work'
    # assert (H + 2 * p - Kh) % s == 0, 'height not work'
    Ho = int(np.floor((H + 2 * p - Kh) / s) + 1)
    Wo = int(np.floor((W + 2 * p - Kw) / s) + 1)
    out = np.zeros((N, o_maps_num, Ho, Wo))
    imgcols = im2col(x,Kw,s,p)
    ret = w.reshape((w.shape[0],-1)).dot(imgcols) + b.reshape(-1,1)
    out = ret.reshape(o_maps_num,Ho,Wo,N ).transpose(3,0,1,2)
    return out


def conv_forward(x,w,b,s=1,p=1):
    '''
    卷积的原始粗糙实现
    :param x:  [N, input_maps, H,W] array
    :param w:  [output_maps,input_maps, Kh, Kw]
    :param b:  [ouputs_maps]
    :param s:  stride
    :param p:  padding
    :return:   [N, output_maps, Ho,Wo]
    '''
    N, input_maps_1, H,W = x.shape
    o_maps_num,input_maps_2,Kh,Kw = w.shape
    assert input_maps_1==input_maps_2
    i_maps_num = input_maps_1
    npad = ((0, 0), (0, 0), (p, p), (p, p))
    x = np.pad(x, pad_width=npad,
                   mode='constant', constant_values=0)
    Ho = int(np.floor((H + 2 * p - Kh) / s) + 1)
    Wo = int(np.floor((W + 2 * p - Kw) / s) + 1)
    out = np.zeros((N,o_maps_num,Ho,Wo))
    for n in range(N):
        for o in range(o_maps_num):
            for row in range(Ho):
                for col in range(Wo):
                    out[n,o,row,col] = np.sum(x[n,:,row*s:row*s+Kh,col*s:col*s+Kw]*w[o,:,:,:]) + b[o]
    return out

def normalize_img(img):
    img_max, img_min = np.max(img), np.min(img)
    img = 255.0 * (img - img_min) / (img_max - img_min)
    img = np.array(img,np.uint8)
    return img

def test_conv():
    #output_maps:2
    #input_maps:3
    #kernel (3,3)
    w = np.zeros((2, 3, 3, 3))
    # The first filter converts the image to grayscale.
    w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
    w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
    w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]
    # The second filter  detects horizontal edges
    w[1, 0, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    w[1, 1, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    b = np.array([0, 128])

    src = cv2.imread('test.jpg')
    src = cv2.resize(src,(200,200))
    img = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
    img = img.transpose([2,0,1])
    img = np.expand_dims(img,0)
    print(img.shape)
    #out = conv_forward(img, w, b, s=1)
    out = conv_as_matmul(img,w,b,s=2)
    print(out.shape)
    cv2.namedWindow('img',0)
    cv2.imshow('img',src)
    cv2.namedWindow('gray',0)
    cv2.imshow('gray',normalize_img(out[0, 0, :, :]))
    cv2.namedWindow('edge',0)
    cv2.imshow('edge', normalize_img(out[0, 1, :, :]))
    cv2.waitKey(0)

if __name__=='__main__':
    test_conv()



