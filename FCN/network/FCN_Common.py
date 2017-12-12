import numpy as np
import tensorflow as tf
import cv2
def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, num_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    k_size = 2 * factor - factor % 2
    weights = np.zeros((k_size,
                        k_size,
                        num_classes,
                        num_classes), dtype=np.float32)
    upsample_kernel = upsample_filt(k_size)
    for i in range(num_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights

if __name__=='__main__':
    w = bilinear_upsample_weights(2,3)
    img = cv2.imread('../test.jpg')
    img = cv2.resize(img,(200,200))
    img = np.expand_dims(img,axis=0)
    

    # x = np.array([[14,20],[15,24]],np.float32)
    # x = np.expand_dims(x,axis=0)
    # x = np.expand_dims(x, axis=3)
    #
    # output_shape = (1,4,4,1)
    # filter = tf.constant(w)
    # out = tf.nn.conv2d_transpose(x,
    #                              filter=filter,
    #                              output_shape=output_shape,
    #                              strides=[1,2,2,1])
    # resize_img = tf.image.resize_images(x,(4,4))
    # with tf.Session() as sess:
    #     y = sess.run([out,resize_img])
    #     o1 = y[0].squeeze()
    #     o2 = y[1].squeeze()
    #     print (o1)
    #     print (o2)
    # print(x[0,:,:,0])