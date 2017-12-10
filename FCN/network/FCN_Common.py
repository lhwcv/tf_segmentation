import numpy as np

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