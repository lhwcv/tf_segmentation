from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import  image_ops_impl
from tensorflow.python.ops import math_ops
import  tensorflow as tf

#### See  https://stackoverflow.com/questions/42147427/tensorflow-how-to-randomly-crop-input-images-and-labels-in-the-same-way
def random_crop_and_pad_image_and_labels(image, labels, size):
  """Randomly crops `image` together with `labels`.
  Args:
    image: A Tensor with shape [D_1, ..., D_K, N]
    labels: A Tensor with shape [D_1, ..., D_K, M]
    size: A Tensor with shape [K] indicating the crop size.
  Returns:
    cropped_image, cropped_label
  """
  combined = tf.concat([image, labels], axis=2)
  image_shape = tf.shape(image)
  combined_pad = tf.image.pad_to_bounding_box(
      combined, 0, 0,
      tf.maximum(size[0], image_shape[0]),
      tf.maximum(size[1], image_shape[1]))
  last_label_dim = tf.shape(labels)[-1]
  last_image_dim = tf.shape(image)[-1]
  combined_crop = tf.random_crop(
      combined_pad,
      size=tf.concat([size, [last_label_dim + last_image_dim]],
                     axis=0))
  return combined_crop[:, :, :last_image_dim],\
          combined_crop[:, :, last_image_dim:]



def random_flip_img_label_left_right(image,label, seed=None):
  ### modified by  tf.random_flip_left_right()
  image = ops.convert_to_tensor(image, name='image')
  label = ops.convert_to_tensor(label, name='label')
  image = control_flow_ops.with_dependencies(
      image_ops_impl._Check3DImage(image, require_static=False), image)
  label = control_flow_ops.with_dependencies(
      image_ops_impl._Check3DImage(label, require_static=False), label)
  uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
  mirror_cond = math_ops.less(uniform_random, .5)
  result1 = control_flow_ops.cond(mirror_cond,
                                 lambda: array_ops.reverse(image, [1]),
                                 lambda: image)
  result2 = control_flow_ops.cond(mirror_cond,
                                  lambda: array_ops.reverse(label, [1]),
                                  lambda: label)
  return image_ops_impl.fix_image_flip_shape(image, result1),\
         image_ops_impl.fix_image_flip_shape(label, result2)

def seg_data_augmentation(image,label,crop_h,crop_w):
    #print (image,label)
    #image,label = random_crop_and_pad_image_and_labels(image, label,[crop_h,crop_w])
    image = tf.image.resize_image_with_crop_or_pad(image,crop_h,crop_w)
    label = tf.image.resize_image_with_crop_or_pad(label, crop_h, crop_w)
    image,label = random_flip_img_label_left_right(image,label)
    image = tf.cast(image, tf.float32)
    image = tf.image.random_brightness(image, max_delta=20)
    image = tf.image.random_contrast(image,lower=0.7, upper=1.3)
    return image,label
