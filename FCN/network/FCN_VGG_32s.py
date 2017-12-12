import tensorflow as tf
import tensorflow.contrib.slim  as slim
import slim.nets.vgg as vgg
import slim.preprocessing.vgg_preprocessing as preprocess
from network.FCN_Common import bilinear_upsample_weights

def FCN_vgg_32s(image_batch,is_training,weight_decay=0.0005, num_classes=21):
    with tf.variable_scope('fcn_vgg_32s') as scope:
        upsample_ratio=32
        mean_centered_input = tf.to_float(image_batch) - \
                              [preprocess._B_MEAN,preprocess._G_MEAN,preprocess._R_MEAN]
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
            logits,end_points=vgg.vgg_16(mean_centered_input,
                                         num_classes=num_classes,
                                         is_training=is_training,
                                         spatial_squeeze=False,
                                         fc_conv_padding='SAME')
        logits_shape=tf.shape(logits)

        output_shape=tf.stack([logits_shape[0],
                              logits_shape[1]*upsample_ratio,
                              logits_shape[2]*upsample_ratio,
                              logits_shape[3]
                              ])
        filter = tf.constant(bilinear_upsample_weights(upsample_ratio,num_classes))
        output_logits = tf.nn.conv2d_transpose(logits,
                                               filter,
                                               output_shape=output_shape,
                                               strides=[1,upsample_ratio,upsample_ratio,1]
                                               )
        ### for load pretraind parameters exclude fc8
        vgg_16_variables_mapping = {}
        vgg_16_variables = slim.get_variables(scope)
        for variable in vgg_16_variables:
            key = variable.name[len(scope.name) + 1:-2]
            if 'fc8' not in key:
                vgg_16_variables_mapping[key] = variable
        return   output_logits,vgg_16_variables_mapping




