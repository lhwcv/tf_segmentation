import tensorflow as tf
import tensorflow.contrib.slim  as slim
import slim.nets.vgg as vgg
import slim.preprocessing.vgg_preprocessing as preprocess
from network.FCN_Common import bilinear_upsample_weights

def FCN_vgg_16s(image_batch,is_training,weight_decay=0.0005, num_classes=21):
    with tf.variable_scope('fcn_vgg_16s') as scope:
        upsample_ratio1 = 2  ## pool5'--> pool4
        upsample_ratio2 = 16 ## to source image size
        mean_centered_input = tf.to_float(image_batch) - \
                              [preprocess._B_MEAN,preprocess._G_MEAN,preprocess._R_MEAN]
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
            logits,end_points=vgg.vgg_16(mean_centered_input,
                                         num_classes=num_classes,
                                         is_training=is_training,
                                         spatial_squeeze=False,
                                         fc_conv_padding='SAME')
        logits_shape=tf.shape(logits)
        fuse_input_from_pool5_shape=tf.stack([logits_shape[0],
                              logits_shape[1]*upsample_ratio1,
                              logits_shape[2]*upsample_ratio1,
                              logits_shape[3]
                              ])
        filter_pool5 = tf.constant(bilinear_upsample_weights(upsample_ratio1,num_classes))
        ## pool5'  upsample to  21* 14*14  (let input size: (224,224) )
        pool5_upscore = tf.nn.conv2d_transpose(logits,
                                               filter_pool5,
                                               output_shape=fuse_input_from_pool5_shape,
                                               strides=[1,upsample_ratio1,upsample_ratio1,1]
                                               )
       ##  pool4  conv to 21*14*14  then fuse with pool5_upscore
        pool4= end_points['fcn_vgg_16s/vgg_16/pool4']
        pool4_logits = slim.conv2d(pool4,
                                   num_classes,
                                   [1, 1],
                                   activation_fn=None,
                                   scope='pool4_fc')
        fuse = pool5_upscore+pool4_logits

        ##  fuse  trans_conv  to the final output
        logits_shape2 = tf.shape(fuse)
        output_shape = tf.stack([logits_shape2[0],
                                 logits_shape2[1] * upsample_ratio2,
                                 logits_shape2[2] * upsample_ratio2,
                                 logits_shape2[3]
                                 ])
        filter_fuse = tf.constant(bilinear_upsample_weights(upsample_ratio2, num_classes))
        output_logits = tf.nn.conv2d_transpose(fuse,
                                               filter_fuse,
                                               output_shape=output_shape,
                                               strides=[1, upsample_ratio2, upsample_ratio2, 1]
                                               )


        ### for load pretraind parameters exclude fc8
        vgg_16_variables_mapping = {}
        vgg_16_variables = slim.get_variables(scope)
        for variable in vgg_16_variables:
            key = variable.name[len(scope.name) + 1:-2]
            if 'pool4_fc' in key:
                continue
            if 'fc8' not in key:
                vgg_16_variables_mapping[key] = variable
        return   output_logits,vgg_16_variables_mapping




