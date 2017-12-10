import tensorflow as tf

def get_one_hot_labels_from_anno(anno,class_labels):
    """
    Returns tensor of size (width, height, num_classes) derived from annotation tensor.
    :param anno: annotation tensor with size (width,height)  whose value like (0,1,2,..num_classes)
    :param class_labels:  List contains the numbers that represent class label
    :return: (width, height, num_classes)   one_hot like  labels
    """
    labels = list(map(lambda  x:tf.equal(anno,x),class_labels))
    labels_stacked = tf.stack(labels,axis=2)
    return tf.to_float(labels_stacked)

def get_one_hot_labels_from_anno_batch(anno_batch,class_labels):
    return tf.map_fn(fn=lambda x: get_one_hot_labels_from_anno(x,class_labels),
                             elems=anno_batch,
                             dtype=tf.float32)

def seg_cross_entropy_loss( logits,anno_batch,class_labels):
    """
    sum of cross entropy loss
    :param logits:  [batch_size, height,width, num_classes]   tensor
    :param anno_batch:  [batch_size, height,width, 1]   tensor  whose value like (0,1,2,..num_classes)
    :param class_labels:  List contains the numbers that represent class label
    :return: sum of cross entropy loss
    """
    anno_batch = tf.squeeze(anno_batch)
    one_hot_labels_batch = get_one_hot_labels_from_anno_batch(anno_batch,class_labels)
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=one_hot_labels_batch)
    cross_entropy_sum = tf.reduce_mean(cross_entropy)
    return cross_entropy_sum
