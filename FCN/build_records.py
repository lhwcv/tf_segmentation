from  utils.pascal_voc_seg_dataset import  build_tfrecords
build_tfrecords('../../dataset/VOC2012/','../tfrecords/train.tfrecords','train')
build_tfrecords('../../dataset/VOC2012/','../tfrecords/val.tfrecords','val')