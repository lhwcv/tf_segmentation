import os
import logging
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
voc_classes_lut_map={
    0: 'background',    1: 'aeroplane',    2: 'bicycle',
    3: 'bird',    4: 'boat',    5: 'bottle',    6: 'bus',
    7: 'car',    8: 'cat',    9: 'chair',    10: 'cow',    11: 'diningtable',
    12: 'dog',    13: 'horse',    14: 'motorbike',
    15: 'person',    16: 'potted-plant',    17: 'sheep',
    18: 'sofa',    19: 'train',    20: 'tv/monitor'
}
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

def anno_img_to_numpy_label_array(colormap,anno):
    """
    :param colormap: colormap defined in pascal VOC
    :param anno:  [height,width,3]  array read from png file  with BGR  order
    :return: [height,width]  array whose value between 0 and class_num-1  (here 0 to 20)
    """
    height = anno.shape[0]
    width = anno.shape[1]
    label_array = np.zeros((height,width),np.int32)
    for i,value in enumerate(colormap):
        ##B G R  oder, so here 2 1 0
        arr = np.where(anno[:,:,2]==value[0],i,0)&\
              np.where(anno[:,:,1]==value[1],i,0)& \
              np.where(anno[:, :, 0] == value[2], i, 0)
        label_array=label_array+arr
    return np.array(label_array,np.uint8)

def label_array_to_color_img(colormap, label):
    """
    :param colormap: colormap defined in pascal VOC
    :param label: [height,width]  array whose value between 0 and class_num-1  (here 0 to 20)
    :return: [height,width,3]  color image with BGR  order
    """
    height =label.shape[0]
    width = label.shape[1]
    img = np.zeros((height, width,3), np.uint8)
    for i, value in enumerate(colormap):
        ## BGR
        img[:, :, 0][np.where(label == i)] = value[2]
        img[:, :, 1][np.where(label == i)] = value[1]
        img[:, :, 2][np.where(label == i)] = value[0]
    return img

def get_pascal_seg_img_lists_txts(pascal_root):
    """
    :param pascal_root:  string  Full path to the root dir of PASCAL_VOC e.g: /home/dataset/VOC2012
    :return: [string string strin] Array that contains paths for train/val/trainval txts
    """
    dir = os.path.join(pascal_root,'ImageSets/Segmentation/')
    train_list_filename = os.path.join(dir,'train.txt')
    val_list_filename = os.path.join(dir, 'val.txt')
    trainval_list_filename = os.path.join(dir, 'trainval.txt')
    lists =  [train_list_filename, val_list_filename, trainval_list_filename]
    for i in lists:
        if not os.path.exists(i):
            logging.error('File not found %s'%i)
    return lists

def readlines_with_strip(filename):
    with  open(filename, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def get_pascal_img_anno_pairs(root_dir,mode='train'):
    """
    :param root_dir:  string  Full path to the root dir of PASCAL_VOC e.g: /home/dataset/VOC2012
    :param mode:  string  'train'/'val'/'trainval'
    :return: zip(img_filename, anno_filename)
    """
    assert  mode in ['train','val','trainval']
    txt_lists = get_pascal_seg_img_lists_txts(root_dir)
    if mode=='train':
        file_lists = readlines_with_strip(txt_lists[0])
    if mode=='val':
        file_lists = readlines_with_strip(txt_lists[1])
    if mode=='trainval':
        file_lists = readlines_with_strip(txt_lists[2])
    images_full_names = [os.path.join(root_dir + 'JPEGImages/', x + '.jpg') for x in file_lists]
    anno_full_names = [os.path.join(root_dir + 'SegmentationClass/', x + '.png') for x in file_lists]
    return  zip(images_full_names, anno_full_names)




def show_img_and_anno():
    root_dir = '../../dataset/VOC2012/'
    pairs = get_pascal_img_anno_pairs(root_dir,'trainval')
    for img_path,anno_path in pairs:
        img= cv2.imread(img_path)
        anno = cv2.imread(anno_path)
        anno = anno_img_to_numpy_label_array(colormap,anno)
        color_label = label_array_to_color_img(colormap,anno)
        # annotation = np.array(Image.open(anno_path),np.uint8)
        # annotation = np.where(annotation==255,0,annotation)
        #print(anno.dtype)
        #print(np.mean(anno))
        #print(np.mean(annotation))
        #print (anno.shape)
        #print(annotation.shape)
        cv2.namedWindow('img',0)
        cv2.imshow('img',img)
        cv2.namedWindow('anno', 0)
        cv2.imshow('anno', color_label)
        cv2.waitKey(0)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_anno_img_pairs_to_records(pairs, tfrecords_filename):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for img_path, anno_path in pairs:
        img = cv2.imread(img_path)
        anno = cv2.imread(anno_path)
        anno = anno_img_to_numpy_label_array(colormap, anno)
        height= img.shape[0]
        width = img.shape[1]
        img_raw = img.tobytes()
        anno_raw = anno.tobytes()
        example= tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(anno_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
def build_tfrecords(root_dir,tfrecords_filename,mode='train'):
    """
    :param root_dir:  string  Full path to the root dir of PASCAL_VOC e.g: /home/dataset/VOC2012
    :param tfrecords_filename:
    :param mode: string  'train'/'val'/'trainval'
    """
    pairs = get_pascal_img_anno_pairs(root_dir, mode)
    write_anno_img_pairs_to_records(pairs,tfrecords_filename)

def read_img_anno_pair_from_records(filename):
    record_iter=tf.python_io.tf_record_iterator(path=filename)
    pairs=[]
    for record in record_iter:
        example=tf.train.Example()
        example.ParseFromString(record)
        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        img_raw = (example.features.feature['image_raw'].bytes_list.value[0])
        mask_raw = (example.features.feature['mask_raw'].bytes_list.value[0])
        img = np.reshape(np.fromstring(img_raw,dtype=np.uint8),(height,width,-1))
        anno = np.reshape(np.fromstring(mask_raw, dtype=np.uint8), (height, width))
        pairs.append((img,anno))
    return pairs
def show_img_and_anno_from_records(filename):
    pairs = read_img_anno_pair_from_records(filename)
    for img,anno in pairs:
        color_label = label_array_to_color_img(colormap, anno)
        cv2.namedWindow('img',0)
        cv2.imshow('img',img)
        cv2.namedWindow('anno', 0)
        cv2.imshow('anno', color_label)
        cv2.waitKey(1000)

def read_and_decode_tensors_from_tfrecords(filename_queue):
    """
    :param filename_queue:   String queue object from tf.train.string_input_producer()
    :return: image and annotation tensors
    """
    reader = tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    fea = tf.parse_single_example(serialized_example,
                                  features={
                                      'height':tf.FixedLenFeature([],tf.int64),
                                      'width': tf.FixedLenFeature([], tf.int64),
                                      'image_raw': tf.FixedLenFeature([], tf.string),
                                      'mask_raw': tf.FixedLenFeature([], tf.string)
                                  })
    height = tf.cast(fea['height'],tf.int32)
    width = tf.cast(fea['width'], tf.int32)
    img = tf.decode_raw(fea['image_raw'],tf.uint8)
    mask = tf.decode_raw(fea['mask_raw'], tf.uint8)

    img = tf.reshape(img, [height, width, 3])
    mask = tf.reshape(mask, [height, width, 1])

    return img,mask


if __name__ =='__main__':
    #show_img_and_anno()
    #build_tfrecords('../../dataset/VOC2012/','../tfrecords/train.tfrecords','train')
    show_img_and_anno_from_records('../tfrecords/val.tfrecords')