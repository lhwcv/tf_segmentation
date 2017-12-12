import tensorflow as tf
import os
from network.FCN_VGG_16s import FCN_vgg_16s
from  utils.pascal_voc_seg_dataset import  *
import  cv2
IMAGE_DIR ='D:\\into_DL\\dataset\\VOC2012\\JPEGImages\\'

images_filename= os.listdir(IMAGE_DIR)

with tf.Session() as sess:
    images = tf.placeholder(dtype=tf.float32,shape=[None,None,None,3])
    logits,_=FCN_vgg_16s(images,is_training=False)
    pred = tf.argmax(logits, dimension=3)
    ckpt = tf.train.get_checkpoint_state('./log/train_16s/')
    print(ckpt.model_checkpoint_path)
    saver = tf.train.Saver()
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
    for f in images_filename:
        src = cv2.imread(IMAGE_DIR+f)
        #src = cv2.resize(src,(224,224))
        img = np.array(src,np.float32)#-[preprocess._B_MEAN,preprocess._G_MEAN,preprocess._R_MEAN]
        img = np.expand_dims(img,axis=0)
        output = sess.run([pred],feed_dict={images:img})
        output=label_array_to_color_img(colormap,output[0][0])
        cv2.namedWindow('pred',0)
        cv2.imshow('pred',output)
        cv2.namedWindow('img', 0)
        cv2.imshow('img', src)
        cv2.waitKey(0)





