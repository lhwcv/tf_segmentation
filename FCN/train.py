import tensorflow as tf
import tensorflow.contrib.slim  as slim
import  argparse
from network.FCN_VGG_32s import FCN_vgg_32s
from  utils.pascal_voc_seg_dataset import  *
from  utils.augmentation import  seg_data_augmentation
from  utils.seg_loss import  seg_cross_entropy_loss
parse = argparse.ArgumentParser()
parse.add_argument('--finetune_model',type=str,default='../pretrained_model/vgg_16_2016_08_28.ckpt')
parse.add_argument('--train_records',type=str,default='./tfrecords/train.tfrecords')
parse.add_argument('--train_dir',type=str,default='./log/train/')
parse.add_argument('--batch_size',type=int,default=8)
parse.add_argument('--crop_w',type=int,default=480)
parse.add_argument('--crop_h',type=int,default=320)
parse.add_argument('--base_lr',type=float,default=0.0001)
parse.add_argument('--decay_steps',type=int,default=500000)
parse.add_argument('--decay_factor',type=float,default=0.1)
parse.add_argument('--weight_decay',type=float,default=0.0005)
parse.add_argument('--max_steps',type=int,default=20000)
parse.add_argument('--save_model_per_steps',type=int,default=1000)
TRAIN_IMAGES_NUM = 1465  ##for  VOC12

FLAGS = parse.parse_args()

num_classes = len(voc_classes_lut_map)
class_labels = voc_classes_lut_map.keys()

assert  os.path.exists(FLAGS.train_records)
### read input data and annotation
filename_queue = tf.train.string_input_producer(
    [FLAGS.train_records], num_epochs=int(FLAGS.max_steps*FLAGS.batch_size/TRAIN_IMAGES_NUM))
image,anno = read_and_decode_tensors_from_tfrecords(filename_queue)

image,anno = seg_data_augmentation(image,anno,FLAGS.crop_h,FLAGS.crop_w)


image_batch, anno_batch = tf.train.shuffle_batch( [image, anno],
                                             batch_size=FLAGS.batch_size,
                                             capacity=3000,
                                             num_threads=4,
                                             min_after_dequeue=1000)
logits,variables_mapping=FCN_vgg_32s(image_batch,True,weight_decay=FLAGS.weight_decay)
loss = seg_cross_entropy_loss(logits=logits,anno_batch=tf.squeeze(anno_batch),class_labels=class_labels)

global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
lr = tf.train.exponential_decay(FLAGS.base_lr,
                                global_step,
                                FLAGS.decay_steps,
                                FLAGS.decay_factor,
                                staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

load_pretrained_param_fn = slim.assign_from_checkpoint_fn(model_path=FLAGS.finetune_model,
                                         var_list=variables_mapping)
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

tf.summary.scalar('cross_entropy_loss', loss)
merged_summary_op = tf.summary.merge_all()

summary_writer = tf.summary.FileWriter(FLAGS.train_dir)

model_variables = slim.get_model_variables()
saver = tf.train.Saver(model_variables)
with tf.Session()  as sess:
    sess.run(init_op)
    load_pretrained_param_fn(sess)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for step in range(FLAGS.max_steps):
        cross_entropy, summary_string, _ = sess.run([loss,
                                                     merged_summary_op,
                                                     train_step])
        print("loss: " + str(cross_entropy)+ " \t step: "+str(step))
        summary_writer.add_summary(summary_string, step)
        if step % FLAGS.save_model_per_steps == 0 and step>0:
            save_path = saver.save(sess, FLAGS.train_dir,global_step=global_step)
            print("Save model in: %s" % save_path)
    coord.request_stop()
    coord.join(threads)

summary_writer.close()

## Take a look at ckpt varibles
#ckpt_varible_list = slim.checkpoint_utils.list_variables(FLAGS.finetune_model)
#for i in ckpt_varible_list:
#    print (i)
#images = tf.placeholder(tf.float32,shape=[None,320,480,3])
#logits,varibles=FCN_vgg_32s(images,True)

#print(varibles)

# with tf.Session() as sess:
#     lists = sess.run([ckpt_varible_list])
#     for item in lists:
#         print(item)

#if __name__=='__main__':
#    tf.app.run()