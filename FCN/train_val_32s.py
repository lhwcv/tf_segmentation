import tensorflow as tf
import tensorflow.contrib.slim  as slim
import  argparse
from network.FCN_VGG_32s import FCN_vgg_32s
from network.FCN_VGG_16s import FCN_vgg_16s
from  utils.pascal_voc_seg_dataset import  *
from  utils.augmentation import  seg_data_augmentation
from  utils.seg_loss import  seg_cross_entropy_loss
parse = argparse.ArgumentParser()
parse.add_argument('--finetune_model',type=str,default='../pretrained_model/vgg_16_2016_08_28.ckpt')
parse.add_argument('--train_records',type=str,default='./tfrecords/train.tfrecords')
parse.add_argument('--val_records',type=str,default='./tfrecords/val.tfrecords')
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


def train():
    num_classes = len(voc_classes_lut_map)
    class_labels = voc_classes_lut_map.keys()

    assert os.path.exists(FLAGS.train_records)
    ### read input data and annotation
    filename_queue = tf.train.string_input_producer(
        [FLAGS.train_records], num_epochs=int(FLAGS.max_steps * FLAGS.batch_size / TRAIN_IMAGES_NUM))
    image, anno = read_and_decode_tensors_from_tfrecords(filename_queue)

    image, anno = seg_data_augmentation(image, anno, FLAGS.crop_h, FLAGS.crop_w)

    image_batch, anno_batch = tf.train.shuffle_batch([image, anno],
                                                     batch_size=FLAGS.batch_size,
                                                     capacity=3000,
                                                     num_threads=4,
                                                     min_after_dequeue=1000)

    logits, variables_mapping = FCN_vgg_32s(image_batch, True, weight_decay=FLAGS.weight_decay)
    loss = seg_cross_entropy_loss(logits=logits, anno_batch=tf.squeeze(anno_batch), class_labels=class_labels)
    tf.add_to_collection('losses', loss)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    lr = tf.train.exponential_decay(FLAGS.base_lr,
                                    global_step=global_step,
                                    decay_steps=FLAGS.decay_steps,
                                    decay_rate=FLAGS.decay_factor
                                    )
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

    load_pretrained_param_fn = slim.assign_from_checkpoint_fn(model_path=FLAGS.finetune_model,
                                                              var_list=variables_mapping)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    tf.summary.scalar('cross_entropy_loss', loss)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir)

    model_variables = slim.get_model_variables()
    saver = tf.train.Saver(model_variables)
    with tf.Session()  as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print('restore from ', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step_np = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            sess.run(tf.assign(global_step, global_step_np))
        else:
            load_pretrained_param_fn(sess)
            print('No past checkpoint file found, using pretrained model in %s '% FLAGS.finetune_model)

        for step in range(FLAGS.max_steps):
            loss_now, global_step_np, lr_np, summary_string, _ = sess.run([total_loss, global_step, lr,
                                                                           merged_summary_op,
                                                                           train_step])
            if step % 10 == 0 and step > 0:
                print("loss: " + str(loss_now) + " \t lr: " + str(lr_np) + "\tstep:" + str(global_step_np))
                summary_writer.add_summary(summary_string, global_step_np)
            if global_step_np % FLAGS.save_model_per_steps == 0 and step > 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                save_path = saver.save(sess, checkpoint_path, global_step=global_step)
                print("Save model in: %s" % save_path)
        coord.request_stop()
        coord.join(threads)

    summary_writer.close()


def val():

    tfrecord_filename = FLAGS.val_records
    number_of_classes = 21
    filename_queue = tf.train.string_input_producer(
        [tfrecord_filename], num_epochs=1)
    image, annotation = read_and_decode_tensors_from_tfrecords(filename_queue)
    image_batch_tensor = tf.expand_dims(image, axis=0)
    annotation_batch_tensor = tf.expand_dims(annotation, axis=0)
    input_image_shape = tf.shape(image_batch_tensor)
    image_height_width = input_image_shape[1:3]
    image_height_width_float = tf.to_float(image_height_width)
    image_height_width_multiple = tf.round(image_height_width_float / 32) * 32
    image_height_width_multiple = tf.to_int32(image_height_width_multiple)
    image_batch_tensor = tf.image.resize_images(image_batch_tensor, image_height_width_multiple)


    logits, _ = FCN_vgg_32s(image_batch_tensor, is_training=False)
    pred = tf.argmax(logits, dimension=3)
    pred = tf.expand_dims(pred, 3)
    original_size_predictions = tf.image.resize_nearest_neighbor(images=pred, size=image_height_width)

    ckpt = tf.train.get_checkpoint_state('./log/train/')
    print(ckpt.model_checkpoint_path)


    miou, update_op = slim.metrics.streaming_mean_iou(predictions=original_size_predictions,
                                                      labels=annotation_batch_tensor,
                                                      num_classes=number_of_classes
                                                      )
    initializer = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(initializer)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            print('restore from ', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print ('No model')
            return
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(1449):
            image_np, annotation_np, pred_np, tmp = sess.run([image, annotation, pred, update_op])
        coord.request_stop()
        coord.join(threads)
        res = sess.run(miou)
        print("Pascal VOC 2012  Mean IOU: " + str(res))

if __name__ == '__main__':
    val()
    #train()

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