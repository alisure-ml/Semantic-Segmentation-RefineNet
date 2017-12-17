import time
import os
import tensorflow as tf
import tensorflow.contrib.metrics as tcm
import model
from utils.tfrecords import read_tfrecord_and_decode_into_image_annotation_pair_tensors

tf.app.flags.DEFINE_string('test_data_path', '../data/pascal_val.tfrecords', '')
tf.app.flags.DEFINE_integer('num_classes', 21, '')
tf.app.flags.DEFINE_string('checkpoint_path', '../checkpoints', '')
tf.app.flags.DEFINE_integer('test_size', 384, '')

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    filename_queue = tf.train.string_input_producer([FLAGS.test_data_path], num_epochs=1)
    image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

    image_batch_tensor = tf.expand_dims(image, axis=0)
    annotation_batch_tensor = tf.expand_dims(annotation, axis=0)

    input_image_shape = tf.shape(image_batch_tensor)
    image_height_width = input_image_shape[1:3]
    image_height_width_float = tf.to_float(image_height_width)
    image_height_width_multiple = tf.to_int32(tf.round(image_height_width_float / 32) * 32)

    image_batch_tensor = tf.image.resize_images(image_batch_tensor, image_height_width_multiple)
    annotation_batch_tensor = tf.image.resize_nearest_neighbor(images=annotation_batch_tensor, size=image_height_width)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    logits = model.model(image_batch_tensor, is_training=False)
    pred = tf.argmax(logits, axis=3)
    pred = tf.expand_dims(pred, 3)
    pred = tf.image.resize_nearest_neighbor(images=pred, size=image_height_width)

    pred = tf.reshape(pred, [-1, ])
    gt = tf.reshape(annotation_batch_tensor, [-1, ])
    temp = tf.less_equal(gt, FLAGS.num_classes - 1)
    weights = tf.cast(temp, tf.int32)
    gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))
    acc, acc_update_op = tcm.streaming_accuracy(pred, gt, weights=weights)
    miou, miou_update_op = tcm.streaming_mean_iou(pred, gt, num_classes=FLAGS.num_classes, weights=weights)

    with tf.get_default_graph().as_default():
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

            # 加载模型
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for i in range(100):  # 1449
                start = time.time()
                sess.run([image, annotation, pred, acc_update_op, miou_update_op])
                print('{}: cost {:.0f}ms'.format(i, (time.time() - start) * 1000))
            acc_res = sess.run(acc)
            miou_res = sess.run(miou)
            print("Pascal VOC 2012 validation dataset pixel accuracy: " + str(acc_res))
            print("Pascal VOC 2012 validation dataset Mean IoU: " + str(miou_res))

            coord.request_stop()
            coord.join(threads)
    pass

if __name__ == '__main__':
    tf.app.run()
