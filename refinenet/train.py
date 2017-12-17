import tensorflow as tf
import time
import shutil
import datetime
import os
import pickle
import numpy as np
from tensorflow.contrib import slim
import model as model
from utils.tfrecords import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from utils.pascal_voc import pascal_segmentation_lut
from utils.augmentation import (distort_randomly_image_color, flip_randomly_left_right_image_with_annotation,
                                scale_randomly_image_with_annotation_with_fixed_size_output)

tf.app.flags.DEFINE_integer('batch_size', 2, '')
tf.app.flags.DEFINE_integer('train_size', 384, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('num_classes', 21, '')
tf.app.flags.DEFINE_string('checkpoint_path', '../checkpoints', '')
tf.app.flags.DEFINE_string('logs_path', '../logs', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 2000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 10, '')
tf.app.flags.DEFINE_string('training_data_path', '../data/pascal_train_val.tfrecords', '')
tf.app.flags.DEFINE_string('pre_train_model_path', '../data/resnet_v1_101.ckpt', '')
tf.app.flags.DEFINE_integer('decay_steps', 40000, '')
tf.app.flags.DEFINE_integer('decay_rate', 0.1, '')
FLAGS = tf.app.flags.FLAGS


# return：模型损失+正则损失，模型损失，预测结果
def tower_loss(images, annotation, class_labels):
    logits = model.model(images, is_training=True)
    pred = tf.argmax(logits, axis=3)

    model_loss = model.loss(annotation, logits, class_labels)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    tf.summary.scalar('model_loss', model_loss)
    tf.summary.scalar('total_loss', total_loss)
    return total_loss, model_loss, pred


def main(argv=None):
    # 类别字典
    pascal_voc_lut = pascal_segmentation_lut()
    # 类别键值
    class_labels = list(pascal_voc_lut.keys())
    # 类别与颜色的对应关系
    with open('data/color_map.pkl', 'rb') as f:
        color_map = pickle.load(f)

    # 日志目录
    style_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(FLAGS.logs_path + style_time)

    # 模型保存目录
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

    # 文件队列
    filename_queue = tf.train.string_input_producer([FLAGS.training_data_path], num_epochs=1000)
    # 解码tf record数据
    image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)
    # 随机左右翻转
    image, annotation = flip_randomly_left_right_image_with_annotation(image, annotation)
    # 随机色彩变换
    image = distort_randomly_image_color(image)
    # 随机缩放
    image_train_size = [FLAGS.train_size, FLAGS.train_size]
    resize_image, resize_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(image, annotation,
                                                                                                  image_train_size)
    # 在读数据的时候，对注解进行了升维。现在进行降维
    resize_annotation = tf.squeeze(resize_annotation)
    # 转成批次
    image_batch, annotation_batch = tf.train.shuffle_batch([resize_image, resize_annotation],
                                                           batch_size=FLAGS.batch_size, capacity=1000,
                                                           num_threads=4, min_after_dequeue=500)

    # 学习率和全局步数
    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # 得到loss和预测
    total_loss, model_loss, output_pred = tower_loss(image_batch, annotation_batch, class_labels)

    # 1.优化损失：loos updates
    gradient_updates_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

    # 2.滑动平均：moving average updates
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 3.BN参数更新：batch norm updates
    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    # 4.合并训练节点
    with tf.control_dependencies([variables_averages_op, gradient_updates_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    summary_writer = tf.summary.FileWriter(FLAGS.logs_path + style_time, tf.get_default_graph())
    # summary：学习率
    tf.summary.scalar('learning_rate', learning_rate)
    # summary：图片
    log_image_data = tf.placeholder(tf.uint8, [None, None, 3])
    log_image_name = tf.placeholder(tf.string)
    log_image = tf.summary.image(log_image_name, tf.expand_dims(log_image_data, 0))
    # 合并summary
    summary_op = tf.summary.merge_all()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        # 恢复模型
        restore_step = 0
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            restore_step = int(ckpt.split('.')[0].split('_')[-1])
            saver.restore(sess, ckpt)
        elif FLAGS.pre_train_model_path is not None:
            # 加载预训练模型
            # Returns a function that assigns specific variables from a checkpoint.
            variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pre_train_model_path,
                                                                 slim.get_trainable_variables(),
                                                                 ignore_missing_vars=True)
            variable_restore_op(sess)
            pass

        start = time.time()
        coord = tf.train.Coordinator()
        # 启动队列
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                for step in range(restore_step, FLAGS.max_steps):
                    # 衰减学习率
                    if step != 0 and step % FLAGS.decay_steps == 0:
                        sess.run(tf.assign(learning_rate, learning_rate.eval() * FLAGS.decay_rate))
                        pass

                    # 执行损失和训练
                    ml, tl, _ = sess.run([model_loss, total_loss, train_op])

                    # 损失发散（不收敛）
                    if np.isnan(tl):
                        print('Loss diverged, stop training')
                        break

                    # 计时并打印信息
                    if step % 10 == 0:
                        print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.3f} seconds/step, lr: {:.7f}'.
                              format(step, ml, tl, (time.time() - start) / 10, learning_rate.eval()))
                        start = time.time()
                        pass

                    # 保存模型
                    if (step + 1) % FLAGS.save_checkpoint_steps == 0:
                        filename = ('RefineNet' + '_step_{:d}'.format(step + 1) + '.ckpt')
                        filename = os.path.join(FLAGS.checkpoint_path, filename)
                        saver.save(sess, filename)
                        print('Write model to: {:s}'.format(filename))

                    # 保存summary
                    if step % FLAGS.save_summary_steps == 0:
                        # 再运行一次
                        img_split, seg_split, pred = sess.run([image_batch, annotation_batch, output_pred])

                        # 降维并取第0个
                        img_split = np.squeeze(img_split)[0]
                        seg_split = np.squeeze(seg_split)[0]
                        pred_split = np.squeeze(pred)[0]

                        # 注解图片
                        color_seg = np.zeros((seg_split.shape[0], seg_split.shape[1], 3))
                        for i in range(seg_split.shape[0]):
                            for j in range(seg_split.shape[1]):
                                color_seg[i, j, :] = color_map[str(seg_split[i][j])]

                        # 预测图片
                        color_pred = np.zeros((pred_split.shape[0], pred_split.shape[1], 3))
                        for i in range(pred_split.shape[0]):
                            for j in range(pred_split.shape[1]):
                                color_pred[i, j, :] = color_map[str(pred_split[i][j])]

                        write_img = np.hstack((img_split, color_seg, color_pred))
                        _, summary_str = sess.run([train_op, summary_op, log_image],
                                                  feed_dict={log_image_name: ('%06d' % step),
                                                             log_image_data: write_img})
                        # 写入summary
                        summary_writer.add_summary(summary_str, global_step=step)

                    pass
                pass
            pass
        except tf.errors.OutOfRangeError:
            print('finish')
        finally:
            coord.request_stop()
        coord.join(threads)
    pass


if __name__ == '__main__':
    tf.app.run()
