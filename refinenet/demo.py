import time
import os
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf
import model as model
from utils.pascal_voc import pascal_segmentation_lut
from utils.visualization import visualize_segmentation_adaptive

tf.app.flags.DEFINE_string('test_data_path', 'demo', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_integer('num_classes', 21, '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints/', '')
tf.app.flags.DEFINE_string('result_path', 'result/', '')

FLAGS = tf.app.flags.FLAGS


# 得到图片的文件名
def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


# resize
def resize_image(im, size=32, max_side_len=2400):
    h, w, c = im.shape
    resize_w = w
    resize_h = h
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    resize_h = resize_h if resize_h % size == 0 else (resize_h // size) * size
    resize_w = resize_w if resize_w % size == 0 else (resize_w // size) * size
    im = np.asarray(Image.fromarray(im).convert("L" if c == 1 else "RGB").resize((int(resize_w), int(resize_h))))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return im, (ratio_h, ratio_w)


def main(argv=None):
    if os.path.exists(FLAGS.result_path):
        shutil.rmtree(FLAGS.result_path)
    os.makedirs(FLAGS.result_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    pascal_voc_lut = pascal_segmentation_lut()

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        logits = model.model(input_images, is_training=False)
        pred = tf.argmax(logits, axis=3)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # 加载模型
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            image_name_list = get_images()
            for image_name in image_name_list:
                im = np.asarray(Image.open(image_name))[:, :, 0: 3][:, :, ::-1]
                print(im.shape)
                im_resized, (ratio_h, ratio_w) = resize_image(im, size=32)

                # 预测
                start = time.time()
                pred_re = sess.run(pred, feed_dict={input_images: [im_resized]})

                # 保存
                img = visualize_segmentation_adaptive(pred_re[0], pascal_voc_lut)
                Image.fromarray(img).convert("RGB").save(os.path.join(FLAGS.result_path, os.path.basename(image_name)))
                print('{}: cost {:.0f}ms'.format(image_name, time.time() - start))
    pass


if __name__ == '__main__':
    tf.app.run()
