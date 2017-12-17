from PIL import Image
import numpy as np
import tensorflow as tf


# Helper functions for defining tf types
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 将数据写入到tfrecord中
def write_image_annotation_pairs_to_tfrecord(filename_pairs, tfrecords_filename):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for img_path, annotation_path in filename_pairs:
        img = np.array(Image.open(img_path))
        annotation = np.array(Image.open(annotation_path))
        # Uncomment this one when working with surgical data
        # annotation = annotation[:, :, 0]
        height = img.shape[0]
        width = img.shape[1]

        img_raw = img.tostring()
        annotation_raw = annotation.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw)
        }))

        writer.write(example.SerializeToString())
    writer.close()
    pass


# 从文件中读取所有的样例
def read_image_annotation_pairs_from_tfrecord(tfrecords_filename):
    image_annotation_pairs = []

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        img_string = (example.features.feature['image_raw'].bytes_list.value[0])
        annotation_string = (example.features.feature['mask_raw'].bytes_list.value[0])

        img = np.fromstring(img_string, dtype=np.uint8).reshape((height, width, -1))
        # TODO: Annotations don't have depth (3rd dimension), check if it works for other datasets
        annotation = np.fromstring(annotation_string, dtype=np.uint8).reshape((height, width))
        image_annotation_pairs.append((img, annotation))
        pass
    return image_annotation_pairs


# 从队列中读取一个样例
def read_tfrecord_and_decode_into_image_annotation_pair_tensors(tfrecord_filenames_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features = tf.parse_single_example(serialized_example, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)
    })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image = tf.reshape(image, shape=[height, width, 3])
    # The last dimension was added because the tf.resize_image_with_crop_or_pad() accepts tensors
    # that have depth. We need resize and crop later.
    # TODO: See if it is necessary and probably remove third dimension
    annotation = tf.reshape(annotation, shape=[height, width, 1])
    
    return image, annotation
