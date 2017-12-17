from utils.pascal_voc import get_pascal_image_annotation_filename_pairs
from utils.tfrecords import write_image_annotation_pairs_to_tfrecord

pascal_root = 'C:\\ALISURE\\Data\\voc\\VOCdevkit\\VOC2012\\'

# Returns a list of (image, annotation) filename pairs (filename.jpg, filename.png)
train_filename_pairs, val_filename_pairs, train_val_filename_pairs = \
    get_pascal_image_annotation_filename_pairs(pascal_root=pascal_root)

# You can create your own tfrecords file by providing your list with (image, annotation) filename pairs here
write_image_annotation_pairs_to_tfrecord(val_filename_pairs, tfrecords_filename='pascal_val.tfrecords')
write_image_annotation_pairs_to_tfrecord(train_filename_pairs, tfrecords_filename='pascal_train.tfrecords')
write_image_annotation_pairs_to_tfrecord(train_val_filename_pairs, tfrecords_filename='pascal_train_val.tfrecords')
