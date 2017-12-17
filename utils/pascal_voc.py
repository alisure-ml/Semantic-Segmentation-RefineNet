import os


# 返回数字和类别的字典：number and correspondng class
def pascal_segmentation_lut():
    """Return look-up table with number and correspondng class names
    for PASCAL VOC segmentation dataset. Two special classes are: 0 -
    background and 255 - ambigious region. All others are numerated from
    1 to 20.
    
    Returns
    -------
    classes_lut : dict
        look-up table with number and correspondng class names
    """

    class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'potted-plant',
                   'sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']
    classes_lut = list(enumerate(class_names[:-1]))
    # Add a special class representing ambigious regions which has index 255.
    classes_lut.append((255, class_names[-1]))
    return dict(classes_lut)


# 获得图片的list
def get_pascal_segmentation_images_lists_txts(pascal_root):
    segmentation_images_lists_folder = os.path.join(pascal_root, 'ImageSets/Segmentation')
    pascal_train_list_filename = os.path.join(segmentation_images_lists_folder, 'train.txt')
    pascal_validation_list_filename = os.path.join(segmentation_images_lists_folder, 'val.txt')
    pascal_train_val_list_filename = os.path.join(segmentation_images_lists_folder, 'trainval.txt')
    return [pascal_train_list_filename, pascal_validation_list_filename, pascal_train_val_list_filename]


# 获取每一个list中的每一行
def readlines_with_strip_array_version(filenames_array):
    def readlines_with_strip(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        return map(lambda x: x.strip(), lines)
    return map(lambda x: set(x), map(readlines_with_strip, filenames_array))


# 得到图片和注解对（包括路径）
def get_pascal_selected_image_annotation_filenames_pairs(pascal_root, selected_names):
    """Returns (image, annotation) filenames pairs from PASCAL VOC segmentation dataset for selected names.
    The function accepts the selected file names from PASCAL VOC segmentation dataset
    and returns image, annotation pairs with fullpath and extention for those names.
    Parameters
    ----------
    pascal_root : string
        Path to the PASCAL VOC dataset root that is usually named 'VOC2012'
        after being extracted from tar file.
    selected_names : array of strings
        Selected filenames from PASCAL VOC that can be read from txt files that
        come with dataset.
    Returns
    -------
    image_annotation_pairs : 
        Array with filename pairs with fullnames.
    """
    pascal_images_folder = os.path.join(pascal_root, 'JPEGImages')
    pascal_annotations_folder = os.path.join(pascal_root, 'SegmentationClass')
    images_full_names = map(lambda x: os.path.join(pascal_images_folder, x) + '.jpg', selected_names)
    annotations_full_names = map(lambda x: os.path.join(pascal_annotations_folder, x) + '.png', selected_names)
    return zip(images_full_names, annotations_full_names)


# 返回zip(map(images), map(annotations))
def get_pascal_image_annotation_filename_pairs(pascal_root):
    pascal_txts = get_pascal_segmentation_images_lists_txts(pascal_root=pascal_root)
    pascal_train_name, pascal_val_name, pascal_train_val_name = readlines_with_strip_array_version(pascal_txts)

    # 验证集
    validation = pascal_val_name
    # 验证集和训练集的并集 - 验证集
    train = list(pascal_train_name | pascal_val_name - validation)
    # 验证集和训练集
    train_val = pascal_train_val_name

    train_filename_pairs = get_pascal_selected_image_annotation_filenames_pairs(pascal_root, train)
    val_filename_pairs = get_pascal_selected_image_annotation_filenames_pairs(pascal_root, validation)
    train_val_filename_pairs = get_pascal_selected_image_annotation_filenames_pairs(pascal_root, train_val)

    return train_filename_pairs, val_filename_pairs, train_val_filename_pairs
