from PIL import Image
import numpy as np
import cv2
import os
import pickle


def color_map(root, all_file, result_pkl_name="data/color_map.pkl"):
    classes = list(range(0, 21))
    classes.append(255)

    map_dict = dict()
    for file in all_file:
        now_file = os.path.join(root, file)
        png = np.asarray(Image.open(now_file))
        rbg = cv2.imread(now_file)
        for i in classes:
            idx = np.where(png == i)
            if len(idx[0]) == 0:
                continue
            else:
                color = rbg[idx[0][0], idx[1][0], :]
                map_dict[str(i)] = color
        if len(map_dict) == 22:
            break
    with open(result_pkl_name, "wb") as f:
        pickle.dump(map_dict, f)
    pass


def test(pkl_file_name, file_name, result_image_name="data/tmp.png"):
    with open(pkl_file_name, "rb") as f:
        d = pickle.load(f)

    img = np.array(Image.open(file_name))
    color_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            color_img[i, j, :] = d[str(img[i][j])]
    Image.fromarray(color_img).convert("RGB").save(result_image_name)
    pass


if __name__ == '__main__':
    root = 'C:\\ALISURE\\Data\\voc\\VOCdevkit\\VOC2012\\SegmentationClass\\'
    result_pkl_name = "data/color_map.pkl"
    all_file = os.listdir(root)
    color_map(root, all_file, result_pkl_name)
    test(result_pkl_name, os.path.join(root, all_file[0]))
