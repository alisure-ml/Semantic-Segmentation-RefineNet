### RefineNet

* 将粗糙的高层语义特征和细粒度的底层特征融合。


#### Paper
* [RefineNet - Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](paper/RefineNet%20-%20Multi-Path%20Refinement%20Networks%20for%20High-Resolution%20Semantic%20Segmentation.pdf)


#### Prepare
- download the pretrain model of [resnet_v1_101.ckpt](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz)
- download the dataset of [pascal voc dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- download the RefineNet [model](http://pan.baidu.com/s/1kVefEIj), `eragonruan` trained on pascal voc

#### Code

1. 直接执行的文件说明（按顺序执行即可）：
    * `build_color_map.py` 得到类别和颜色对应的字典
    * `convert_pascal_voc_to_tfrecords.py` 将数据变成tfrecord格式
    * `train.py` 训练，读取数据并进行训练
    * `demo.py` 测试，对`test_data_path`的所有图片进行测试

2. 其它文件说明：
    * `model.py` 模型，定义了RefineNet
    * `resnet` 残差网络
    * `utils` 工具函数
        * `augmentation.py` 训练时对数据进行增强
        * `pascal_voc.py` 获取voc图片path
        * `tfrecords.py` 对数据进行转换
        * `training.py` 训练时处理label的相关函数
        * `visualization.py` 对测试结果进行可视化

#### Reference
* [eragonruan/refinenet-image-segmentation](https://github.com/eragonruan/refinenet-image-segmentation)
