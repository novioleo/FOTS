# FOTS
[TOC]

本项目基于[Ning Lu](https://github.com/jiangxiluning)以及[DongLiang Ma](https://vipermdl.github.io/)还有其他优秀的开源项目的共同实现的FOTS。**感恩所有开源项目的贡献者。**

## 识别样张【持续更新】

### 场景1：特定部分文本

![1557730185462](./imgs/result_example/scenario_1/1.png)

![1557730365238](./imgs/result_example/scenario_1/2.png)

## HOW TO RUN
### 全局参数配置

|       字段        |         作用         |          参考值          |                      备注                      |
| :---------------: | :------------------: | :----------------------: | :--------------------------------------------: |
|       name        |       项目名称       | FOTS_2019-05-11_自有数据 |                   你高兴就好                   |
|       cuda        |      是否用显卡      |           true           |                 有的话还是用吧                 |
|       gpus        |  配置多显卡并行训练  |           [0]            |      `nvidia-smi` 看下自己想用的显卡的id       |
|     finetune      |       finetune       |            ""            |    finetune的模型一定要是同mode的，不然报错    |
|    data_loader    |      数据集配置      |       详情查看下方       |                                                |
|    validation     |      验证集配置      |       详情查看下方       |                                                |
| lr_scheduler_type |    学习率调度类型    |      ExponentialLR       |         线性、指数、自定义，你高兴就好         |
| lr_scheduler_freq |    学习率调度频率    |            50            |           根据模型的收敛速度自行调整           |
|   lr_scheduler    | 学习率调度函数的参数 |                          |          根据调度器类型不同，自行传参          |
|  optimizer_type   |      优化器类型      |           Adam           |            PyTorch支持的优化器类型             |
|     optimizer     |    优化器具体参数    |                          |          根据优化器类型不同，自行传参          |
|       loss        |       损失函数       |         FOTSLoss         |              目前只有支持这个loss              |
|      metrics      |       度量函数       |       fots_metrics       | recognition_metric和detection_metric未去做实现 |
|      trainer      |      训练器参数      |       详情查看下方       |                                                |
|       arch        |       算法模型       |        FOTSModel         |               目前只有FOTSModel                |
|       model       |   FOTSModel的参数    |       详情查看下方       |               **会更新的很频繁**               |

### 数据集配置 【data_loader】

|字段|作用|参考值|备注|
|:--:|:--:|:--:|:--:|
|dataset|选择特定类型的数据集|icdar2015、mydataset|如果自己数据格式比较特殊，在`data_loader\dataset.py`中自行增加|
|data_dir|数据集所在文件夹|trainging_gt,training_images|必须包含图像和ground truth文件夹|
|annotation_dir|标注数据所在文件夹|training_gt|功能还未完善|
|batch_size|batch大小|32|太大的话显存撑不住|
|shuffle|随机排布|true|别管那么多，true就对了|
|workers|配置datasetloader的构建效率|0|1. docker内需要传递环境参数，否则会报错<br />2. 如果get_item的效率比较低，强烈建议多开几个，否则gpu会赋闲。|


#### ICDAR2015

修改配置文件如下：
```json
{
    "data_loader": {
        "dataset":"icdar2015",
        "data_dir": "/mnt/disk1/dataset/icdar2015/4.4/training",
        "batch_size": 16,
        "shuffle": true,
        "workers": 0 
    }
}
```
#### 自有数据
修改配置文件如下：
```json
{
  "data_loader": {
        "dataset":"mydataset",
        "image_dir": "/data/OCR/自有数据/own_dataset/training_images",
        "annotation_dir": "/data/OCR/自有数据/own_dataset/training_gt",
        "batch_size": 4,
        "shuffle": true,
        "workers": 0
    }
}
```

#### 验证集配置【validation】

```json
"validation": {
    "validation_split": 0.15,
    "shuffle": true
}
```

由于训练和测试是采用固定比例的方式，其中`validation_split`表示测试集所占比例，`shuffle`为是否重排。

#### 训练器参数【trainer】

```json
"trainer": {
    "epochs": 10000,
    "save_dir": "/path/to/save_model",
    "save_freq": 1,
    "verbosity": 2,
    "monitor": "loss",
    "monitor_mode": "min"
}
```

训练器参数中`epochs`表示总的训练回合数，`save_dir`表示模型存储的位置，最终模型所在位置为`save_dir/name`下面，其中`name`为全局变量中的项目名称。`save_freq`表示每N个epoch存储一次模型。`verbosity`为设置logger显示等级，`monitor`与`monitor_mode`是为了生成最优模型`model_best.pth.tar`文件而生的，在默认的信息输出默认为：

- [val_]loss 全局损失
- [val_]det_loss 文本检测损失
- [val_]rec_log 文本识别损失
- [val_]precious `metric`中定义的精度
- [val_]recall `metric`中定义的召回率
- [val_]hmean `metric`中定义的Fscore

示例中为希望`loss`越小越好。当然也可以设置为`precious`越大越好，即：`monitor`为`precious`，`monitor_mode`为`max`。

#### FOTSModel参数【model】

```json
"model": {
    "mode": "united",
    "scale": 512,
    "crnn": {
        "img_h": 16,
        "hidden": 1024
    },
    "keys": "number_and_dot"
}
```

> NOTE
>
> 这块还没有完善，可以根据个人需要自己定制。

`mode`有三个模式可以选，分别为：`recognition`只进行识别模型的训练，`detection`只进行检测模型的训练，`united`检测和是别一起训练。如果是需要测试某个单一模块，那么可以自行选择是检测还是识别，默认是一起训练。

`scale`参数暂时还未适配完成，后面会用于调整识别画幅大小。

`crnn`中的`img_h`为`ROIRotate`后传入CRNN的模型的FeatureMap的高度，**此处必须是8的倍数**。`hidden`为`crnn`中BiLSTM中的隐层的个数，具体参数自行调整。

`keys`为当前识别所用到的字符集，如果需要添加或查看已有字符集，请移步：[common_str.py](./utils/common_str.py)

### 训练

`python train.py -c \path\to\your\config.json`

### 评估

`python eval.py -m \path\to\your\model.pth.tar  -i \path\to\eval\images -o \path\to\output\result`

### gRPC服务【coming soon】

`python service\serve.py -m \path\to\your\model.pth.tar -p port_number` 

## 原理解释

>  **NOTE**
>
> 本项目已经跟原论文有一定差异了，为了更好收敛模型做的各种调整，以适用于实际场景。而且到后面可能都不是FOTS，所以请大家不要纠结是否跟原论文一致。效果好就行了。

### 网络结构图
![网络结构图](./imgs/fots.jpg)

本质上来说当前FOTS是升级版east+crnn的实现。与普通的两个模型简单粗暴的合在一起不一样，FOTS是把两个模型放到同一个大模型里面了，也就是梯度下降能应用于两个部分。模型结构如下图所示：

> **NOTE**
>
> 其中论文中类FPN的部分是det和rec共享，但是在训练的时候发现只能方便det或者rec进行收敛，如果要保证这个部分收敛，那么就需要更大的参数规模，所以为了方便起见，这里直接使用两个类FPN的部分，分别用于det和rec，这样模型更容易收敛。

其中det部分训练就是普通的目标检测模型的训练。rec的部分是将ground-truth中的rbox区域的FeatureMap进行ROIRotate（本质上来说就是完成了仿射变换），然后做为CRNN的输入。

训练的时候det部分将得到的rbox，对rec的类FPN的FeatureMap的rbox区域进行ROIRotate，然后传入CRNN。

整个模型总的来说还是算OneStage。

## TODO

- [ ] 利用apex，将模型从float32变换到float16
- [ ] CRNN的lstm的step参数可调，使得rec部分可以支持cnn+ctc部分
- [ ] 超长文本识别
- [ ] 适配TextSnake