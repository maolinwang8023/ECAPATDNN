# 目录

<!-- TOC -->

- [目录](#目录)
- [ECAPA-TDNN说明](#ECAPA-TDNN说明)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [单卡训练](#单卡训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# ECAPA-TDNN说明

ECAPA-TDNN是基于时延神经网络（TDNN）进行了改进，主要有三个方面的优化，分别是增加了一维SE残差模块、多特征融合以及通道和上下文相关的统计池化。ECAPA-TDNN是发表于2020年5月的文章，在Voxceleb数据集上中取得当时最优的结果。
[论文](https://arxiv.org/pdf/2005.07143.pdf)：ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification

# 模型架构

ECAPA-TDNN由多个SE-Res2Block模块串联起来，可以更加深入。SE-Res2Block的基本卷积单元使用和传统tdnn模块相同的1d卷积和dilation参数。模块一般包括**1×1卷积**、**3×1卷积**和**SE-block**以及**Res2net**结构。

# 数据集

## 使用的数据集：[voxceleb](<https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html>)

- 数据集大小：

    - 训练数据集：Voxceleb2，超过一百万条语音，总时长2000+，约240G
    - 测试数据集：Voxceleb1-test，约1.3G
    - 增强数据集：MUSAN 约10G 和 RIR 约1.2G

- 数据格式：训练数据和测试数据都为wav格式，voxceleb2为m4a格式，需转成wav格式

- 准备数据：voxceleb2语音文件为m4a格式，需要转成wav格式才能加入训练。请按照以下流程准备数据：
    - 下载数据集。脚本可参考：https://github.com/clovaai/voxceleb_trainer/blob/master/dataprep.py

    - 转换vox2数据格式，脚本可参考：https://github.com/clovaai/voxceleb_trainer/blob/master/dataprep.py

    - 把voxceleb2的训练集作为训练数据集，拷贝[测试使用的train.txt文件](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt) 到voxceleb2/train.txt

    - 使用voxceleb1-test作为测试数据集，拷贝[测试使用的eval.txt文件](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt) 到voxceleb2/eval.txt

    - 数据集目录结构如下：

      ``` bash
      dataset
      ├── Voxceleb2
      |   ├──train                # 训练集
      |   |   └── wav
      |   |   |   ├── id00012
      |   |   |   |   ├── 2DLq_Kkc1r8
      |   |   |   |   ├── 21Uxsk56VDQ
      |   |   |   |   └── ...
      |   |   |   ├── id00015
      |   |   |   |   ├── 0fijmz4vTVU
      |   |   |   |   ├── 0iQWqFw6FOU
      |   |   |   |   └── ...
      |   |   |   ├── ...
      |   ├──eval                 # 测试集
      |   |   └── wav
      |   |   |   ├── id10270
      |   |   |   |   ├── 5r0dWxy17C8
      |   |   |   |   ├── 5sJomL_D0_g
      |   |   |   |   └── ...
      |   |   |   ├── id10271
      |   |   |   |   ├── 1gtz-CUIygI
      |   |   |   |   ├── 8gcdEYAKNkE
      |   |   |   |   └── ...
      |   |   |   └── ...
      |   ├──train.txt            # 训练文件
      |   └──eval.txt             # 测试文件
      └── Others                  # 数据增强
          ├── musan
          │   ├── music
          │   ├── noise
          │   └── noise
          ├── musan_split
          │   ├── music
          │   ├── noise
          │   └── noise
          └── RIRS_NOISES
              └── simulated_rirs
      ```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend或GPU）
    - 使用Ascend或GPU处理器来搭建硬件环境。
- 框架
    - python3及其它依赖安装包
        - 安装完python3后，执行命令 `pip3 install -r requirements.txt``
        - 其中librosa库主要用来从wav语音中提取melspectrogram特征。
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

配置完环境后，您可以按照如下步骤进行训练和评估：

  ```text
  # src/config.py中修改数据路径
  train_list: "/home/abc000/dataset/Voxceleb2/train.txt"                # 训练数据集文件
  train_path: "/home/abc000/dataset/Voxceleb2/train/wav"                # 训练数据集存储路径
  musan_path: "/home/abc000/dataset/Others/musan_split"                 # 增强数据集存储路径
  rir_path: "/home/abc000/dataset/Others/RIRS_NOISES/simulated_rirs"    # 增强数据集存储路径
  eval_list: "/home/abc000/dataset/Voxceleb2/eval.txt"                  # 训练数据集文件
  eval_path: "/home/abc000/dataset/Voxceleb2/eval/wav"                  # 训练数据集存储路径
  model_path: "/home/abc000/model/pretain.ckpt"                         # 推理使用的ckpt路径
  ```

  ```bash
  # 运行训练示例
  python train.py > train.log 2>&1 &

  # 运行推理示例
  python eval.py > eval.log 2>&1 &

  # 使用脚本的单卡训练
  bash run_standalone_train_ascend.sh DEVICE_ID

  # 使用脚本的分布式训练
  bash run_distribute_train_ascend.sh RANK_TABLE_FILE

  # 运行推理示例
  bash run_eval_ascend.sh DEVICE_ID PATH_CHECKPOINT
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>.

# 脚本说明

## 脚本及样例代码

```bash
    ECAPATDNN
    ├── ascend310_infer                              # 310 推理代码
    ├── scripts                                      # 训练和测试相关的shell脚本
    ├── src                                          # mindspore模型相关代码
    ├── eval.py                                      # 测试脚本
    ├── export.py                                    # 转出310模型脚本
    ├── postprocess.py                               # 310计算精度脚本
    ├── preprocess.py                                # 310数据预处理脚本
    ├── README_CN.md                                 # ecapa-tdnn相关说明
    ├── requirements.txt                             # python依赖包
    └── train.py                                     # 训练脚本
```

## 脚本参数

在src/config.py中可以同时配置训练参数和评估参数。

```text
  in_channels: 80                                            # 输入层特征通道数
  channels: 1024                                             # 中间层特征图的通道数
  emb_size: 192                                              # embedding 维度
  base_lrate: 0.000001                                       # cyclic LR学习策略的基础学习率
  max_lrate: 0.0001                                          # cyclic LR学习策略的最大学习率
  weight_decay: 0.000002                                     # 优化器参数
  num_epochs: 80                                             # 训练epoch数
  minibatch_size: 100                                        # batch size
  class_num: 5944                                            # voxceleb2 的说话人个数
  ckpt_save_dir: "/home/abc000/model/"                       # 模型训练的输出目录
  keep_checkpoint_max: 80                                    # 最大保存模型数
  length: 1199                                               # 固定长度
  train_list: "/home/abc000/dataset/Voxceleb2/train.txt"     # 训练数据集文件
  train_path: "/home/abc000/dataset/Voxceleb2/train/wav"     # 训练数据集存储路径
  musan_path: "/home/abc000/dataset/Others/musan_split"      # 增强数据集存储路径
  eval_list: "/home/abc000/dataset/Voxceleb2/eval.txt"       # 训练数据集文件
  eval_path: "/home/abc000/dataset/Voxceleb2/eval/wav"       # 训练数据集存储路径
```

## 训练过程

### 单卡训练

- Ascend处理器环境运行

  ```bash
  # 执行python脚本开始训练
  python train.py > train.log 2>&1 &
  # 执行shell脚本开始训练
  bash run_standalone_train_ascend.sh DEVICE_ID
  ```

### 分布式训练

  ```bash
  bash scripts/run_distribute_train_ascend.sh RANK_TABLE_FILE
  ```

上述python命令将在后台运行，您可以通过日志文件train0.log查看结果。

## 评估过程

### 评估

- 在Ascend环境运行时评估voxceleb2数据集

  ```bash
  # 执行python脚本开始推理
  python3 eval.py > eval.log 2>&1 &
  # 执行shell脚本开始推理
  bash run_eval_ascend.sh DEVICE_ID PATH_CHECKPOINT
  ```

  上述python命令将在后台运行，您可以通过日志文件eval.log查看结果。测试数据集的准确性如下：

  ```bash
  # cat eval.log | grep EER
  EER = 1.43%
  ```

## 导出过程

### 导出

```bash
python export.py  --checkpoint_path [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

## 推理过程

### 推理

在推理之前我们需要先导出模型。mindir可以在任意环境上导出。batch_size只支持1。

- 在昇腾310上使用voxceleb2数据集进行推理

  由于310只支持固定长度推理，所以我们的310推理代码中会把测试语音截取固定长度，整体结果要差于完整语音的推理结果，完整语音的推理结果可参考上文910评估结果。推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```bash
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
  # example: bash run_infer_310.sh /path/ecapatdnn.mindir /path/feat_eval 0
  cat acc.log | grep eer
  EER = 1.43%
  ```

# 模型描述

## 性能

### 训练性能

| 参数                       | Ascend|
| -------------------------- | -----------------------------------------------------------|
| 模型版本                   | ECAPA-TDNN|
| 资源                   | Ascend 910；系统 Euler2.0 |
| 上传日期              | 2022-09-28 |
| MindSpore版本          | 1.3.0 |
| 数据集                    | voxceleb2 |
| 训练参数        | epoch=80,  batch_size = 100, min_lr=0.000001, max_lr=0.0001 |
| 优化器                  | Adam|
| 损失函数              | AAM-Softmax交叉熵|
| 输出                    | 概率|
| 损失                       | 1.3|
| 总时长                 | 单卡：120小时 |
| 微调检查点 | 247M (.ckpt文件) |
| 推理模型        |  76.60M(.mindir文件)|

### 评估性能

#### voxceleb1上评估ECAPA-TDNN

| 参数          | Ascend|
| ------------------- | ---------------------------|
| 模型版本       | ECAPA-TDNN|
| 资源            |  Ascend 910；系统 Euler2.8|
| 上传日期       | 2021-09-28 |
| MindSpore 版本   | 1.3.0|
| 数据集             | voxceleb1-eval, 4715条语音|
| batch_size          | 1|
| 输出             | 概率|
| 准确性            | 单卡: EER=1.43%; |
| 推理模型 | 76.60M(.mindir文件)        |

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
