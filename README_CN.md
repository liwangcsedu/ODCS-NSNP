# Paper Repository.

## ODCS-NSNP: Optic disc and cup segmentation using deep networks enhanced by nonlinear spiking neural P systems

## 项目概述

ODCS-NSNP (Optic disc and cup segmentation using deep networks enhanced by nonlinear spiking neural P systems) 是一个基于深度学习的医学图像分割模型，专注于眼底图像中光学视盘(OD)和视杯(OC)的自动分割。该模型使用了先进的神经网络结构和非局部感知技术，能够准确地识别和分割眼底图像中的关键结构，对青光眼等眼部疾病的临床诊断具有重要意义。

## 模型结构

该项目实现了ODCS-NSNP模型，主要包含以下创新点：

1. **SDN单元(Structural Deep Network Units)**: 使用可分离卷积和密集连接结构，提高特征提取能力
2. **SRS操作器(Scale Reconfiguration Structure Operator)**: 替代传统的池化和上采样操作，更好地保留空间信息
3. **多尺度特征融合**: 通过多层次特征提取和融合实现精确分割

模型采用编码器-解码器结构，通过多种损失函数(交叉熵损失、Focal Loss、Dice Loss)的组合优化训练过程。

## 项目结构

```
ODCS-NSNP/
├── train.py              # 训练主程序
├── test.py              # 测试主程序
├── run.sh                # 训练启动脚本
├── models/               # 模型定义
│   ├── ODCS_NSNP.py      # 主网络模型
│   └── SRS_Operator.py   # 尺度重构结构操作器
├── tools/                # 工具库
│   ├── datasets.py       # 数据集加载
│   ├── preprocessing.py  # 数据预处理
│   ├── evaluation_metrics_for_segmentation.py  # 评估指标
│   ├── evaluate.py       # 模型评估函数
│   ├── post_processing.py # 后处理
│   └── ...               # 其他工具函数
├── data/                 # 数据目录
│   ├── Drishti-files/    # Drishti眼底图像数据集
│   ├── Rim-one-r3/       # RIM-ONE眼底图像数据集
│   └── Refuge/           # REFUGE眼底图像数据集
└── checkpoints/          # 模型检查点
```

## 数据集

本项目支持以下眼底图像数据集：

1. **Drishti-GS**: 印度Aravind眼科医院收集的高分辨率眼底图像数据集，包含50张训练图像和51张测试图像，每张图像都有详细的光学视盘和视杯分割标注。

2. **RIM-ONE**: 由西班牙研究团队提供的用于青光眼诊断的视网膜图像数据集，包含有不同级别的青光眼病变。

3. **REFUGE**: 由MICCAI 2018挑战赛发布的眼底图像数据集，专注于青光眼评估。

本项目使用的数据集已经过预处理，以适应模型训练需求。数据通常按以下结构组织：

```
data/
├── Drishti-files/
│   ├── Training/
│   │   └── labels/
│   └── Test/
├── Rim-one-r3/
└── Refuge/
```

在训练前，您可能需要使用tools目录下的预处理工具来准备您自己的数据。

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- CUDA 10.2+ (用于GPU训练)
- 其他依赖包：
  - numpy
  - matplotlib
  - scikit-image
  - tqdm
  - torchvision
  - PIL
  - torchstat
  - opencv-python (cv2)

## 安装

1. 克隆项目仓库：
```bash
git clone https://github.com/liwangcdedu/ODCS-NSNP.git
cd ODCS-NSNP
```

2. 创建虚拟环境并安装依赖：
```bash
conda create -n odcs-nsnp python=3.8
conda activate odcs-nsnp
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-image tqdm pillow torchstat opencv-python
```

## 使用方法

### 数据预处理

在开始训练前，您可能需要对数据进行预处理。使用tools文件夹中的预处理工具：

```bash
python tools/preprocessing.py
```

也可以使用其他预处理工具，如：
```bash
python tools/crop.py   # 裁剪图像
python tools/resize.py # 调整图像大小
```

### 训练模型

可以通过以下命令开始训练：

```bash
python train.py
```

或者使用提供的脚本直接在后台运行：

```bash
bash run.sh
```

训练参数可在train.py文件中进行调整，包括：
- 批量大小
- 学习率
- 训练轮数
- 正则化参数
- 数据集路径

### 模型评估

训练过程中会自动在每个epoch结束后对模型进行评估。您也可以使用tools/evaluate.py中的评估函数对模型进行单独评估：

```bash
# 评估特定数据集上的性能
python -c "from tools.evaluate import evaluate_segmentation_results; evaluate_segmentation_results('模型名称', '预测路径', '真实标签路径')"
```

评估指标包括：
- Dice系数(DC)
- Jaccard系数(JAC)
- 精确度(ACC)
- 敏感性(SEN)
- 特异性(SPC)
- 杯盘比(CDR)

## 结果可视化

训练过程中会生成TensorBoard日志，可以使用以下命令查看训练曲线：

```bash
tensorboard --logdir=runs
```

## 模型检查点

训练过程中的模型检查点会保存在checkpoints目录下。您可以使用这些检查点继续训练或进行推理。

## 引用

```
@article{li2025odcs,
  title={ODCS-NSNP: Optic disc and cup segmentation using deep networks enhanced by nonlinear spiking neural P systems},
  author={Li, Wang and Xia, Meichen and Peng, Hong and Liu, Zhicai and Guo, Jun},
  journal={Biomedical Signal Processing and Control},
  volume={108},
  pages={107935},
  year={2025},
  publisher={Elsevier}
}
```