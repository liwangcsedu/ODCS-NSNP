import random  # 导入随机数模块
import os  # 导入操作系统模块
import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.utils.data import DataLoader  # 导入PyTorch的数据加载模块
import torchvision.transforms as transforms  # 导入PyTorch的图像转换模块
import torch.optim as optim  # 导入PyTorch的优化器模块
from matplotlib import pyplot as plt  # 导入Matplotlib库，用于绘图
import PIL.Image as Image  # 导入PIL库，用于图像处理
import torch.nn.functional as F  # 导入PyTorch的函数模块
import datetime  # 导入datetime模块，用于处理日期和时间
from torch.utils.data.sampler import SubsetRandomSampler  # 导入PyTorch的子集随机采样器
from read_testdata import Images_Dataset, Images_Dataset_folder_pre  # 导入自定义的数据集类
import cv2  # 导入OpenCV库，用于图像处理
from models.ODCS_NSNP import ODCS_NSNP  # 导入自定义的ODCS_NSNP模型

# 设置随机种子的函数
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# 判断是否有GPU可用，如果有则使用GPU，否则使用CPU
device = torch.device('cuda',1)

gamma = 2  # 定义gamma值
alpha = 0.25  # 定义alpha值
reduction = 'elementwise_mean'  # 定义reduction方式
set_seed()  # 设置随机种子
dir_checkpoint = '../checkpoints/'  # 定义检查点保存的目录

# 设置训练的最大epoch数量、批次大小、学习率等参数
MAX_EPOCH = 1000
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
shuffle = True
num_workers = 2
random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

# 可视化数量
vis_num = 10

# 数据路径和保存路径
valid_size = 0.
data_path = './data/crop400/data/train/'
test_dir = './data/crop400/data/train/'
save_path_img = '../data/crop400/train_first/'

# 构建测试数据集实例
Testing_Data = Images_Dataset_folder_pre(test_dir)
num_train = len(Testing_Data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

test_idx = indices[split:]
test_sampler = SubsetRandomSampler(test_idx)

# 构建数据加载器
test_loader = DataLoader(Testing_Data, batch_size=1, sampler=test_sampler, num_workers=0)
print('Test data number are %d' % len(test_loader))
print('DataLoader Done')

# 加载模型
net = ODCS_NSNP(3, 3)  # 构建ODCS_NSNP模型
net.to(device)  # 将模型移动到对应的设备

path_checkpoint = './checkpoints/REFUGE_train_599_epoch.pkl'  # 检查点文件路径
checkpoint = torch.load(path_checkpoint, map_location='cpu')  # 加载检查点文件
net.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
net.eval()  # 设置模型为评估模式

# 使用无梯度计算的上下文管理器，遍历测试数据集进行推理
with torch.no_grad():
    for i, data in enumerate(test_loader):  # 遍历测试数据集
        inputs, image_name, img_size = data  # 获取输入数据、图像名称和图像大小
        x_1 = img_size[0]  # 图像大小的第一个维度
        x_2 = img_size[1]  # 图像大小的第二个维度
        inputs = inputs.to(device)  # 将输入数据移动到对应的设备上
        image_name = image_name[0]  # 图像名称
        output = net(inputs)  # 使用模型进行推理得到输出

        output = output.to('cpu')  # 将输出数据移动到CPU上
        output1 = F.relu(output)  # 对输出进行ReLU激活
        n1 = output1.numpy()  # 转换为NumPy数组
        outputs = np.squeeze(output1)  # 压缩维度
        outputs = transforms.ToPILImage()(outputs)  # 转换为PIL图像
        out1 = F.sigmoid(output)  # 对输出进行Sigmoid激活
        #21 = out1.numpy()  # 转换为NumPy数组
        out2 = np.squeeze(out1)  # 压缩维度
        n22 = out2.numpy()  # 转换为NumPy数组
        target = np.rint(n22)  # 四舍五入
        target = target.astype(np.uint8)  # 转换数据类型
        new_label = np.zeros(target.shape, dtype=np.int64)  # 创建全零数组
        new_label[target == 1] = 255  # 根据条件修改数组值

        target = np.mean(new_label, axis=0).astype(np.uint8)    # 转换数据类型和通道
        target = cv2.resize(target, (int(x_1), int(x_2)), interpolation=cv2.INTER_CUBIC)  # 调整图像大小

        cv2.imwrite(save_path_img + image_name[:-4] + '.bmp', target)  # 保存图像
        print(i)  # 打印索引
