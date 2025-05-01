import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from matplotlib import pyplot as plt
import datetime
from torch.utils.data.sampler import SubsetRandomSampler
from tools.datasets import Images_Dataset_folder
from torch.utils.tensorboard import SummaryWriter
import os
from models.ODCS_NSNP import ODCS_NSNP
from test import test_model_and_evaluate
from tqdm import tqdm
import torchstat
import warnings

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

# Binary Focal Loss function
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

# Model weight regularization class
class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")

# Set random seed
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Define Dice Loss function
def dice_loss(input, target):
    smooth = 1.
    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    return 1.0 - (((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)))

# Set random seed
set_seed()

# Use GPU if available, otherwise use CPU
device = torch.device('cuda',2)

# Directory for model checkpoints
dir_checkpoint = './checkpoints/'

# Parameters
MAX_EPOCH = 30
BATCH_SIZE = 8
LR = 0.00001
log_interval = 10
val_interval = 1
shuffle = True
num_workers = 0
random_seed = random.randint(1, 100)

# Visualization count
vis_num = 10

# Regularization parameter
weight_decay = 0

# Training data and label paths
train_dir = './data/Rimone/od/train/images/'
label_dir = './data/Rimone/od/train/labels/'
valid_size = 0.1

# Build training dataset
Training_Data = Images_Dataset_folder(train_dir, label_dir)

num_train = len(Training_Data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

# Shuffle data randomly
if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

# Split training and validation indices
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Build data loaders
train_loader = DataLoader(Training_Data, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=num_workers)
valid_loader = DataLoader(Training_Data, batch_size=BATCH_SIZE, sampler=valid_sampler, num_workers=num_workers)

# Build model
net = ODCS_NSNP(3, 3)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Move model and loss function to GPU
criterion.to(device)
net.to(device)

# Define optimizer
optimizer = optim.Adam(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

# Regularization
if weight_decay > 0:
    reg_loss = Regularization(net, weight_decay, p=1).to(device)
else:
    print("no regularization")

# Training loop
train_curve = list()
valid_curve = list()
train_dice_curve = list()
valid_dice_curve = list()
writer = SummaryWriter(comment='train_comment', filename_suffix='_train_suffix')

# Initialize variables to track previous model evaluation results
previous_mean_dc = 0.0
start_epoch = 0

# Checkpoint saving during training
checkpoint_path = './checkpoints/checkpoint_Rim_to_Dri_OD.pth'
if os.path.exists(checkpoint_path):
    # Load previous checkpoint
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print("Checkpoint loaded! Resuming training from epoch {}".format(start_epoch))
else:
    start_epoch = 0

# Start training
for epoch in range(MAX_EPOCH):
    if  epoch < start_epoch:    # Skip epochs that have already been trained
        continue
    print('Epoch {}/{}'.format(epoch + 1, MAX_EPOCH))

    train_loss_total = 0.
    train_dice_total = 0.
    valid_loss_mean = 0.
    loss_mean = 0.
    correct = 0.
    step = 0

    start = datetime.datetime.now()
    net.train()

    bar = tqdm(train_loader)
    for i, data in enumerate(bar):
        step += 1
        inputs, labels = data

        labels = torch.squeeze(labels, 1)
        n1 = labels.numpy()
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward propagation
        outputs = net(inputs)

        # Calculate accuracy
        pred = torch.max(outputs, 1)[1]
        correct = (pred == labels).sum().item()
        total = np.prod(pred.size())
        train_acc = (correct / total)

        # Backward propagation
        optimizer.zero_grad()
        loss = criterion(outputs, labels)

        if weight_decay > 0:
            loss = loss + reg_loss(net)
        loss.backward()

        # Update weights
        optimizer.step()

        # Print training information
        train_loss_total += loss.item()
        train_curve.append(loss.item())
        if i % 10 == 0:
            writer.add_scalar('Train', loss.item(), epoch)
        if (i + 1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
        # Use tqdm to display training progress
        bar.set_description(f'Train mloss:{train_loss_total / (i + 1):.4f} ACC:{train_acc:.5f}')

    end = datetime.datetime.now()
    print(f"Running Time：{(end - start).seconds}s")

    scheduler.step()
    with SummaryWriter(comment='Net1') as w:
        w.add_graph(net, (inputs,))

    # Validate model
    if (epoch + 1) % val_interval == 0:
        net.eval()
        valid_loss_total = 0.
        valid_dice_total = 0.
        with torch.no_grad():
            for j, data in enumerate(tqdm(valid_loader, desc=f'Validation {epoch+1}/{MAX_EPOCH}')):
                inputs, labels = data
                labels = torch.squeeze(labels)
                if BATCH_SIZE == 1:
                    labels = torch.unsqueeze(labels, 0)

                labels = labels.long()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward propagation
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                pred = torch.max(outputs, 1)[1]
                valid_loss_total += loss.item()

            # Calculate mean loss on validation set
            valid_loss_mean += valid_loss_total / len(valid_loader)
            valid_curve.append(valid_loss_mean)
            print('Valid:Epoch[{:0>3}/{:0>3}]  mean_loss: {:4f} '.format(
                epoch, MAX_EPOCH, valid_loss_mean
            ))

    # Save checkpoint for resuming training at the end of each epoch
    checkpoint = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_path)
    print("Checkpoint saved for epoch {}".format(epoch))

    # Test model at the end of each epoch
    mean_dc = test_model_and_evaluate('./data/Drishti/od/test/images/',
                                    './data/Drishti/od/test/seg/',
                                    net,
                                    './data/Drishti/od/test/seg-post/',
                                    './data/Drishti/od/test/labels/',
                                    epoch + 1,
                                    'Dri',
                                    'OD')
    # Evaluate whether current model performs better than previous model
    if mean_dc > previous_mean_dc:
        # Update previous model evaluation results
        previous_mean_dc = mean_dc
        # Save current model
        checkpoint = {'model_state_dict': net.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch
                      }
        path_checkpoint = './checkpoints/Rim_to_Dri_od_snp_epoch{}.pkl'.format(epoch + 1)
        torch.save(checkpoint, path_checkpoint)
        print('Checkpoint {} saved!'.format(epoch))
    else:
        print('Model performance did not improve. Checkpoint not saved.')

# Draw loss curve
train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)

valid_x = np.arange(1, len(valid_curve) + 1) * train_iters * val_interval
valid_y = valid_curve

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.title('Drishti-GS loss curve in {} epochs'.format(MAX_EPOCH))
plt.savefig('Dri_loss_curve.png')

# Release GPU cache
torch.cuda.empty_cache()
writer.close()
