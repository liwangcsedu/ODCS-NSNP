# Paper Repository.

## ODCS-NSNP: Optic disc and cup segmentation using deep networks enhanced by nonlinear spiking neural P systems

## Project Overview

ODCS-NSNP (Optic disc and cup segmentation using deep networks enhanced by nonlinear spiking neural P systems) is a deep learning-based medical image segmentation model that focuses on automatic segmentation of optical disc (OD) and optical cup (OC) in fundus images. The model employs advanced neural network structures and non-local perception techniques to accurately identify and segment key structures in fundus images, which is of significant importance for clinical diagnosis of eye diseases such as glaucoma.

## Model Structure

This project implements the ODCS-NSNP model with the following innovations:

1. **SDN Units (Structural Deep Network Units)**: Using separable convolutions and dense connection structures to enhance feature extraction capabilities
2. **SRS Operator (Scale Reconfiguration Structure Operator)**: Replacing traditional pooling and upsampling operations to better preserve spatial information
3. **Multi-scale Feature Fusion**: Achieving precise segmentation through multi-level feature extraction and fusion

The model adopts an encoder-decoder structure and optimizes the training process through a combination of multiple loss functions (Cross Entropy Loss, Focal Loss, Dice Loss).

## Project Structure

```
ODCS-NSNP/
├── train.py              # Main training program
├── test.py               # Main testing program
├── run.sh                # Training launch script
├── models/               # Model definitions
│   ├── ODCS_NSNP.py      # Main network model
│   └── SRS_Operator.py   # Scale Reconfiguration Structure Operator
├── tools/                # Tool library
│   ├── datasets.py       # Dataset loading
│   ├── preprocessing.py  # Data preprocessing
│   ├── evaluation_metrics_for_segmentation.py  # Evaluation metrics
│   ├── evaluate.py       # Model evaluation functions
│   ├── post_processing.py # Post-processing
│   └── ...               # Other utility functions
├── data/                 # Data directory
│   ├── Drishti-files/    # Drishti fundus image dataset
│   ├── Rim-one-r3/       # RIM-ONE fundus image dataset
│   └── Refuge/           # REFUGE fundus image dataset
└── checkpoints/          # Model checkpoints
```

## Datasets

This project supports the following fundus image datasets:

1. **Drishti-GS**: A high-resolution fundus image dataset collected by Aravind Eye Hospital in India, including 50 training images and 51 test images, each with detailed optical disc and cup segmentation annotations.

2. **RIM-ONE**: A retinal image dataset for glaucoma diagnosis provided by Spanish research teams, containing different levels of glaucoma lesions.

3. **REFUGE**: A fundus image dataset released by the MICCAI 2018 challenge, focusing on glaucoma assessment.

The datasets used in this project have been preprocessed to adapt to model training requirements. Data is typically organized as follows:

```
data/
├── Drishti-files/
│   ├── Training/
│   │   └── labels/
│   └── Test/
├── Rim-one-r3/
└── Refuge/
```

You may need to use preprocessing tools in the tools directory to prepare your own data before training.

## Requirements

- Python 3.6+
- PyTorch 1.7+
- CUDA 10.2+ (for GPU training)
- Other dependencies:
  - numpy
  - matplotlib
  - scikit-image
  - tqdm
  - torchvision
  - PIL
  - torchstat
  - opencv-python (cv2)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/liwangcdedu/ODCS-NSNP.git
cd ODCS-NSNP
```

2. Create a virtual environment and install dependencies:
```bash
conda create -n odcs-nsnp python=3.8
conda activate odcs-nsnp
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-image tqdm pillow torchstat opencv-python
```

## Usage

### Data Preprocessing

Before starting training, you may need to preprocess the data. Use the preprocessing tools in the tools folder:

```bash
python tools/preprocessing.py
```

You can also use other preprocessing tools, such as:
```bash
python tools/crop.py   # Crop images
python tools/resize.py # Resize images
```

### Training the Model

You can start training with the following command:

```bash
python train.py
```

Or use the provided script to run directly in the background:

```bash
bash run.sh
```

Training parameters can be adjusted in the train.py file, including:
- Batch size
- Learning rate
- Training epochs
- Regularization parameters
- Dataset paths

### Model Evaluation

The model will be automatically evaluated at the end of each epoch during training. You can also use the evaluation functions in tools/evaluate.py to evaluate the model separately:

```bash
# Evaluate performance on a specific dataset
python -c "from tools.evaluate import evaluate_segmentation_results; evaluate_segmentation_results('model_name', 'prediction_path', 'ground_truth_path')"
```

Evaluation metrics include:
- Dice coefficient (DC)
- Jaccard coefficient (JAC)
- Accuracy (ACC)
- Sensitivity (SEN)
- Specificity (SPC)
- Cup-to-Disc Ratio (CDR)

## Result Visualization

TensorBoard logs are generated during the training process. You can view the training curves with the following command:

```bash
tensorboard --logdir=runs
```

## Model Checkpoints

Model checkpoints during training are saved in the checkpoints directory. You can use these checkpoints to continue training or for inference.

## Citation

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