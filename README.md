# Enhanced Camouflaged Object Detection via Boundary-Guided Iterative Feature Interaction and Attention Mechanism

## Usage

> The training and testing experiments are set in Implementation Details of the *BAINet*.

### 1. Prerequisites

> Note that FSPNet is only tested on Windows with the following environments.

- Creating a virtual environment in terminal: `conda create -n BAINet python=3.8`.
- Installing necessary packages: `pip install -r requirements.txt`

### 2. Downloading Training and Testing Datasets

- Download the [training set](https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view?usp=sharing) (COD10K-train) used for training 
- Download the [testing sets](https://drive.google.com/file/d/1SLRB5Wg1Hdy7CQ74s3mTQ3ChhjFRSFdZ/view?usp=sharing) (COD10K-test + CAMO-test + CHAMELEON + NC4K ) used for testing

### 3. Training Configuration

- `python train.py`

### 4. Testing Configuration

- `python test.py`

### 5. Results download

The prediction results and pretrained weights of our BAINet are stored on [Baidu Drive](https://pan.baidu.com/s/1yncX2Ct7oh3dWXPCwKupJQ?pwd=ryzg) (ryzg) please check.

## Citation
