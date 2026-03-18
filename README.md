# CIFAR PyTorch Reimplementation
This project is a PyTorch implementation of CIFAR image classification, including  some classic models. The code reproduces training and testing pipelines similar to [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).
## Project Structure
```
Reproduce_pytorch_cifar/
├── data/ -> this is data of cifar
├── checkpoint/ -> this is ckpt.pth
├── models/
│   ├── cnn.py
│   └── resnet.py
├── main.py
├── utils.py
└── cifar.py
```
## Installation
1. Clone the repository:
```Bash
git clone https://github.com/Jiucheng-night/Reproduce_pytorch_cifar.git
cd Reproduce_pytorch_cifar
```
2. Create a conda environment (recommended):
```Bash
conda create -n cifar python=3.12
conda activate cifar
```
3. Install dependencies:
```Bash
pip install torch torchvision numpy
```
Note: Adjust CUDA version according to your GPU

## Usage
1. Training
```Bash
python main.py
```
* Training configurations (batch size, epochs, learning rate) can be modified in ```main.py```.

* Model choice is specified in ```models.py``` (e.g., **CNN** or **ResNet**).
2. Testing
The trained model is automatically saved in ```checkpoint/ckpt.pth```.
To evaluate:
```python
# inside main.py or a separate test script
net.load_state_dict(torch.load('./checkpoint/resnet18_ckpt.pth')['net'])
net.eval()
```
## Data
Place the CIFAR-10 dataset (cifar-10-batches-py) inside the data/ folder.
The dataset is loaded using the custom CIFAR10Dataset class in ```cifar.py```
