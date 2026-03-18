import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from cifar import CIFAR10Dataset
from utils import  progress_bar
from models import get_model
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR

cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    print("==> Preparing data...")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = CIFAR10Dataset(root='./data/cifar-10-batches-py', train=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True,
                             num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    testset = CIFAR10Dataset(root='./data/cifar-10-batches-py', train=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False,
                            num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    print("==> Building model...")
    net = get_model('resnet18').to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    best_acc = 0
    epoch_numer = 50
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    for epoch in range(epoch_numer):
        print(f'\nEpoch {epoch+1}/{epoch_numer}')
        train(epoch, net, trainloader, criterion, optimizer)
        acc = test(epoch, net, testloader, criterion)
        scheduler.step()
        if acc > best_acc:
            print("Saving model...")
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/resnet18_ckpt.pth')
            best_acc = acc
def train(epoch, net, loader, criterion, optimizer):
    net.train()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(loader), f'Train Loss: {loss.item():.4f}')


def test(epoch, net, loader, criterion):
    net.eval()

    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(loader),
                         f'Test Loss: {test_loss / (batch_idx + 1):.4f} | Acc: {100 * correct / total:.2f}%')
    acc = 100 * correct / total
    return acc



if __name__ == '__main__':
    main()


