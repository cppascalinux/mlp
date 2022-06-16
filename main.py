import torch
from models import ResNet18,ResNet50
# from torchvision.models import resnet18,resnet50
from train import Trainer
import torchvision
import torchvision.transforms as transforms

device='cuda' if torch.cuda.is_available() else 'cpu'
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# for s in trainloader:
# 	print(s[0].shape,s[1].shape)
# 	print(s[0][0])
# 	break
# for s in testloader:
# 	print(s[0].shape,s[1].shape)
# 	break
# print(s[0].shape,s[1].shape)

# print(testset.shape)

model=ResNet18().to(device)
# model=resnet50(pretrained=False).to(device)
trainer=Trainer(model,trainset,testset,device,lr=0.002,batch_size=128,save_per_steps=1000,steps=80000,save_directory='./results18')
trainer.load(27000)
trainer.train(adv=False,eps=0.08)
# train_accuracy,test_accuracy=trainer.evaluate(adv=True,eps=0.08)
# print(train_accuracy,test_accuracy)