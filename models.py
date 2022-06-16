import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
	expansion=1
	def __init__(self,in_channel,out_channel,stride=1):
		super().__init__()
		self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
		self.bn1=nn.BatchNorm2d(out_channel)
		self.conv2=nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1,bias=False)
		self.bn2=nn.BatchNorm2d(out_channel)
		self.shortcut=nn.Sequential()
		if stride!=1 or in_channel !=out_channel*self.expansion:
			self.shortcut=nn.Sequential(
				nn.Conv2d(in_channel,out_channel*self.expansion,kernel_size=1,stride=stride,bias=False),
				nn.BatchNorm2d(out_channel*self.expansion)
			)

	def forward(self,x):
		out=self.conv1(x)
		out=self.bn1(out)
		out=F.relu(out)

		out=self.conv2(out)
		out=self.bn2(out)
		
		out+=self.shortcut(x)
		out=F.relu(out)
		
		return out

class Bottleneck(nn.Module):
	expansion=4
	def __init__(self,in_channel,out_channel,stride=1):
		super().__init__()
		self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False)
		self.bn1=nn.BatchNorm2d(out_channel)
		self.conv2=nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
		self.bn2=nn.BatchNorm2d(out_channel)
		self.conv3=nn.Conv2d(out_channel,out_channel*self.expansion,kernel_size=1,bias=False)
		self.bn3=nn.BatchNorm2d(out_channel*self.expansion)
		self.shortcut=nn.Sequential()
		if stride!=1 or in_channel !=out_channel*self.expansion:
			self.shortcut=nn.Sequential(
				nn.Conv2d(in_channel,out_channel*self.expansion,kernel_size=1,stride=stride,bias=False),
				nn.BatchNorm2d(out_channel*self.expansion)
			)

	def forward(self,x):
		out=self.conv1(x)
		out=self.bn1(out)
		out=F.relu(out)

		out=self.conv2(out)
		out=self.bn2(out)
		out=F.relu(out)

		out=self.conv3(out)
		out=self.bn3(out)
		
		out+=self.shortcut(x)
		out=F.relu(out)
		
		return out

class ResNet(nn.Module):
	def __init__(self,block,num_blocks,num_classes=10):
		super().__init__()
		self.cur_channel=64
		self.conv1=nn.Conv2d(3,self.cur_channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.bn1=nn.BatchNorm2d(self.cur_channel)
		self.layer1=self._make_layer(block,64,1,num_blocks[0])
		self.layer2=self._make_layer(block,128,2,num_blocks[1])
		self.layer3=self._make_layer(block,256,2,num_blocks[2])
		self.layer4=self._make_layer(block,512,2,num_blocks[3])
		self.linear=nn.Linear(self.cur_channel,num_classes)
	
	def _make_layer(self,block,out_channel,stride,num_blocks):
		strides=[stride]+[1]*(num_blocks-1)
		layers=[]
		for s in strides:
			layers.append(block(self.cur_channel,out_channel,stride=s))
			self.cur_channel=out_channel*block.expansion
		return nn.Sequential(*layers)
	
	def forward(self,x):
		out=self.conv1(x)
		out=self.bn1(out)
		out=F.relu(out)

		out=self.layer1(out)
		out=self.layer2(out)
		out=self.layer3(out)
		out=self.layer4(out)

		out=F.avg_pool2d(out,kernel_size=4)
		out=torch.flatten(out,1)
		out=self.linear(out)
		return out

def ResNet18():
	return ResNet(BasicBlock,[2,2,2,2])

def ResNet50():
	return ResNet(Bottleneck,[3,4,6,3])

def test():
	net=ResNet50().cuda()
	x=torch.randn(32,3,32,32).cuda()
	y=net(x)
	print(y.size())


# test()