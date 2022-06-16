from pickletools import optimize
import torch
from torch.optim import SGD
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import os

from models import test

def cycle(dataset):
	while True:
		for data in dataset:
			yield data

class Trainer:
	def __init__(
			self,
			model,
			train_data,
			test_data,
			device,
			steps=10000,
			save_per_steps=1000,
			lr=0.01,
			batch_size=128,
			momentum=0.9,
			weight_decay=5e-4,
			save_directory='./results',
			log_file='log.txt'
		):
		self.device=device
		self.model=model.to(device)
		self.num_steps=steps
		self.save_per_steps=save_per_steps
		self.train_dataloader=DataLoader(train_data,batch_size,shuffle=True,num_workers=2)
		self.test_dataloader=DataLoader(test_data,batch_size,shuffle=False,num_workers=2)
		self.criterion=nn.CrossEntropyLoss()
		self.optimizer=SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
		self.save_directory=save_directory
		self.log_file=log_file
		if not os.path.exists(save_directory):
			os.mkdir(save_directory)
		self.step=0
	def save(self,steps):
		data={
			'step':self.step,
			'model':self.model.state_dict(),
			'optimizer':self.optimizer.state_dict()
		}
		torch.save(data,self.save_directory+f'/model_{steps}.pt')
		
	def load(self,steps):
		data=torch.load(self.save_directory+f'/model_{steps}.pt')
		self.step=data['step']
		self.model.load_state_dict(data['model'])
		self.optimizer.load_state_dict(data['optimizer'])

	def pgd(self,x,y,eps,step_size=None,steps=7,random_start=True,inc=True):
		if step_size is None:
			step_size=2.5*eps/steps
		lower,upper=x-eps,x+eps
		x_adv=x.clone()

		if random_start:
			x_adv=torch.rand_like(x)*(2*eps)+lower

		for i in range(steps):
			self.model.zero_grad()
			x_test=x_adv.clone().requires_grad_()
			outputs=self.model(x_test)
			loss=self.criterion(outputs,y)
			loss.backward()

			if inc:
				x_adv+=x_test.grad.sign()*step_size
			else:
				x_adv-=x_test.grad.sign()*step_size

			x_test=x_test.detach()
			x_adv=torch.max(x_adv,lower)
			x_adv=torch.min(x_adv,upper)

		self.model.zero_grad()
		return x_adv

	def evaluate(self,adv=False,eps=0.08):
		self.model.eval()
		correct=0
		total=0
		s=' adversarial' if adv else ''
		with tqdm(self.test_dataloader) as pbar:
			pbar.set_description(f'Test set{s} evaluation')
			for data in pbar:
				inputs,labels=data
				inputs,labels=inputs.to(self.device),labels.to(self.device)
				if adv:
					inputs=self.pgd(inputs,labels,eps)
				with torch.no_grad():
					outputs=self.model(inputs)
					predict=outputs.argmax(dim=1)
					correct+=(predict==labels).sum()
					total+=inputs.shape[0]
				pbar.set_description(f'Test set{s} evaluation {correct}/{total}')
		test_accuracy=correct/total

		correct=0
		total=0
		with tqdm(self.train_dataloader) as pbar:
			pbar.set_description(f'Train set{s} evaluation')
			for data in pbar:
				inputs,labels=data
				inputs,labels=inputs.to(self.device),labels.to(self.device)
				if adv:
					inputs=self.pgd(inputs,labels,eps)
				with torch.no_grad():
					outputs=self.model(inputs)
					predict=outputs.argmax(dim=1)
					correct+=(predict==labels).sum()
					total+=inputs.shape[0]
				pbar.set_description(f'Train set{s} evaluation {correct}/{total}')
		train_accuracy=correct/total

		self.model.train()
		return train_accuracy,test_accuracy

	def train(self,adv=False,eps=0.08):
		cycle_data=cycle(self.train_dataloader)
		sum_loss=0
		with tqdm(initial=self.step,total=self.num_steps) as pbar:
			while self.step<self.num_steps:
				inputs,labels=next(cycle_data)
				inputs,labels=inputs.to(self.device),labels.to(self.device)
				if adv:
					self.model.eval()
					inputs=self.pgd(inputs,labels,eps)
					self.model.train()
				self.optimizer.zero_grad()
				
				outputs=self.model(inputs)
				loss=self.criterion(outputs,labels)
				pbar.set_description(f'Training loss: {loss.item():.4f}')
				sum_loss+=loss.item()
				loss.backward()
				self.optimizer.step()

				self.step+=1
				pbar.update(1)

				if self.step%self.save_per_steps==0 or self.step==self.num_steps:
					self.save(self.step)
					train_accuracy,test_accuracy=self.evaluate()
					print(f'\nStep: {self.step}, Train set accuracy:{train_accuracy:.4f}, Test set accuracy:{test_accuracy:.4f}, Average train loss:{sum_loss/self.save_per_steps:.4f}')
					if self.log_file is not None:
						with open(self.log_file,'a') as fp:
							print(f'Step: {self.step}, Train set accuracy:{train_accuracy:.4f}, Test set accuracy:{test_accuracy:.4f}, Average train loss:{sum_loss/self.save_per_steps:.4f}',file=fp)
					sum_loss=0
		print('Training complete')