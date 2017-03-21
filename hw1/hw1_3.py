#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from numpy import *;
import numpy as np;
import math;
import datetime;
import random

from matplotlib import pyplot as plt
from matplotlib import font_manager

f = open('./log1.txt', 'w')


class Perception(object):
	"""docstring for Perception"""
	# 权值矩阵
	#u v
	# 偏置
	#b
	#学习速率 1；0.8；0.5；0.2
	#lerning_rate
	#输入
	#input_
	#净输出
	#pureOutput_
	#输出
	#output_
	#局域梯度
	#local_eta
	#全局梯度
	#ueta veta

	#传输函数
	#func
	#传输函数的导数
	#funcp

	def __init__(self, lerning_rate, input_num, func, funcp):
		super(Perception, self).__init__()
		self.lerning_rate = lerning_rate
		self.func = func
		self.funcp = funcp
		self.u = np.zeros((1,input_num))
		self.u[0][0] = random.random()*2.4 - 1.2
		self.u[0][1] = random.random()*2.4 - 1.2
		self.v = np.zeros((1,input_num))
		self.v[0][0] = random.random()*2.4 - 1.2
		self.v[0][1] = random.random()*2.4 - 1.2
		self.b = random.random()*2.4 - 1.2
	def forCal(self, input_):
		#计算出净输出和输出
		self.input_ = input_
		self.pureOutput_ = self.u.dot(pow(self.input_,2))[0][0] + self.v.dot(self.input_)[0][0] + self.b
		self.output_ = self.func(self.pureOutput_)
	def backCalForOutput(self, error):
		self.local_eta = -error*self.funcp(self.pureOutput_)
		self.ueta = self.local_eta*pow(self.input_.transpose(),2)
		self.veta = self.local_eta*self.input_.transpose()

	def backCalForHide(self, leta_vec, u_vec, v_vec):
		self.local_eta = self.funcp(self.pureOutput_) * leta_vec.dot(2*u_vec*self.output_+v_vec)[0][0] #因为输出只有一个神经元且只有两层。。
		self.ueta = self.local_eta*pow(self.input_.transpose(),2)
		self.veta = self.local_eta*self.input_.transpose()

	def update(self):
		#f.write("更新之前：\n")
		#print(self.u)
		s = ''.join(str(x[0])+',' for x in self.u)
		s2 = ''.join(str(x[0])+',' for x in self.v)
		self.u = self.u - self.lerning_rate * self.ueta
		self.v = self.v - self.lerning_rate * self.veta
		self.b = self.b - self.lerning_rate * self.local_eta
class MultyLayerNeuralNetwork(object):
	"""docstring for NeuralNetwork"""
	#numbers格式:输入维数|第一层传输函数|第一层funcp|第一层神经元个数...依次类推
	def __init__(self, layer, lerning_rate, numbers):
		super(MultyLayerNeuralNetwork, self).__init__()
		self.layer = layer-1
		self.lerning_rate = lerning_rate
		self.iteration = 0
		self.network = []
		self.correct_sum = 0
		i=1
		while i<len(numbers):
			curLayer = []
			j=0
			while j<numbers[i]:
				curLayer.append(Perception(lerning_rate, numbers[i-1],sigmoid,sigmoidp))
				j = j + 1
			self.network.append(curLayer)
			i = i + 1

	

	def trainning(self, data, name=''):
		print("神经元",name,"开始训练")
		#对每一个数据
		self.data = data
		#data = [[0,0,0],[1,1,0],[1,0,1],[0,1,1]] 异或问题训练
		self.end = False
		self.iteration = 0
		self.err = 0
		self.rate = 1.0
		self.correct_sum = 0
		while self.rate > 0.0001 and self.iteration < 300:
			random.shuffle(self.data)
			self.correct_sum = 0
			tmp = 0
			for x in self.data:
				item = np.array([[x[0]],[x[1]]])
				t = hardlim(x[2])
				res = self.forward_pro(item)
				if (res >0.5 and t==1) or (res <0.5 and t==0):
					self.correct_sum = self.correct_sum + 1
				error = t - res
				tmp = tmp + pow(error,2)
				self.back_pro(error)
				self.update()
			self.iteration = self.iteration + 1
			#print(correct_sum)
			#if correct_sum == 4:
				#break
			if self.err == 0:
				self.err = tmp
			else:
				self.rate = abs(tmp-self.err)/self.err
				self.err = tmp
		print("神经元",name,"迭代了",self.iteration,"次","训练数据一共",len(self.data),",正确了", self.correct_sum)

	def forward_pro(self, item):
		#前向网络，计算各个神经元的净输入和输出，返回最终结果
		i = 0
		inputx = item
		while i < self.layer:
			j = 0
			tmp = np.zeros((len(self.network[i]),1))
			while j < len(self.network[i]):
				self.network[i][j].forCal(inputx)
				tmp[j][0] = self.network[i][j].output_
				j = j + 1
			inputx = tmp
			i = i + 1
		return self.network[self.layer-1][0].output_
	
	def back_pro(self, error):
		#反向传播，计算各个神经元的局域梯度和梯度
		i = self.layer-1
		self.network[i][0].backCalForOutput(error)
		i = i-1
		while i>=0:
			j = 0
			leta_vec = np.zeros((1,len(self.network[i+1])))
			u_vec = np.zeros((len(self.network[i+1]),1))
			v_vec = np.zeros((len(self.network[i+1]),1))
			
			while j<len(self.network[i]):
				k = 0
				while k < len(self.network[i+1]):
					leta_vec[0][k] = self.network[i+1][k].local_eta
					u_vec[k][0] = self.network[i+1][k].u[0][j]
					v_vec[k][0] = self.network[i+1][k].v[0][j]
					k = k + 1
				self.network[i][j].backCalForHide(leta_vec, u_vec, v_vec)
				j = j + 1
			i = i - 1

	def update(self):
		#更新权值
		i = 0
		while i<self.layer:
			j = 0
			while j<len(self.network[i]):
				self.network[i][j].update()
				j = j + 1
			i = i + 1
	def test(self, data, name=""):
		print("测试正确个数:",self.correct_sum)
		#测试训练效果
		self.data = data
		#self.data = [[0,0,0],[1,1,0],[1,0,1],[0,1,1]] 异或问题
		correct_sum = 0
		f.write('[')
		for x in self.data:
			item = np.array([[x[0]],[x[1]]])
			t = hardlim(x[2])
			res = self.forward_pro(item)
			f.write('['+str(x[0])+','+str(x[1])+',')
			if res >0.5:
				f.write('1],')
			else:
				f.write('0],')
			if (res >0.5 and t==1) or (res <0.5 and t==0):
				correct_sum = correct_sum + 1
			#print(x,",",res)
		print("数据一共：",len(self.data))
		print("神经元",name,"正确",correct_sum)

	def plot(self, xdata, ydata):
		plt.figure(figsize=(8, 5), dpi=80)
		axes = plt.subplot(111)

		xdata = np.array(xdata)
		ydata = np.array(ydata)
		type1_x = []
		type1_y = []
		type2_x = []
		type2_y = []
		n = 100
		x = np.linspace(-3.5,3.5,n)
		y = np.linspace(-3.5,3.5,n)

		for i in x:
			for j in y:
				item = np.array([[i],[j]])
				res = self.forward_pro(item)
				if res > 0.5:
					type1_x.append(i)
					type1_y.append(j)
				else:
					type2_x.append(i)
					type2_y.append(j)

		type1 = axes.scatter(type1_x, type1_y, s=20, alpha=0.4, c='grey')
		type2 = axes.scatter(type2_x, type2_y, s=40, alpha=0.4, c='white')

		plt.scatter(xdata[:, 0], xdata[:, 1], c='red', cmap=plt.cm.Paired)
		plt.scatter(ydata[:, 0], ydata[:, 1], c='green', cmap=plt.cm.Paired)

		plt.show()

def hardlim(n):
	if n>=0.5:
		return 1
	return 0

def sigmoid(n):
	return 1.0/(1+pow(math.e, -n))

def sigmoidp(n):
	x = pow(math.e, -n)
	return x/pow(1+x, 2)

def readData(filename):
	#读取文件，返回数据集
	data = []
	with open(filename, 'r') as f:
		for line in f.readlines():
			a = line.strip().split('\t')
			data.append([float(x) for x in a])
	#print(data)
	return data

if __name__=='__main__':
	nn = MultyLayerNeuralNetwork(3,0.5,[2,10,1])
	#network.readData('two_spiral_train.txt')
	begin = datetime.datetime.now()

	data = readData('./two_spiral_train.txt')
	nn.trainning(data)

	end = datetime.datetime.now()
	print("训练时间：",end-begin)

	data = readData('./two_spiral_test.txt')
	nn.test(data)
	f.close()
