#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import *;
import numpy as np;

# 训练数据
training_set = [[-1,2,2],[3,1,0],[-2,1,2],[2,-1,1],[0,2,0],[1,1,0],[1,-2,1],[-1,1,2],[2,0,1]]



class Perceptron(object):
	# 权值矩阵
	W = np.array([[0,0],[0,0]])
	# 偏置向量
	b = np.array([[0],[0]])
	#学习速率 1；0.8；0.5；0.2
	delta = 1.0
	#迭代次数
	iteration = 0
	#分类是否完成
	end = False

	"""docstring for perce"""
	def __init__(self):
		super(Perceptron, self).__init__()
	
	def initArg(self, delta):
		self.W = np.array([[0,0],[0,0]])
		self.b = np.array([[0],[0]])
		self.delta = delta
		self.iteration = 0
		self.end = False

	def updateWb(self, item, error):
		self.W = self.W + self.delta * error.dot(item)
		self.b = self.b + self.delta * error

	#采用硬极限传输函数
	def hardlim(self, x):
		return 1 if x>0 else 0

	def trans(self, target):
		if target == 0:
			return np.array([[0],[0]])
		elif target == 1:
			return np.array([[0],[1]])
		else :
			return np.array([[1],[0]])


	def cal(self, item):
		res = self.W.dot(item.transpose()) + self.b
		#print("res",res)
		i = len(res) - 1
		kind = 0
		exp = 0
		while i >= 0:
			res[i][0] = self.hardlim(res[i][0])
			kind += res[i][0]*pow(2,exp)
			exp = exp + 1
			i = i-1
		return (res,kind)




	def train(self, trainingset):
		while self.end == False and self.iteration < 50:
			self.end = True
			for t in training_set:
				#构造输入行向量
				item = np.array([[t[0],t[1]]])
				target = t[2]
				#获得类别列向量表示
				t = self.trans(target)
				result = self.cal(item)
				if result[1] != target and (target==2 and result[1] == 3) == False:
					#print("t=",t)
					#print("result[0]=",result[0])
					error = t - result[0]
					self.updateWb(item, error)
					self.end = False
			self.iteration = self.iteration + 1
		print("迭代次数：",self.iteration)
		self.end = False
		self.iteration = 0

	def test(self, data):
		for t in data:
			item = np.array([[t[0],t[1]]])
			result = self.cal(item)
			print(t[0]," ",t[1]," ",result[1])

perceptron = Perceptron()
print("学习速率=1")
perceptron.train(training_set)
print(perceptron.W)
print(perceptron.b)
perceptron.test(training_set)
#测试不同的学习速率
print("学习速率=0.8")
perceptron.initArg(0.8)
perceptron.train(training_set)
print(perceptron.W)
print(perceptron.b)
perceptron.test(training_set)
print("学习速率=0.5")
perceptron.initArg(0.5)
perceptron.train(training_set)
print(perceptron.W)
print(perceptron.b)
perceptron.test(training_set)
print("学习速率=0.2")
perceptron.initArg(0.2)
perceptron.train(training_set)
print(perceptron.W)
print(perceptron.b)
perceptron.test(training_set)