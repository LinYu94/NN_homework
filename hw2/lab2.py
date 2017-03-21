# !usr/bin/env  python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../hw1/')

import hw1_3
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count

from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
import os, time, random

class MinMaxNNTwoClass(object):
	#根据xNum确定有多少组, 对每一组执行Min操作, 再对所有组进行Max操作

	"""docstring for MaxMin"""
	def __init__(self, xNum, yNum, xdata, ydata):
		super(MinMaxNNTwoClass, self).__init__()
		self.xNum = xNum
		self.yNum = yNum
		self.xdata = xdata
		self.ydata = ydata
		self.MinGroup = []
		i = 0
		while i < self.xNum:
			self.MinGroup.append([])
			j = 0
			while j < self.yNum:
				self.MinGroup[i].append(hw1_3.MultyLayerNeuralNetwork(3, 0.5, [2,10,1]))
				j = j + 1
			i = i + 1

	def trainning(self):
		print("目前只能顺序执行")
		i = 0
		start = time.time()
		x = hw1_3.readData('../hw1/two_spiral_test.txt')
		txdata, tydata = RandomDecompostionData(2, 2, x)

		while i < self.xNum:
			j = 0
			while j < self.yNum:
				print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
				dataij = self.xdata[i]+self.ydata[j]
				#random.shuffle(dataij)
				print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
				self.MinGroup[i][j].trainning(dataij,str(i)+","+str(j))
				
				self.MinGroup[i][j].test(x)
				self.MinGroup[i][j].plot(txdata[0]+txdata[1],tydata[0]+tydata[1])

				end = time.time()
				print("耗时",end-start,"s")
				start = end
				#p.apply_async(self.task, args=(i,j, self.xdata[i] + self.ydata[j],str(i)+","+str(j),))
				#p.apply_async(self.MinGroup[i][j].trainning, args =(self.xdata[i] + self.ydata[j],str(i)+","+str(j),)) #简直坑爹啊
				j = j + 1
			i = i + 1
		end = time.time()
		print("所有模块训练结束")

	def MinMax(self, item):
		self.Min = []
		self.Res = []
		i = 0
		while i < self.xNum:
			j = 0
			self.Min.append(5)
			while j < self.yNum:
				self.Min[i] = min(self.Min[i], self.MinGroup[i][j].forward_pro(item)) 
				j = j+1
			i = i + 1
		#self.Res = [a,b,c,d]
		i = 0
		res = -5
		while i < len(self.Min):
			res = max(self.Min[i],res)
			i = i + 1
		return res

	def test(self, data):
		print('开始测试')
		plt.figure(figsize=(8, 5), dpi=80)
		axes = plt.subplot(111)
		type1_x = []
		type1_y = []
		type2_x = []
		type2_y = []


		correct_sum = 0
		for x in data:
			item = np.array([[x[0]],[x[1]]])
			t = hw1_3.hardlim(x[2])
			res = self.MinMax(item)
			if (res >0.5 and t==1) or (res <0.5 and t==0):
				correct_sum = correct_sum + 1
			else:
				print(x,",",res)

			if res > 0.5:
				type1_x.append(x[0])
				type1_y.append(x[1])
			else:
				type2_x.append(x[0])
				type2_y.append(x[1])
		type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
		type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
		print("一共：",len(data))
		print("正确",correct_sum)
		plt.show()

		

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
				res = self.MinMax(item)
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


	def xx(self, data):
		X = []
		for x in data:
			item = np.array([[x[0]],[x[1]]])
			res = hw1_3.hardlim(self.MinMax(item))
			X.append(res)
		return np.array([X]).transpose()


def RandomDecompostionData(xNum, yNum, data):
	random.shuffle(data)
	xlist = [[] for x in range(xNum)]
	ylist = [[] for y in range(yNum)]

	for x in data:
		if x[2] > 0.5:
			i = random.randint(0, xNum-1)
			xlist[i].append(x)
		else:
			i = random.randint(0, yNum-1)
			ylist[i].append(x)

	return xlist,ylist
def YDecompostion(xNum, yNum, data):
	x00 = []
	x01 = []
	x10 = []
	x11 = []
	for x in data:
		if x[0] < 0:
			if x[2] > 0.5:
				x01.append(x)
			else:
				x00.append(x)
		else:
			if x[2] > 0.5:
				x11.append(x)
			else:
				x10.append(x)
	return [x11,x01],[x00,x10]

if __name__ == '__main__':
	data = hw1_3.readData('../hw1/two_spiral_train.txt')
	xNum = 2
	yNum = 2
	print("cpu核心数:",cpu_count())
	xdata, ydata = RandomDecompostionData(xNum, yNum, data)
	#xdata, ydata = YDecompostion(xNum, yNum, data)
	print(len(xdata))
	print(len(ydata))

	MinMaxNetwork = MinMaxNNTwoClass(xNum, yNum, xdata, ydata)
	MinMaxNetwork.trainning()
	data = hw1_3.readData('../hw1/two_spiral_test.txt')
	MinMaxNetwork.test(data)

	txdata, tydata = RandomDecompostionData(xNum, yNum, data)
	MinMaxNetwork.plot(txdata[0]+txdata[1],tydata[0]+tydata[1])
			



