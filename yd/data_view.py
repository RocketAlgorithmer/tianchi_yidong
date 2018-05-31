# -*- coding: utf-8 -*-
import pandas as pd
from collections import Counter
from config import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from time import time
import datetime

#print(action_time_label)
def parse_time(time):
	return ''.join(time.split('-')).split(' ')

class DataView:
	def __init__(self, file_path=' '):
		self.data= pd.read_csv(file_path)
#		 self.fields = self.data.columns.tolist()
#		print(1)
#	def __init__(self,data):
#		self.data=data
#		print(2)
	#@property
	def user_list(self):
		return self.data[user_label].tolist()

	#@property
	def user_set(self):
		return set(self.data[user_label].tolist())

	#@property
	def item_list(self):
		return self.data[item_label].tolist()

	#@property
	def item_set(self):
		return set(self.data[item_label].tolist())

	#@property
	def action_list(self):
		return self.data[action_label].tolist()
	#@property
#	 def action_set(self):
#		 return self.data[action_label].tolist()
 
	#@property
	def category_list(self):
		return self.data[category_label].tolist()

	#@property
	def category_set(self):
		return set(self.data[category_label].tolist())
	
	#@property
	def time_list(self):
		return self.data[action_time_label].tolist()

#	#@property
#	def category_set(self):
#		return set(self.data[category_label].tolist())
	
#	#@property
#	def category_list(self):
#		return self.data[category_label].tolist()
		
#	#@property
#	def time_list(self):
#		return self.data[action_time_label].tolist()

#	 #@property
#	 def coupon_consumption_list(self):
#		 fullcut_list = self.data[discount_label][self.data[discount_label].str.contains(':')].tolist()#加个astype（str）
#		 return [x.split(':')[0] for x in fullcut_list]#提取慢减值的第一个数

#	 #@property
#	 def received_data_distribution(self):#优惠券每天发多少张
#		 return dict(self.data.groupby(date_received_label)[date_received_label].count())

	def filter_by_time(self, start_time, end_time):#筛选在规定时间内的条目
		return self.data[self.data[action_time_label].map(lambda x: True if start_time <= parse_time(x)[0] <= end_time else False)]

#def filter_by_time(self,data, start_time, end_time):
#	return data[data[action_time_label].map(lambda x: True if start_time <= parse_time(x)[0] <= end_time else False)]

class ItemView:
	def __init__(self,file_path=''):
		self.data=pd.read_csv(file_path)

	#@property
	def item_list(self):
		return self.data[item_label].tolist()

	#@property
	def item_set(self):
		return set(self.data[item_label].tolist())
	
	#@property
	def category_list(self):
		return self.data[category_label].tolist()
	
	#@property
	def category_set(self):
		return set(self.data[category_label].tolist())

def get_time_diff(start_time, end_time):
	# 计算时间差
	start_time=''.join(start_time.split('-')).split(' ')
	end_time=''.join(end_time.split('-')).split(' ')
	month_diff = int(end_time[0][-4:-2]) - int(start_time[0][-4:-2])
	if month_diff == 0:
		return int(end_time[0][-2:]) - int(start_time[0][-2:])
	else:
		return int(end_time[0][-2:]) - int(start_time[0][-2:]) + month_diff * 30

#df=DataView(train_file_path)
#print((df.time_list())[0:10])
#data= ItemView(item_file_path)
#print(data.item_list)

