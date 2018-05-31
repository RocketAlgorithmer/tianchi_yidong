# -*- coding: utf-8 -*-
from data_view import DataView,ItemView
import pandas as pd
from config import *
import os

def parse_time(time):
	return ''.join(time.split('-')).split(' ')

def split_by_time(data, start_time, end_time):#筛选在规定时间内的条目
		return data[data[action_time_label].map(lambda x: True if start_time <= parse_time(x)[0] <= end_time else False)]
		
def time_diff(time,diff):
	if int(diff)<30:
		c=int(str(time)[-2:])-int(diff)
		if c>=10:
			return str(time[:6])+str(c)
		elif c>0:
			return str(time[:6])+'0'+str(c)
		else:
			return str(int(time[:6])-1)+str(30+c)
	else:
		return -1

def get_data(df,time):
	time7=time_diff(time,6)
	data7=split_by_time(df,time7,time)#最近一周
	return data7

def preprocess(raw_path,cleaned_path):
	df=pd.read_csv(raw_path)
	print 'original user',len(df)
	user_buyer=set(df[df[action_label]==4][user_label])
	df=df[df[user_label].isin(user_buyer)].reset_index(drop=True)
	print 'user_buyer',len(df)
   # 人数统计
	# rlp = dict(df[user_label].value_counts())
	# frame = df[user_label].map(rlp)
	# frame.name ='user_action_count'
	# action_user = df[[user_label]].join(frame)
	# grouped=action_user.groupby(user_label,as_index=False)
	# d1=grouped[frame.name].sum()
	# normal_user=set(d1[d1[frame.name]<10000][user_label])
	# df=df[df[user_label].isin(normal_user)].reset_index(drop=True)
	# print 'normal_user',len(df)

	df.to_csv(cleaned_path,index=False)

if __name__=='__main__':
	print('preprocessing datainininggg')
	preprocess(train_raw_path,train_file_path)
	print('loading training data...')
	train_data=DataView(train_file_path)
	item_data=ItemView(item_file_path)


	train_user_list,trian_user_set=train_data.user_list(),train_data.user_set()
	train_item_list,train_item_set=train_data.item_list(),train_data.item_set()
#	train_category_list,train_category_set=train_data.category_list,train_data.category_set
	all_item_list,all_item_set=item_data.item_list(),item_data.item_set()
#	all_item_category_list,all_item_category_set=item_data.category_list,item_data.category_set
	
	print("spliting data...")
	#if(not os.path.exists(train_raw_data_path)):
	train_raw_data = train_data.filter_by_time(train_feature_start_time, train_feature_end_time)
	one_week=get_data(train_raw_data,train_feature_end_time)
	one_week.reset_index(drop=True).to_csv(train_path+'one_week.csv',index=False)
	print 'train_raw_data',train_data.data.shape, train_raw_data.shape
	train_raw_data.reset_index(drop=True).to_csv(train_raw_data_path, index=False)
	
	#if(not os.path.exists(validate_raw_data_path)):
	validate_raw_data = train_data.filter_by_time(validate_feature_start_time, validate_feature_end_time)
	one_week=get_data(validate_raw_data,validate_feature_end_time)
	one_week.reset_index(drop=True).to_csv(validate_path+'one_week.csv',index=False)
	print 'validate_raw_data',train_data.data.shape, validate_raw_data.shape
	validate_raw_data.reset_index(drop=True).to_csv(validate_raw_data_path, index=False)
	
	#if(not os.path.exists(predict_raw_data_path)):
	predict_raw_data = train_data.filter_by_time(predict_feature_start_time, predict_feature_end_time)
	one_week=get_data(predict_raw_data,predict_feature_end_time)
	one_week.reset_index(drop=True).to_csv(predict_path+'one_week.csv',index=False)
	print 'predict_raw_data',train_data.data.shape, predict_raw_data.shape
	predict_raw_data.reset_index(drop=True).to_csv(predict_raw_data_path, index=False)
	
	#if(not os.path.exists(train_dataset_path)):
	train_dataset = train_data.filter_by_time(train_dataset_time,train_dataset_time)
	print 'train_dataset',train_data.data.shape, train_dataset.shape
	train_dataset.reset_index(drop=True).to_csv(train_dataset_path, index=False)

	#if(not os.path.exists(validate_dataset_path)):
	validate_dataset = train_data.filter_by_time(validate_dataset_time,validate_dataset_time)
	print 'validate_dataset',train_data.data.shape, validate_dataset.shape
	validate_dataset.reset_index(drop=True).to_csv(validate_dataset_path, index=False)

	#if(not os.path.exists(predict_dataset_path)):
	predict_dataset = train_data.filter_by_time(predict_dataset_time,predict_dataset_time)
	print 'predict_dataset',train_data.data.shape, predict_dataset.shape
	predict_dataset.reset_index(drop=True).to_csv(predict_dataset_path, index=False)
	
	print('split data finished')
