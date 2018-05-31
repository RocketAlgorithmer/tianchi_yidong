#! /usr/local/bin/python2.7
# -*- coding: utf-8 -*-
from config import *
from feature_extract import *
import feature_extract
import pandas as pd
import numpy as np

#负样本采样比例
percent=0.1

def mysample(df,percent=0.10):
	df1=df[df['Label']==1]
	df0=df[df['Label']==0].sample(frac=percent,random_state=1,axis=0)
	m=pd.concat([df1,df0],axis=0)
	return m.sample(frac=1,random_state=1,axis=0).reset_index(drop=True) 

def add_label(f_df,d_df):
	frame=pd.Series(list(map(lambda x:1. if x==4 else 0.,d_df[action_label])))
	frame.name='Label'
	label=d_df[[user_label,item_label]].join(frame)
	return f_df.merge(label,on=[user_label,item_label],how='left').fillna({frame.name:0.})#.drop_duplicates([user_label,item_label])#不一定加，训练时可重复，预测集合直接删

def gen_feature_data(features_dir, features_data,dataset,label=True,output1=None):#,output2=None):
	user_features = pd.read_csv(features_dir + 'user_features.csv')
	category_features = pd.read_csv(features_dir + 'category_features.csv')
	item_features=pd.read_csv(features_dir+'item_features.csv')
	user_category_features = pd.read_csv(features_dir + 'user_category_features.csv')
	user_item_features=pd.read_csv(features_dir+'user_item_features.csv')
	category_item_features = pd.read_csv(features_dir + 'category_item_features.csv')

	columns = features_data.columns.tolist()
	features_data
	df= features_data.merge(user_features, on=user_label, how='left')
	df= df.merge(category_features, on=category_label, how='left')
	df= df.merge(item_features,on=item_label,how='left')
	df= df.merge(user_category_features, on=[user_label, category_label], how='left')
	df= df.merge(user_item_features,on=[user_label,item_label],how='left')
	df= df.merge(category_item_features, on=[category_label, item_label], how='left')
	#df.drop(columns, axis=1, inplace=True) 
	#if label:
		#df=add_label(df,dataset)
#负样本欠采样
		#df=mysample(df,percent)
		#df[['Label']].astype(float).to_csv(output2,index=False)
#	 df = add_dataset_features(df)
#		 frame=pd.Series(list(map(lambda x:1. if x==4 else 0.,dataset[action_label])))
#		 frame.name='Label'
#		 label=dataset[[user_label,item_label]].join(frame)
#		 df= df.merge(label,on=[user_label,item_label],how='left')
	#print(type(df))

	df.drop(columns, axis=1, inplace=True)#删除列
	df.fillna(-1., inplace=True)
	print 'feature shape is',df.shape
	print 'start dump feature data'
	#if label:
	#	df.drop('Label',axis=1).astype(float).to_csv(output1, index=False)
	#else:
	df.astype(float).to_csv(output1, index=False)
#	 if label:
#		 df[['Label']].astype(float).to_csv(output2,index=False)
#特征不能加index
#def gen_label_data(dataset, output):
#	 df = add_label(feature,dataset)
#	 print 'start dump label data'
#	 df[['Label']].astype(float).to_csv(output, index=False)


def gen_data(path, label=True):
	features_dir = path + 'features/'
	features_file=path + 'raw_data.csv'
	features_file1=path + 'one_week.csv'#用一周前的行为做训练集
	dataset_file = path + 'dataset.csv'

	#features_data=pd.read_csv(features_file)
	features_data1=pd.read_csv(features_file1)
	dataset = pd.read_csv(dataset_file)
	dataset=dataset.drop_duplicates([user_label,item_label]).reset_index(drop=True)
	
	features_data1=features_data1[features_data1[action_label]!=4].reset_index(drop=True)#除去已经买过的干扰
	#features_data=features_data.drop_duplicates([user_label,item_label]).reset_index(drop=True)
	features_data1=features_data1.drop_duplicates([user_label,item_label]).reset_index(drop=True)
	#features_data1.astype(float).to_csv(path + 'psd_one_week.csv',index=False)
	if label:
		features_data1=add_label(features_data1,dataset)
		features_data1=mysample(features_data1,percent)
		features_data1[['Label']].astype(float).to_csv(path+'one_week_labels.csv',index=False)
		features_data1=features_data1.drop('Label',axis=1)
		features_data1.to_csv(path + 'psd_one_week.csv',index=False)
	else:
		features_data1.to_csv(path + 'psd_one_week.csv',index=False)
	#gen_feature_data(features_dir, features_data,dataset,label, path + 'train_features.csv',path+'labels.csv')
	gen_feature_data(features_dir, features_data1,dataset,label, path + 'one_week_train_features.csv')#,path+'one_week_labels.csv')


if __name__ == '__main__':
	print 'generate train data...'
	gen_data(train_path)
	print 'generate validate data...'
	gen_data(validate_path)
	print 'generate predict features...'
	gen_data(predict_path, label=False)

	print'Done'


