# -*- coding:utf-8 -*-
import pandas as pd
from config import *

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

def get_3data(df,time):
	time1=time_diff(time,0)
	time3=time_diff(time,2)
	time5=time_diff(time,4)
	time7=time_diff(time,6)
	time14=time_diff(time,13)
	time21=time_diff(time,20)
	data1=split_by_time(df,time1,time)
	data3=split_by_time(df,time3,time)
	data5=split_by_time(df,time5,time)
	data7=split_by_time(df,time7,time)#最近一周
	data14=split_by_time(df,time14,time)
	data21=split_by_time(df,time21,time)
	return data1,data3,data5,data7,data14,data21

#给行为带上权重
def action_weight(action):
	if action==1:
		return 1.
	elif action==2:
		return 2.
	elif action==3:
		return 4.
	elif action==4:
		return 8.


def min_max_normalize(df, name):
	# 归一化
	max_number = df[name].max()
	min_number = df[name].min()
	# assert max_number != min_number, 'max == min in COLUMN {0}'.format(name)
	df[name] = df[name].map(lambda x: float(x - min_number + 1) / float(max_number - min_number + 1))
	# 做简单的平滑,试试效果如何
	return df

# user features


# 用户和行为统计
def user_action1_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(lambda x: 1. if x==1 else 0.,df[action_label])))
	frame.name=prefix+'user_action1_rate'
	consume_user=df[[user_label]].join(frame)
	grouped=consume_user.groupby(user_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name=prefix+'user_action1_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def user_action2_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(lambda x: 1. if x==2 else 0.,df[action_label])))
	frame.name=prefix+'user_action2_rate'
	consume_user=df[[user_label]].join(frame)
	grouped=consume_user.groupby(user_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name=prefix+'user_action2_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def user_action3_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(lambda x: 1. if x==3 else 0.,df[action_label])))
	frame.name=prefix+'user_action3_rate'
	consume_user=df[[user_label]].join(frame)
	grouped=consume_user.groupby(user_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name=prefix+'user_action3_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def user_action4_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(lambda x: 1. if x==4 else 0.,df[action_label])))
	frame.name=prefix+'user_action4_rate'
	consume_user=df[[user_label]].join(frame)
	grouped=consume_user.groupby(user_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name=prefix+'user_action4_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)

#用户活跃度,4种都有
def user_action_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	rlp = dict(df[user_label].value_counts())
	frame = df[user_label].map(rlp)
	frame.name =prefix+'user_action_count'
	action_user = df[[user_label]].join(frame)
	grouped=action_user.groupby(user_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)#[frame.name]
	#normalized_frame.name=prefix+'user_action_normalized_rate'
	return normalized_frame.rename(columns={frame.name:prefix+'user_action_normalized_rate'})


def user_weight_action_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(action_weight,df[action_label])))
	frame.name=prefix+'user_action_count'
	consume_user=df[[user_label]].join(frame)
	grouped=consume_user.groupby(user_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)#[frame.name]
	#normalized_frame.name=prefix+'user_action_normalized_rate'
	return normalized_frame.rename(columns={frame.name:prefix+'user_weight_action_normalized_rate'})
	
user_action_rates=[user_action1_rate,user_action2_rate,user_action3_rate,user_action4_rate,user_action_rate,user_weight_action_rate]

#用户消费的占各种行为比率
def user_consume_action_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame = pd.Series(list(map(lambda x: 1. if x ==4 else 0., df[action_label])))
	frame.name=prefix+'user_consume_counts'
	user_consume=df[[user_label]].join(frame)
	frame=df[action_label].map(lambda x: 1.)
	frame.name=prefix+'user_action_counts'
	user_consume_want=user_consume.join(frame)
	grouped=user_consume_want.groupby(user_label,as_index=False).sum()
	frame=pd.Series(list(map(lambda x,y: -1.0 if x==0 else float(x)/float(y),grouped[prefix+'user_consume_counts'],grouped[prefix+'user_action_counts'])))
	frame.name=prefix+'user_consume_action_rate'
	return grouped[[user_label]].join(frame)


# def user_consume_action2_rate(df):
	# frame = pd.Series(list(map(lambda x: 1. if x ==4 else 0., df[action_label])))
	# frame.name=prefix+'user_consume_counts'
	# user_consume=df[[user_label]].join(frame)
	# frame=df[action_label].map(lambda x: 1. if x==2 else 0.)
	# frame.name=prefix+'user_action_counts'
	# user_consume_want=user_consume.join(frame)
	# grouped=user_consume_want.groupby(user_label,as_index=False).sum()
	# frame=pd.Series(list(map(lambda x,y: -1.0 if x==0 else float(x)/float(y),grouped[prefix+'user_consume_counts'],grouped[prefix+'user_action_counts'])))
	# frame.name=prefix+'user_consume_action2_rate'
	# return grouped[[user_label]].join(frame)


# def user_consume_action3_rate(df):
	# frame = pd.Series(list(map(lambda x: 1. if x ==4 else 0., df[action_label])))
	# frame.name=prefix+'user_consume_counts'
	# user_consume=df[[user_label]].join(frame)
	# frame=df[action_label].map(lambda x: 1. if x==3 else 0.)
	# frame.name=prefix+'user_action_counts'
	# user_consume_want=user_consume.join(frame)
	# grouped=user_consume_want.groupby(user_label,as_index=False).sum()
	# frame=pd.Series(list(map(lambda x,y: -1.0 if x==0 else float(x)/float(y),grouped[prefix+'user_consume_counts'],grouped[prefix+'user_action_counts'])))
	# frame.name=prefix+'user_consume_action3_rate'
	# return grouped[[user_label]].join(frame)


#用户消费的占想要的（有过行为234）的比率用户核销率
def user_consume_want_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame = pd.Series(list(map(lambda x: 1. if x ==4 else 0., df[action_label])))
	frame.name=prefix+'user_consume_counts'
	user_consume=df[[user_label]].join(frame)
	frame=df[action_label].map(lambda x: 1. if x!=1 else 0.)
	frame.name=prefix+'user_want_counts'
	user_consume_want=user_consume.join(frame)
	grouped=user_consume_want.groupby(user_label,as_index=False).sum()
	frame=pd.Series(list(map(lambda x,y: -1.0 if x==0 else float(x)/(float(y)+1),grouped[prefix+'user_consume_counts'],grouped[prefix+'user_want_counts'])))
	frame.name=prefix+'user_consume_want_rate'
	return grouped[[user_label]].join(frame)
	
user_consume_rates=[user_consume_action_rate,user_consume_want_rate]

#用户和类目种类的统计（只有用户）
def user_action_category_counts(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 用户浏览的不同类目数量归一化
 #	 mask = pd.Series(list(map(lambda x: True if x!=1 else False, df[action_label])))
	#统计不同商家的数目并变成正常索引
	grouped = df[[user_label, category_label]].groupby(user_label)[category_label].nunique().reset_index()
	#print(grouped)
	return min_max_normalize(grouped, category_label).rename(columns={category_label: prefix+'user_action_category_coutnt_normalized'})


def user_want_category_counts(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 用户想要的不同类目数量归一化
	mask = pd.Series(list(map(lambda x: True if x!=1 else False, df[action_label])))
	#统计不同类目的数目并变成正常索引
	grouped = df[mask][[user_label, category_label]].groupby(user_label)[category_label].nunique().reset_index()
	#print(grouped)
	return min_max_normalize(grouped, category_label).rename(columns={category_label: prefix+'user_want_category_count_normalized'})


def user_consume_category_counts(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 用户消费的不同类目数量归一化
	mask = pd.Series(list(map(lambda x: True if x==4 else False, df[action_label])))
	#统计不同类目的数目并变成正常索引
	grouped = df[mask][[user_label, category_label]].groupby(user_label)[category_label].nunique().reset_index()
	#print(grouped)
	return min_max_normalize(grouped, category_label).rename(columns={category_label: prefix+'user_consume_category_count_normalized'})

user_category_counts=[user_action_category_counts,user_want_category_counts,user_consume_category_counts]

#反应用户关注类目重复度，公式：（每个类目重复数方和/总类目方）度量用户是否喜欢只在某些类目逛（用户专一度）
def user_category_chongfudu(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame =pd.Series(list(map(lambda x: 1.,df[action_label])))
	frame.name = prefix+'user_action_category'
	user_want_category = df[[user_label, category_label]].join(frame)
	user_dicts = dict(user_want_category.groupby(user_label)[frame.name].sum())
	user_category_dicts = dict(user_want_category.groupby([user_label, category_label])[frame.name].sum())
	unique_user_category = df[[user_label, category_label]].drop_duplicates([user_label, category_label]).reset_index(drop=True)
	#unique_user=df[[user_label]].drop_duplicates([user_label]).reset_index(drop=True)
	
	frame = pd.Series(list(map(lambda x, y: (user_category_dicts[(x, y)]*user_category_dicts[(x, y)]) / ((user_dicts[x]+1)*(user_dicts[x]+1)), unique_user_category[user_label], unique_user_category[category_label])))
	frame.name = prefix+'user_action_category_rate'
	user_chongfudu=unique_user_category.join(frame)
	user_chongfudu=user_chongfudu.groupby(user_label,as_index=False)[frame.name].sum().rename(columns={frame.name:prefix+'user_category_chongfudu'})
	return user_chongfudu


def user_category_want_chongfudu(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame =pd.Series(list(map(lambda x: 1. if x!=1 else 0. ,df[action_label])))
	frame.name = prefix+'user_action_category'
	user_want_category = df[[user_label, category_label]].join(frame)
	user_dicts = dict(user_want_category.groupby(user_label)[frame.name].sum())
	user_category_dicts = dict(user_want_category.groupby([user_label, category_label])[frame.name].sum())
	unique_user_category = df[[user_label, category_label]].drop_duplicates([user_label, category_label]).reset_index(drop=True)
	#unique_user=df[[user_label]].drop_duplicates([user_label]).reset_index(drop=True)
	
	frame = pd.Series(list(map(lambda x, y: (user_category_dicts[(x, y)]*user_category_dicts[(x, y)]) / ((user_dicts[x]+1)*(user_dicts[x]+1)), unique_user_category[user_label], unique_user_category[category_label])))
	frame.name = prefix+'user_action_category_rate'
	user_chongfudu=unique_user_category.join(frame)
	user_chongfudu=user_chongfudu.groupby(user_label,as_index=False)[frame.name].sum().rename(columns={frame.name:prefix+'user_category_want_chongfudu'})
	return user_chongfudu


def user_category_consume_chongfudu(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame =pd.Series(list(map(lambda x: 1. if x==4 else 0. ,df[action_label])))
	frame.name = prefix+'user_action_category'
	user_want_category = df[[user_label, category_label]].join(frame)
	user_dicts = dict(user_want_category.groupby(user_label)[frame.name].sum())
	user_category_dicts = dict(user_want_category.groupby([user_label, category_label])[frame.name].sum())
	unique_user_category = df[[user_label, category_label]].drop_duplicates([user_label, category_label]).reset_index(drop=True)
	#unique_user=df[[user_label]].drop_duplicates([user_label]).reset_index(drop=True)
	
	frame = pd.Series(list(map(lambda x, y: (user_category_dicts[(x, y)]*user_category_dicts[(x, y)]) / ((user_dicts[x]+1)*(user_dicts[x]+1)), unique_user_category[user_label], unique_user_category[category_label])))
	frame.name = prefix+'user_action_category_rate'
	user_chongfudu=unique_user_category.join(frame)
	user_chongfudu=user_chongfudu.groupby(user_label,as_index=False)[frame.name].sum().rename(columns={frame.name:prefix+'user_category_consume_chongfudu'})
	return user_chongfudu

category_chongfudu=[user_category_chongfudu,user_category_want_chongfudu,user_category_consume_chongfudu]

def user_item_chongfudu(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame =pd.Series(list(map(lambda x: 1.,df[action_label])))
	frame.name = prefix+'user_action_item'
	user_want_item = df[[user_label, item_label]].join(frame)
	user_dicts = dict(user_want_item.groupby(user_label)[frame.name].sum())
	user_item_dicts = dict(user_want_item.groupby([user_label, item_label])[frame.name].sum())
	unique_user_item = df[[user_label, item_label]].drop_duplicates([user_label, item_label]).reset_index(drop=True)
	#unique_user=df[[user_label]].drop_duplicates([user_label]).reset_index(drop=True)
	
	frame = pd.Series(list(map(lambda x, y: (user_item_dicts[(x, y)]*user_item_dicts[(x, y)]) / ((user_dicts[x]+1)*(user_dicts[x]+1)), unique_user_item[user_label], unique_user_item[item_label])))
	frame.name =prefix+ 'user_action_item_rate'
	user_chongfudu=unique_user_item.join(frame)
	user_chongfudu=user_chongfudu.groupby(user_label,as_index=False)[frame.name].sum().rename(columns={frame.name:prefix+'user_item_chongfudu'})
	return user_chongfudu

def user_item_want_chongfudu(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame =pd.Series(list(map(lambda x: 1. if x!=1 else 0.,df[action_label])))
	frame.name = prefix+'user_want_item'
	user_want_item = df[[user_label, item_label]].join(frame)
	user_dicts = dict(user_want_item.groupby(user_label)[frame.name].sum())
	user_item_dicts = dict(user_want_item.groupby([user_label, item_label])[frame.name].sum())
	unique_user_item = df[[user_label, item_label]].drop_duplicates([user_label, item_label]).reset_index(drop=True)
	#unique_user=df[[user_label]].drop_duplicates([user_label]).reset_index(drop=True)
	
	frame = pd.Series(list(map(lambda x, y: (user_item_dicts[(x, y)]*user_item_dicts[(x, y)]) / ((user_dicts[x]+1)*(user_dicts[x]+1)), unique_user_item[user_label], unique_user_item[item_label])))
	frame.name = prefix+'user_want_item_rate'
	user_chongfudu=unique_user_item.join(frame)
	user_chongfudu=user_chongfudu.groupby(user_label,as_index=False)[frame.name].sum().rename(columns={frame.name:prefix+'user__want_item_chongfudu'})
	return user_chongfudu


def user_item_consume_chongfudu(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame =pd.Series(list(map(lambda x: 1. if x!=1 else 0.,df[action_label])))
	frame.name = prefix+'user_consume_item'
	user_want_item = df[[user_label, item_label]].join(frame)
	user_dicts = dict(user_want_item.groupby(user_label)[frame.name].sum())
	user_item_dicts = dict(user_want_item.groupby([user_label, item_label])[frame.name].sum())
	unique_user_item = df[[user_label, item_label]].drop_duplicates([user_label, item_label]).reset_index(drop=True)
	#unique_user=df[[user_label]].drop_duplicates([user_label]).reset_index(drop=True)
	
	frame = pd.Series(list(map(lambda x, y: (user_item_dicts[(x, y)]*user_item_dicts[(x, y)]) / ((user_dicts[x]+1)*(user_dicts[x]+1)), unique_user_item[user_label], unique_user_item[item_label])))
	frame.name = prefix+'user_consume_item_rate'
	user_chongfudu=unique_user_item.join(frame)
	user_chongfudu=user_chongfudu.groupby(user_label,as_index=False)[frame.name].sum().rename(columns={frame.name:prefix+'user__consume_item_chongfudu'})
	return user_chongfudu

item_chongfudu=[user_item_chongfudu,user_item_want_chongfudu,user_item_consume_chongfudu]

def add_user_features(df,prefix):
	user_features = []
	user_features.extend(user_action_rates)
	user_features.extend(user_consume_rates)
	user_features.extend(user_category_counts)
	user_features.extend(category_chongfudu)
	user_features.extend(item_chongfudu)
	user_feature_data = df[[user_label]].drop_duplicates([user_label])

	for f in user_features:
		user_feature_data = user_feature_data.merge(f(df,prefix), on=user_label, how='left')
	user_feature_data.fillna(-1, inplace=True)#默认填充-1，就地修改则为inplace=True
	return user_feature_data


#category_features
#类目和行为
def category_action1_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(lambda x: 1. if x==1 else 0.,df[action_label])))
	frame.name=prefix+'category_action1_rate'
	action_category=df[[category_label]].join(frame)
	grouped=action_category.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name=prefix+'category_action1_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def category_action2_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(lambda x: 1. if x==2 else 0.,df[action_label])))
	frame.name=prefix+'category_action2_rate'
	action_category=df[[category_label]].join(frame)
	grouped=action_category.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name=prefix+'category_action2_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def category_action3_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(lambda x: 1. if x==3 else 0.,df[action_label])))
	frame.name=prefix+'category_action3_rate'
	action_category=df[[category_label]].join(frame)
	grouped=action_category.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name=prefix+'category_action3_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def category_want_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(lambda x: 1. if x!=1 else 0,df[action_label])))#1. if x!=1 else 0
	frame.name=prefix+'category_want'
	category_want=df[[category_label]].join(frame)
	grouped=category_want.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name=prefix+'category_want_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def category_action4_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(lambda x: 1. if x==4 else 0.,df[action_label])))
	frame.name=prefix+'category_action4_rate'
	action_category=df[[category_label]].join(frame)
	grouped=action_category.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name=prefix+'category_action4_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def category_action_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 相当于类目流行度
	rlp = dict(df[category_label].value_counts())
	frame = df[category_label].map(rlp)
	frame.name =prefix+'category_action_count'
	action_category = df[[category_label]].join(frame)
	grouped=action_category.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)#[frame.name]
	#normalized_frame.name=prefix+'user_action_normalized_rate'
	return normalized_frame.rename(columns={frame.name:prefix+'category_action_normalized_rate'})


def category_weight_action_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(action_weight,df[action_label])))
	frame.name=prefix+'category_action_rate'
	action_category=df[[category_label]].join(frame)
	grouped=action_category.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)#[frame.name]
	#normalized_frame.name=prefix+'category_action_normalized_rate'
	return normalized_frame.rename(columns={frame.name:prefix+'category_weight_action_normalized_rate'})


category_action_rates=[category_action1_rate,category_action2_rate,category_action3_rate,category_action4_rate,category_want_rate,category_action_rate,category_weight_action_rate]

#类目和用户
def category_action_user_counts(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 类目用户量
 #	 mask = pd.Series(list(map(lambda x: True if x!=1 else False, df[action_label])))
	#统计不同商家的数目并变成正常索引
	grouped = df[[category_label, user_label]].groupby(category_label)[user_label].nunique().reset_index()
	#print(grouped)
	return min_max_normalize(grouped, user_label).rename(columns={user_label: prefix+'category_action_user_count_normalized'})


def category_want_user_counts(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 用户想要的不同类目数量归一化
	mask = pd.Series(list(map(lambda x: True if x!=1 else False, df[action_label])))
	#统计不同类目的数目并变成正常索引
	grouped = df[mask][[category_label, user_label]].groupby(category_label)[user_label].nunique().reset_index()
	#print(grouped)
	return min_max_normalize(grouped, category_label).rename(columns={user_label: prefix+'category_want_user_count_normalized'})


def category_consume_user_counts(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 用户想要的不同类目数量归一化
	mask = pd.Series(list(map(lambda x: True if x==4 else False, df[action_label])))
	#统计不同类目的数目并变成正常索引
	grouped = df[mask][[category_label, user_label]].groupby(category_label)[user_label].nunique().reset_index()
	#print(grouped)
	return min_max_normalize(grouped, category_label).rename(columns={user_label:prefix+ 'category_consuem_user_count_normalized'})

category_user_counts=[category_action_user_counts,category_want_user_counts,category_consume_user_counts]


def category_consume_action_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame = pd.Series(list(map(lambda x: 1. if x ==4 else 0., df[action_label])))
	frame.name=prefix+'category_consume_counts'
	category_consume=df[[category_label]].join(frame)
	frame=df[action_label].map(lambda x: 1.)
	frame.name=prefix+'category_action_counts'
	category_consume_want=category_consume.join(frame)
	grouped=category_consume_want.groupby(category_label,as_index=False).sum()
	frame=pd.Series(list(map(lambda x,y: -1.0 if x==0 else float(x)/float(y),grouped[prefix+'category_consume_counts'],grouped[prefix+'category_action_counts'])))
	frame.name=prefix+'category_consume_action_rate'
	return grouped[[category_label]].join(frame)


def category_consume_want_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame = pd.Series(list(map(lambda x: 1. if x ==4 else 0., df[action_label])))
	frame.name=prefix+'category_consume_counts'
	category_consume=df[[category_label]].join(frame)
	frame=df[action_label].map(lambda x: 1. if x!=1 else 0.)
	frame.name=prefix+'category_want_counts'
	category_consume_want=category_consume.join(frame)
	grouped=category_consume_want.groupby(category_label,as_index=False).sum()
	frame=pd.Series(list(map(lambda x,y: -1.0 if x==0 else float(x)/(float(y)+1),grouped[prefix+'category_consume_counts'],grouped[prefix+'category_want_counts'])))
	frame.name=prefix+'category_consume_want_rate'
	return grouped[[category_label]].join(frame)

category_consume_rates=[category_consume_action_rate,category_consume_want_rate]

def add_category_features(df,prefix):
	category_features=[]
	category_features.extend(category_action_rates)
	category_features.extend(category_user_counts)
	category_features.extend(category_consume_rates)
	
	category_feature_data=df[[category_label]].drop_duplicates([category_label])

	for f in category_features:
		category_feature_data=category_feature_data.merge(f(df,prefix),on=category_label,how='left')
	category_feature_data.fillna(-1,inplace=True)
	return category_feature_data

#物品
def item_action1_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(lambda x: 1. if x==1 else 0.,df[action_label])))
	frame.name=prefix+'item_action1_rate'
	action_item=df[[item_label]].join(frame)
	grouped=action_item.groupby(item_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name=prefix+'item_action1_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)

def item_want_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(lambda x: 1. if x!=1 else 0,df[action_label])))#1. if x!=1 else 0
	frame.name=prefix+'item_want'
	item_want=df[[item_label]].join(frame)
	grouped=item_want.groupby(item_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name=prefix+'item_want_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)

def item_action4_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(lambda x: 1. if x==4 else 0.,df[action_label])))
	frame.name=prefix+'item_action4_rate'
	action_item=df[[item_label]].join(frame)
	grouped=action_item.groupby(item_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name=prefix+'item_action4_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)

def item_action_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 相当于类目流行度
	rlp = dict(df[item_label].value_counts())
	frame = df[item_label].map(rlp)
	frame.name =prefix+'item_action_count'
	action_item = df[[item_label]].join(frame)
	grouped=action_item.groupby(item_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)#[frame.name]
	#normalized_frame.name=prefix+'user_action_normalized_rate'
	return normalized_frame.rename(columns={frame.name:prefix+'item_action_normalized_rate'})

def item_weight_action_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame=pd.Series(list(map(action_weight,df[action_label])))
	frame.name=prefix+'item_action_rate'
	action_item=df[[item_label]].join(frame)
	grouped=action_item.groupby(item_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)#[frame.name]
	#normalized_frame.name=prefix+'category_action_normalized_rate'
	return normalized_frame.rename(columns={frame.name:prefix+'item_weight_action_normalized_rate'})

item_action_rates=[item_action1_rate,item_want_rate,item_action4_rate,item_action_rate,item_weight_action_rate]

def item_action_user_counts(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 类目用户量
 #	 mask = pd.Series(list(map(lambda x: True if x!=1 else False, df[action_label])))
	#统计不同商家的数目并变成正常索引
	grouped = df[[item_label, user_label]].groupby(item_label)[user_label].nunique().reset_index()
	#print(grouped)
	return min_max_normalize(grouped, user_label).rename(columns={user_label: prefix+'item_action_user_count_normalized'})


def item_want_user_counts(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 用户想要的不同类目数量归一化
	mask = pd.Series(list(map(lambda x: True if x!=1 else False, df[action_label])))
	#统计不同类目的数目并变成正常索引
	grouped = df[mask][[item_label, user_label]].groupby(item_label)[user_label].nunique().reset_index()
	#print(grouped)
	return min_max_normalize(grouped, user_label).rename(columns={user_label: prefix+'item_want_user_count_normalized'})


def item_consume_user_counts(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 用户想要的不同类目数量归一化
	mask = pd.Series(list(map(lambda x: True if x!=1 else False, df[action_label])))
	#统计不同类目的数目并变成正常索引
	grouped = df[mask][[item_label, user_label]].groupby(item_label)[user_label].nunique().reset_index()
	#print(grouped)
	return min_max_normalize(grouped, user_label).rename(columns={user_label: prefix+'item_consume_user_count_normalized'})

item_user_counts=[item_action_user_counts,item_want_user_counts,item_consume_user_counts]


def item_consume_action_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame = pd.Series(list(map(lambda x: 1. if x ==4 else 0., df[action_label])))
	frame.name=prefix+'item_consume_counts'
	item_consume=df[[item_label]].join(frame)
	frame=df[action_label].map(lambda x: 1.)
	frame.name=prefix+'item_action_counts'
	item_consume_want=item_consume.join(frame)
	grouped=item_consume_want.groupby(item_label,as_index=False).sum()
	frame=pd.Series(list(map(lambda x,y: -1.0 if x==0 else float(x)/float(y),grouped[prefix+'item_consume_counts'],grouped[prefix+'item_action_counts'])))
	frame.name=prefix+'item_consume_action_rate'
	return grouped[[item_label]].join(frame)

def item_consume_want_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	frame = pd.Series(list(map(lambda x: 1. if x ==4 else 0., df[action_label])))
	frame.name=prefix+'item_consume_counts'
	item_consume=df[[item_label]].join(frame)
	frame=df[action_label].map(lambda x: 1. if x!=1 else 0.)
	frame.name=prefix+'item_want_counts'
	item_consume_want=item_consume.join(frame)
	grouped=item_consume_want.groupby(item_label,as_index=False).sum()
	frame=pd.Series(list(map(lambda x,y: -1.0 if x==0 else float(x)/(float(y)+1),grouped[prefix+'item_consume_counts'],grouped[prefix+'item_want_counts'])))
	frame.name=prefix+'item_consume_want_rate'
	return grouped[[item_label]].join(frame)

item_consume_rates=[item_consume_action_rate,item_consume_want_rate]

def add_item_features(df,prefix):
	item_features=[]
	item_features.extend(item_action_rates)
	item_features.extend(item_user_counts)
	item_features.extend(item_consume_rates)
	
	item_feature_data=df[[item_label]].drop_duplicates([item_label])

	for f in item_features:
		item_feature_data=item_feature_data.merge(f(df,prefix),on=item_label,how='left')
	item_feature_data.fillna(-1,inplace=True)
	return item_feature_data

# 用户和类目
def user_action_category_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 用户对每个类目的普通行为次数占用户普通行为所有类目的比重# 字典索引方式可能有问题
	frame =pd.Series(list(map(lambda x: 1.,df[action_label])))
	frame.name = prefix+'user_action_category'
	user_want_category = df[[user_label, category_label]].join(frame)
	user_dicts = dict(user_want_category.groupby(user_label)[frame.name].sum())
	user_category_dicts = dict(user_want_category.groupby([user_label, category_label])[frame.name].sum())
	unique_user_category = df[[user_label, category_label]].drop_duplicates([user_label, category_label]).reset_index(drop=True)
	frame = pd.Series(list(map(lambda x, y: user_category_dicts[(x, y)] / (user_dicts[x]+1), unique_user_category[user_label], unique_user_category[category_label])))
	frame.name = prefix+'user_action_category_rate'
	return unique_user_category.join(frame)


def user_want_category_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 用户对每个类目的普通关注次数占用户普通关注所有类目的比重# 字典索引方式可能有问题
	frame = pd.Series(list(map(lambda x: 1. if x != 1 else 0., df[action_label])))
	frame.name = prefix+'user_want_category'
	user_want_category = df[[user_label, category_label]].join(frame)
	user_dicts = dict(user_want_category.groupby(user_label)[frame.name].sum())
	user_category_dicts = dict(user_want_category.groupby([user_label, category_label])[frame.name].sum())
	unique_user_category = df[[user_label, category_label]].drop_duplicates([user_label, category_label]).reset_index(drop=True)
	frame = pd.Series(list(map(lambda x, y: user_category_dicts[(x, y)] / (user_dicts[x]+1), unique_user_category[user_label], unique_user_category[category_label])))
	frame.name = prefix+'user_want_category_rate'
	return unique_user_category.join(frame)


def user_consume_category_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 用户对每个类目的消费次数占用户普通关注所有类目的比重# 字典索引方式可能有问题
	frame = pd.Series(list(map(lambda x: 1. if x == 4 else 0., df[action_label])))
	frame.name = prefix+'user_consume_category'
	user_want_category = df[[user_label, category_label]].join(frame)
	user_dicts = dict(user_want_category.groupby(user_label)[frame.name].sum())
	user_category_dicts = dict(user_want_category.groupby([user_label, category_label])[frame.name].sum())
	unique_user_category = df[[user_label, category_label]].drop_duplicates([user_label, category_label]).reset_index(drop=True)
	frame = pd.Series(list(map(lambda x, y: user_category_dicts[(x, y)] / (user_dicts[x]+1), unique_user_category[user_label], unique_user_category[category_label])))
	frame.name = prefix+'user_consume_category_rate'
	return unique_user_category.join(frame)

user_category_rates=[user_action_category_rate,user_want_category_rate,user_consume_category_rate]

def add_user_category_features(df,prefix):
	user_category_features=[]
	user_category_features.extend(user_category_rates)
	
	user_category_feature_data=df[[user_label,category_label]].drop_duplicates([user_label,category_label])

	for f in user_category_features:
		user_category_feature_data=user_category_feature_data.merge(f(df,prefix),on=[user_label,category_label],how='left')
	user_category_feature_data.fillna(-1,inplace=True)
	return user_category_feature_data


#用户和物品
def user_action_item_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 用户对每个物品的普通行为次数占用户普通行为所有类目的比重# 字典索引方式可能有问题
	frame =pd.Series(list(map(lambda x: 1.,df[action_label])))
	frame.name = prefix+'user_action_item'
	user_want_item = df[[user_label, item_label]].join(frame)
	user_dicts = dict(user_want_item.groupby(user_label)[frame.name].sum())
	user_item_dicts = dict(user_want_item.groupby([user_label, item_label])[frame.name].sum())
	unique_user_item = df[[user_label, item_label]].drop_duplicates([user_label, item_label]).reset_index(drop=True)
	frame = pd.Series(list(map(lambda x, y: user_item_dicts[(x, y)] / (user_dicts[x]+1), unique_user_item[user_label], unique_user_item[item_label])))
	frame.name = prefix+'user_action_item_rate'
	return unique_user_item.join(frame)


def user_want_item_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 用户对每个类目的普通行为次数占用户普通行为所有类目的比重# 字典索引方式可能有问题
	frame =pd.Series(list(map(lambda x: 1. if x!=1 else 0. ,df[action_label])))
	frame.name = prefix+'user_action_item'
	user_want_item = df[[user_label, item_label]].join(frame)
	user_dicts = dict(user_want_item.groupby(user_label)[frame.name].sum())
	user_item_dicts = dict(user_want_item.groupby([user_label, item_label])[frame.name].sum())
	unique_user_item = df[[user_label, item_label]].drop_duplicates([user_label, item_label]).reset_index(drop=True)
	frame = pd.Series(list(map(lambda x, y: user_item_dicts[(x, y)] / (user_dicts[x]+1), unique_user_item[user_label], unique_user_item[item_label])))
	frame.name = prefix+'user_want_item_rate'
	return unique_user_item.join(frame)


def user_consume_item_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 用户对每个类目的普通行为次数占用户普通行为所有类目的比重# 字典索引方式可能有问题
	frame =pd.Series(list(map(lambda x: 1. if x==4 else 0. ,df[action_label])))
	frame.name = prefix+'user_consume_item'
	user_want_item = df[[user_label, item_label]].join(frame)
	user_dicts = dict(user_want_item.groupby(user_label)[frame.name].sum())
	user_item_dicts = dict(user_want_item.groupby([user_label, item_label])[frame.name].sum())
	unique_user_item = df[[user_label, item_label]].drop_duplicates([user_label, item_label]).reset_index(drop=True)
	frame = pd.Series(list(map(lambda x, y: user_item_dicts[(x, y)] / (user_dicts[x]+1), unique_user_item[user_label], unique_user_item[item_label])))
	frame.name = prefix+'user_consume_item_rate'
	return unique_user_item.join(frame)

user_item_rates=[user_action_item_rate,user_want_item_rate,user_consume_item_rate]

def add_user_item_features(df,prefix):
	user_item_features=[]
	user_item_features.extend(user_item_rates)
	
	user_item_feature_data=df[[user_label,item_label]].drop_duplicates([user_label,item_label])

	for f in user_item_features:
		user_item_feature_data=user_item_feature_data.merge(f(df,prefix),on=[user_label,item_label],how='left')
	user_item_feature_data.fillna(-1,inplace=True)
	return user_item_feature_data

#类目和物品
def category_item_action_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 每个物品在类别中受关注度
	frame = pd.Series(list(map(lambda x: 1. , df[action_label])))
	frame.name = prefix+'item_category_action'
	item_category_want = df[[category_label,item_label]].join(frame)
	category_dicts = dict(item_category_want.groupby(category_label)[frame.name].sum())
	category_item_dicts = dict(item_category_want.groupby([category_label, item_label])[frame.name].sum())
	unique_item_category = df[[category_label, item_label]].drop_duplicates([category_label, item_label]).reset_index(drop=True)
	frame = pd.Series(list(map(lambda x, y: category_item_dicts[(x, y)] / (category_dicts[x]+1), unique_item_category[category_label], unique_item_category[item_label])))
	frame.name =prefix+ 'category_item_action_rate'
	return unique_item_category.join(frame)

def category_item_want_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 每个物品在类别中受关注度
	frame = pd.Series(list(map(lambda x: 1. if x != 1 else 0., df[action_label])))
	frame.name = prefix+'item_category_want'
	item_category_want = df[[category_label,item_label]].join(frame)
	category_dicts = dict(item_category_want.groupby(category_label)[frame.name].sum())
	category_item_dicts = dict(item_category_want.groupby([category_label, item_label])[frame.name].sum())
	unique_item_category = df[[category_label, item_label]].drop_duplicates([category_label, item_label]).reset_index(drop=True)
	frame = pd.Series(list(map(lambda x, y: category_item_dicts[(x, y)] / (category_dicts[x]+1), unique_item_category[category_label], unique_item_category[item_label])))
	frame.name = prefix+'category_item_want_rate'
	return unique_item_category.join(frame)

def category_item_consume_rate(df,prefix):
	df.reset_index(drop=True,inplace=True)
	# 每个物品在类别中受关注度
	frame = pd.Series(list(map(lambda x: 1. if x == 4 else 0., df[action_label])))
	frame.name = prefix+'item_category_consume'
	item_category_consume = df[[category_label,item_label]].join(frame)
	category_dicts = dict(item_category_consume.groupby(category_label)[frame.name].sum())
	category_item_dicts = dict(item_category_consume.groupby([category_label, item_label])[frame.name].sum())
	unique_item_category = df[[category_label, item_label]].drop_duplicates([category_label, item_label]).reset_index(drop=True)
	frame = pd.Series(list(map(lambda x, y: category_item_dicts[(x, y)] / (category_dicts[x]+1), unique_item_category[category_label], unique_item_category[item_label])))
	frame.name = prefix+'category_item_consume_rate'
	return unique_item_category.join(frame)

item_category=[category_item_want_rate,category_item_consume_rate,category_item_action_rate]

def add_category_item_features(df,prefix):
	item_category_features=[]
	item_category_features.extend(item_category)
	
	feature_data=df[[category_label,item_label]].drop_duplicates([category_label,item_label])

	for f in item_category_features:
		feature_data=feature_data.merge(f(df,prefix),on=[category_label,item_label],how='left')
	feature_data.fillna(-1,inplace=True)
	return feature_data


#time feature
def time_feature(df,prefix):
	#TODO
	return df

#OTHER category item feature
	



def add_label(f_df,d_df):
	frame=pd.Series(list(map(lambda x:1. if x==4 else 0.,d_df[action_label])))
	frame.name='Label'
	print 'd_df',d_df[:10]
	label=d_df[[user_label,item_label]].join(frame)
	print(label[:10])
	print'label',type(label)
	return f_df.merge(label,on=[user_label,item_label],how='left')#.fillna(0,inplace=True)#drop_duplicates([])


#另一个文件夹的类目占率
def category_rate(df):
	frame=pd.Series(list(map(lambda x: 1.,df[item_label])))
	frame.name='category_count'
	category_rate=df[[category_label]].join(frame)
	grouped=category_rate.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)#[frame.name]
	#normalized_frame.name='category_rate'
	return normalized_frame.rename(columns={frame.name:'category_rate'})


def extract_features(raw_data,category_raw_data,features_path):
#整个时间的特征先只用最近一周的特征
	if features_path==train_feature_data_path:
		day1,day3,day5,day7,day14,day21=get_3data(raw_data,train_feature_end_time)
		#data1.to_csv(train_path+'one_week.csv',index=False)
	elif features_path==validate_feature_data_path:
		day1,day3,day5,day7,day14,day21=get_3data(raw_data,validate_feature_end_time)
		#data1.to_csv(validate_path+'one_week.csv',index=False)
	elif features_path==predict_feature_data_path:
		day1,day3,day5,day7,day14,day21=get_3data(raw_data,predict_feature_end_time)
		#data1.to_csv(predict_path+'one_week.csv',index=False)

	# print 'start extract user features'
	# user_features=add_user_features(raw_data,'all')
	# data1=add_user_features(day1,'day1')
	# data3=add_user_features(day3,'day3')
	# data5=add_user_features(day5,'day5')
	# data7=add_user_features(day7,'day7')
	# user_features=user_features.merge(data1,on=user_label,how='left')
	# user_features=user_features.merge(data3,on=user_label,how='left')
	# user_features=user_features.merge(data5,on=user_label,how='left')
	# user_features=user_features.merge(data7,on=user_label,how='left')
	# user_features.fillna(-1, inplace=True)
	# user_features.to_csv(features_path + 'user_features.csv', index=False)
	
	# print 'start extract category features'
	# category_features=add_category_features(raw_data,'all')
	# data1=add_category_features(day1,'day1')
	# data3=add_category_features(day3,'day3')
	# data5=add_category_features(day5,'day5')
	# data7=add_category_features(day7,'day7')
	# category_features=category_features.merge(data1,on=category_label,how='left')
	# category_features=category_features.merge(data3,on=category_label,how='left')
	# category_features=category_features.merge(data5,on=category_label,how='left')
	# category_features=category_features.merge(data7,on=category_label,how='left')
	
	# category_rate_feature=category_rate(category_raw_data)
# #合并后的特征
	# category_features=category_features.merge(category_rate_feature,on=category_label,how='left')
	# category_features.fillna(-1,inplace=True)
	# category_features.to_csv(features_path + 'category_features.csv', index=False)
	
	# print 'start extract item features'
	# item_features=add_item_features(raw_data,'all')
	# data1=add_item_features(day1,'day1')
	# data3=add_item_features(day3,'day3')
	# data5=add_item_features(day5,'day5')
	# data7=add_item_features(day7,'day7')
	# item_features=item_features.merge(data1,on=item_label,how='left')
	# item_features=item_features.merge(data3,on=item_label,how='left')
	# item_features=item_features.merge(data5,on=item_label,how='left')
	# item_features=item_features.merge(data7,on=item_label,how='left')

	# item_features.fillna(-1,inplace=True)
	# item_features.to_csv(features_path+'item_features.csv',index=False)
	
	# print 'start extract user category features'
	# user_category_features=add_user_category_features(raw_data,'all')
	# data1=add_user_category_features(day1,'day1')
	# data3=add_user_category_features(day3,'day3')
	# data5=add_user_category_features(day5,'day5')
	# data7=add_user_category_features(day7,'day7')
	# user_category_features=user_category_features.merge(data1,on=[user_label,category_label],how='left')
	# user_category_features=user_category_features.merge(data3,on=[user_label,category_label],how='left')
	# user_category_features=user_category_features.merge(data5,on=[user_label,category_label],how='left')
	# user_category_features=user_category_features.merge(data7,on=[user_label,category_label],how='left')

	# user_category_features.fillna(-1, inplace=True)
	# user_category_features.to_csv(features_path+'user_category_features.csv',index=True)

	# print 'start extract user item features'
	# user_item_features=add_user_item_features(raw_data,'all')
	# data1=add_user_item_features(day1,'day1')
	# data3=add_user_item_features(day3,'day3')
	# data5=add_user_item_features(day5,'day5')
	# data7=add_user_item_features(day7,'day7')
	# user_item_features=user_item_features.merge(data1,on=[user_label,item_label],how='left')
	# user_item_features=user_item_features.merge(data3,on=[user_label,item_label],how='left')
	# user_item_features=user_item_features.merge(data5,on=[user_label,item_label],how='left')
	# user_item_features=user_item_features.merge(data7,on=[user_label,item_label],how='left')

	# user_item_features.fillna(-1,inplace=True)
	# user_item_features.to_csv(features_path+'user_item_features.csv',index=True)
	
	print 'start extract category item features'
	category_item_features=add_category_item_features(raw_data,'all')
	data1=add_category_item_features(day1,'day1')
	data3=add_category_item_features(day3,'day3')
	data5=add_category_item_features(day5,'day5')
	data7=add_category_item_features(day7,'day7')
	category_item_features=category_item_features.merge(data1,on=[category_label,item_label],how='left')
	category_item_features=category_item_features.merge(data3,on=[category_label,item_label],how='left')
	category_item_features=category_item_features.merge(data5,on=[category_label,item_label],how='left')
	category_item_features=category_item_features.merge(data7,on=[category_label,item_label],how='left')
	
	category_item_features.fillna(-1, inplace=True)
	category_item_features.to_csv(features_path + 'category_item_features.csv', index=False)


	# print 'fill nan with -1'
	# user_features.fillna(-1, inplace=True)
	# category_features.fillna(-1,inplace=True)
	# item_features.fillna(-1,inplace=True)


	# user_category_features.fillna(-1, inplace=True)
	# user_item_features.fillna(-1,inplace=True)
	# category_item_features.fillna(-1, inplace=True)









def feature_extract(raw_data_path, item_data, features_path):
	raw_data = pd.read_csv(raw_data_path)
#	 category_data = pd.read_csv(category_item_data_path)
	extract_features(raw_data, item_data, features_path)

# def preprocess(raw_path,cleaned_path):
	# df=pd.read_csv(raw_path)
	# print 'original user',len(df)
	# user_buyer=set(df[df[aciton_label]==4][user_label])
	# df=df[df[user_label].isin(user_buyer)]
	# print 'user_buyer',len(df)
   # # 人数统计
	# rlp = dict(df[user_label].value_counts())
	# frame = df[user_label].map(rlp)
	# frame.name ='user_action_count'
	# action_user = df[[user_label]].join(frame)y
	# grouped=action_user.groupby(user_label,as_index=False)
	# d1=grouped[frame.name].sum()
	# normal_user=set(d1[d1[frame.name]<10000][user_label])
	# df=df[df[user_label].isin(normal_user)]
	# print 'normal_user',len(df)

	# df.to_csv(cleaned_path,index=False)



if __name__=='__main__':
	item_data = pd.read_csv(item_file_path)

	#print 'Train features extracting...'
	#feature_extract(train_raw_data_path, item_data, train_feature_data_path)

	print 'Validate features extracting...'
	feature_extract(validate_raw_data_path, item_data, validate_feature_data_path)

	#print 'Predict features extracting...'
	#feature_extract(predict_raw_data_path, item_data, predict_feature_data_path)

	print 'Done'

