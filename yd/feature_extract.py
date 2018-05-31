# -*- coding:utf-8 -*-
import pandas as pd
from config import *

def preprocess(raw_path,cleaned_path):
	df=pd.read_csv(raw_path)
	


def min_max_normalize(df, name):
    # 归一化
    max_number = df[name].max()
    min_number = df[name].min()
    # assert max_number != min_number, 'max == min in COLUMN {0}'.format(name)
    df[name] = df[name].map(lambda x: float(x - min_number + 1) / float(max_number - min_number + 1))
    # 做简单的平滑,试试效果如何
    return df

# user features


# 用户分别4种行为占用户自身和全部的比例
def user_action1_rate(df):
	frame=pd.Series(list(map(lambda x: 1. if x==1 else 0.,df[action_label])))
	frame.name='user_action1_rate'
	consume_user=df[[user_label]].join(frame)
	grouped=consume_user.groupby(user_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name='user_action1_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def user_action2_rate(df):
	frame=pd.Series(list(map(lambda x: 1. if x==2 else 0.,df[action_label])))
	frame.name='user_action2_rate'
	consume_user=df[[user_label]].join(frame)
	grouped=consume_user.groupby(user_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name='user_action2_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def user_action3_rate(df):
	frame=pd.Series(list(map(lambda x: 1. if x==3 else 0.,df[action_label])))
	frame.name='user_action3_rate'
	consume_user=df[[user_label]].join(frame)
	grouped=consume_user.groupby(user_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name='user_action3_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def user_action4_rate(df):
	frame=pd.Series(list(map(lambda x: 1. if x==4 else 0.,df[action_label])))
	frame.name='user_action4_rate'
	consume_user=df[[user_label]].join(frame)
	grouped=consume_user.groupby(user_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name='user_action4_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)

#用户活跃度,4种都有
#def user_action_rate(df):
#	frame=pd.Series(list(map(lambda x: 1. if x!=-1 else 0.,df[action_label])))
#	frame.name='user_action_count'
#	consume_user=df[[user_label]].join(frame)
#	grouped=consume_user.groupby(user_label,as_index=False)
#	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)#[frame.name]
#	normalized_frame.name='user_action_normalized_rate'
#	return normalized_frame.rename(columns={normalized_frame.name:'user_action_normalized_rate'})
def user_action_rate(df):
    # 人数统计
    rlp = dict(df[user_label].value_counts())
    frame = df[user_label].map(rlp)
    frame.name ='user_action_count'
    action_user = df[[user_label]].join(frame)
    grouped=action_user.groupby(user_label,as_index=False)
    normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)#[frame.name]
	#normalized_frame.name='user_action_normalized_rate'
    return normalized_frame.rename(columns={frame.name:'user_action_normalized_rate'})
user_action_rates=[user_action1_rate,user_action2_rate,user_action3_rate,user_action4_rate,user_action_rate]

#用户消费的占浏览的比率用户核销率
def user_consume_action_rate(df):
	frame = pd.Series(list(map(lambda x: 1. if x ==4 else 0., df[action_label])))
	frame.name='user_consume_counts'
	user_consume=df[[user_label]].join(frame)
	frame=df[action_label].map(lambda x: 1.)
	frame.name='user_action_counts'
	user_consume_want=user_consume.join(frame)
	grouped=user_consume_want.groupby(user_label,as_index=False).sum()
	frame=pd.Series(list(map(lambda x,y: -1.0 if x==0 else float(x)/float(y),grouped['user_consume_counts'],grouped['user_action_counts'])))
	frame.name='user_consume_action_rate'
	return grouped[[user_label]].join(frame)


def user_consume_action2_rate(df):
	frame = pd.Series(list(map(lambda x: 1. if x ==4 else 0., df[action_label])))
	frame.name='user_consume_counts'
	user_consume=df[[user_label]].join(frame)
	frame=df[action_label].map(lambda x: 1. if x==2 else 0.,df[action_label])
	frame.name='user_action_counts'
	user_consume_want=user_consume.join(frame)
	grouped=user_consume_want.groupby(user_label,as_index=False).sum()
	frame=pd.Series(list(map(lambda x,y: -1.0 if x==0 else float(x)/float(y),grouped['user_consume_counts'],grouped['user_action_counts'])))
	frame.name='user_consume_action2_rate'
	return grouped[[user_label]].join(frame)


def user_consume_action3_rate(df):
	frame = pd.Series(list(map(lambda x: 1. if x ==4 else 0., df[action_label])))
	frame.name='user_consume_counts'
	user_consume=df[[user_label]].join(frame)
	frame=df[action_label].map(lambda x: 1. if x==3 else 0., df[action_label])
	frame.name='user_action_counts'
	user_consume_want=user_consume.join(frame)
	grouped=user_consume_want.groupby(user_label,as_index=False).sum()
	frame=pd.Series(list(map(lambda x,y: -1.0 if x==0 else float(x)/float(y),grouped['user_consume_counts'],grouped['user_action_counts'])))
	frame.name='user_consume_action3_rate'
	return grouped[[user_label]].join(frame)


#用户消费的占想要的（有过行为234）的比率用户核销率
def user_consume_want_rate(df):
	frame = pd.Series(list(map(lambda x: 1. if x ==4 else 0., df[action_label])))
	frame.name='user_consume_counts'
	user_consume=df[[user_label]].join(frame)
	frame=df[action_label].map(lambda x: 1. if x!=1 else 0.)
	frame.name='user_want_counts'
	user_consume_want=user_consume.join(frame)
	grouped=user_consume_want.groupby(user_label,as_index=False).sum()
	frame=pd.Series(list(map(lambda x,y: -1.0 if x==0 else float(x)/float(y),grouped['user_consume_counts'],grouped['user_want_counts'])))
	frame.name='user_consume_want_rate'
	return grouped[[user_label]].join(frame)
	

user_consume=[user_consume_action_rate,user_consume_action2_rate,user_consume_action3_rate,user_consume_want_rate]
#只有用户
def user_action_category_counts(df):
    # 用户浏览的不同类目数量归一化
 #   mask = pd.Series(list(map(lambda x: True if x!=1 else False, df[action_label])))
	#统计不同商家的数目并变成正常索引
    grouped = df[[user_label, category_label]].groupby(user_label)[category_label].nunique().reset_index()
    #print(grouped)
    return min_max_normalize(grouped, category_label).rename(columns={category_label: 'user_action_category_coutnt_normalized'})



def user_consume_category_counts(df):
    # 用户想要的不同类目数量归一化
    mask = pd.Series(list(map(lambda x: True if x==4 else False, df[action_label])))
	#统计不同类目的数目并变成正常索引
    grouped = df[mask][[user_label, category_label]].groupby(user_label)[category_label].nunique().reset_index()
    #print(grouped)
    return min_max_normalize(grouped, category_label).rename(columns={category_label: 'user_consume_category_count_normalized'})


def user_want_category_counts(df):
    # 用户想要的不同类目数量归一化
    mask = pd.Series(list(map(lambda x: True if x!=1 else False, df[action_label])))
	#统计不同类目的数目并变成正常索引
    grouped = df[mask][[user_label, category_label]].groupby(user_label)[category_label].nunique().reset_index()
    #print(grouped)
    return min_max_normalize(grouped, category_label).rename(columns={category_label: 'user_want_category_count_normalized'})



#只有类目
def category_action1_rate(df):
	frame=pd.Series(list(map(lambda x: 1. if x==1 else 0.,df[action_label])))
	frame.name='category_action1_rate'
	action_category=df[[category_label]].join(frame)
	grouped=action_category.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name='category_action1_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def category_action2_rate(df):
	frame=pd.Series(list(map(lambda x: 1. if x==2 else 0.,df[action_label])))
	frame.name='category_action2_rate'
	action_category=df[[category_label]].join(frame)
	grouped=action_category.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name='category_action2_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def category_action3_rate(df):
	frame=pd.Series(list(map(lambda x: 1. if x==3 else 0.,df[action_label])))
	frame.name='category_action3_rate'
	action_category=df[[category_label]].join(frame)
	grouped=action_category.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name='category_action3_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def category_action4_rate(df):
	frame=pd.Series(list(map(lambda x: 1. if x==4 else 0.,df[action_label])))
	frame.name='category_action4_rate'
	action_category=df[[category_label]].join(frame)
	grouped=action_category.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name='category_action4_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)


def category_action_rate(df):
    # 人数统计
    rlp = dict(df[category_label].value_counts())
    frame = df[category_label].map(rlp)
    frame.name ='category_action_count'
    action_category = df[[category_label]].join(frame)
    grouped=action_category.groupby(category_label,as_index=False)
    normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)#[frame.name]
	#normalized_frame.name='user_action_normalized_rate'
    return normalized_frame.rename(columns={frame.name:'category_action_normalized_rate'})

def category_action_user_counts(df):
    # 类目用户量
 #   mask = pd.Series(list(map(lambda x: True if x!=1 else False, df[action_label])))
	#统计不同商家的数目并变成正常索引
    grouped = df[[category_label, user_label]].groupby(category_label)[user_label].nunique().reset_index()
    #print(grouped)
    return min_max_normalize(grouped, user_label).rename(columns={user_label: 'category_action_user_count_normalized'})


def category_consume_user_counts(df):
    # 用户想要的不同类目数量归一化
    mask = pd.Series(list(map(lambda x: True if x==4 else False, df[action_label])))
	#统计不同类目的数目并变成正常索引
    grouped = df[mask][[category_label, user_label]].groupby(category_label)[user_label].nunique().reset_index()
    #print(grouped)
    return min_max_normalize(grouped, category_label).rename(columns={category_label: 'category_consuem_user_count_normalized'})


def category_want_user_counts(df):
    # 用户想要的不同类目数量归一化
    mask = pd.Series(list(map(lambda x: True if x!=1 else False, df[action_label])))
	#统计不同类目的数目并变成正常索引
    grouped = df[mask][[category_label, user_label]].groupby(category_label)[user_label].nunique().reset_index()
    #print(grouped)
    return min_max_normalize(grouped, category_label).rename(columns={category_label: 'category_want_user_count_normalized'})


#只有用户

def user_action_category_rate(df):
    # 用户对每个类目的普通行为次数占用户普通行为所有类目的比重# 字典索引方式可能有问题
	frame =pd.Series(list(map(lambda x: 1.,df[action_label])))
	frame.name = 'user_action_category'
	user_want_category = df[[user_label, category_label]].join(frame)
	user_dicts = dict(user_want_category.groupby(user_label)[frame.name].sum())
	user_category_dicts = dict(user_want_category.groupby([user_label, category_label])[frame.name].sum())
	unique_user_category = df[[user_label, category_label]].drop_duplicates([user_label, category_label]).reset_index(drop=True)
	frame = pd.Series(list(map(lambda x, y: user_category_dicts[(x, y)] / user_dicts[x], unique_user_category[user_label], unique_user_category[category_label])))
	frame.name = 'user_action_category_rate'
	return unique_user_category.join(frame)	


def user_want_category_rate(df):
    # 用户对每个类目的普通关注次数占用户普通关注所有类目的比重# 字典索引方式可能有问题
    frame = pd.Series(list(map(lambda x: 1. if x != 1 else 0., df[action_label])))
    frame.name = 'user_want_category'
    user_want_category = df[[user_label, category_label]].join(frame)
    user_dicts = dict(user_want_category.groupby(user_label)[frame.name].sum())
    user_category_dicts = dict(user_want_category.groupby([user_label, category_label])[frame.name].sum())
    unique_user_category = df[[user_label, category_label]].drop_duplicates([user_label, category_label]).reset_index(drop=True)
    frame = pd.Series(list(map(lambda x, y: user_category_dicts[(x, y)] / (user_dicts[x]+1), unique_user_category[user_label], unique_user_category[category_label])))
    frame.name = 'user_want_category_rate'
    return unique_user_category.join(frame)
user_category=[user_action_category_rate,user_want_category_rate]	

def add_user_features(df):
    user_features = []
    user_features.extend(user_action_rates)
    user_features.extend(user_consume)

    user_features.extend([user_action_category_counts,user_want_category_counts])

    user_feature_data = df[[user_label]].drop_duplicates([user_label])

    for f in user_features:
        user_feature_data = user_feature_data.merge(f(df), on=user_label, how='left')
    user_feature_data.fillna(-1, inplace=True)#默认填充-1，就地修改则为inplace=True
    return user_feature_data

def add_user_category_features(df):
	user_category_features=[]
	user_category_features.extend(user_category)
	
	user_category_feature_data=df[[user_label,category_label]].drop_duplicates([user_label,category_label])

	for f in user_category_features:
		user_category_feature_data=user_category_feature_data.merge(f(df),on=[user_label,category_label],how='left')
	user_category_feature_data.fillna(-1,inplace=True)
	return user_category_feature_data


#category features
#类目流行度
def category_popularity(df):
	frame=pd.Series(list(map(lambda x: 1.,df[action_label])))
	frame.name='category_popularity'
	category_pop=df[[category_label]].join(frame)
	grouped=category_pop.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)#[frame.name]
#	normalized_frame.name='user_action_normalized_rate'
	return normalized_frame.rename(columns={frame.name:'category_pop_rate'})

#类目关注率
def category_want_rate(df):
	frame=pd.Series(list(map(lambda x: 1.if x!=1 else 0,df[action_label])))
	frame.name='category_want'
	category_want=df[[category_label]].join(frame)
	grouped=category_want.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name='category_want_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)

def category_consume_rate(df):
	frame=pd.Series(list(map(lambda x: 1.if x==4 else 0,df[action_label])))
	frame.name='category_consume'
	category_consume=df[[category_label]].join(frame)
	grouped=category_consume.groupby(category_label,as_index=False)
	normalized_frame=min_max_normalize(grouped[frame.name].sum(),frame.name)[frame.name]
	normalized_frame.name='category_consume_normalized_rate'
	return grouped[frame.name].mean().join(normalized_frame)

category=[category_popularity,category_want_rate,category_consume_rate]

def add_category_features(df):
	category_features=[]
	category_features.extend(category)
	category_features.append(category_action_user_counts)
	
	category_feature_data=df[[category_label]].drop_duplicates([category_label])

	for f in category_features:
		category_feature_data=category_feature_data.merge(f(df),on=category_label,how='left')
	category_feature_data.fillna(-1,inplace=True)
	return category_feature_data
#category _item_feature
def category_item_want_rate(df):
    # 每个物品在类别中受关注度
    frame = pd.Series(list(map(lambda x: 1. if x != 1 else 0., df[action_label])))
    frame.name = 'item_category_want'
    item_category_want = df[[category_label,item_label]].join(frame)
    category_dicts = dict(item_category_want.groupby(category_label)[frame.name].sum())
    category_item_dicts = dict(item_category_want.groupby([category_label, item_label])[frame.name].sum())
    unique_item_category = df[[category_label, item_label]].drop_duplicates([category_label, item_label]).reset_index(drop=True)
    frame = pd.Series(list(map(lambda x, y: category_item_dicts[(x, y)] / (category_dicts[x]+1), unique_item_category[category_label], unique_item_category[item_label])))
    frame.name = 'category_item_want_rate'
    return unique_item_category.join(frame)

def category_item_consume_rate(df):
    # 每个物品在类别中受关注度
    frame = pd.Series(list(map(lambda x: 1. if x == 4 else 0., df[action_label])))
    frame.name = 'item_category_consume'
    item_category_consume = df[[category_label,item_label]].join(frame)
    category_dicts = dict(item_category_consume.groupby(category_label)[frame.name].sum())
    category_item_dicts = dict(item_category_consume.groupby([category_label, item_label])[frame.name].sum())
    unique_item_category = df[[category_label, item_label]].drop_duplicates([category_label, item_label]).reset_index(drop=True)
    frame = pd.Series(list(map(lambda x, y: category_item_dicts[(x, y)] / (category_dicts[x]+1), unique_item_category[category_label], unique_item_category[item_label])))
    frame.name = 'category_item_consume_rate'
    return unique_item_category.join(frame)

item_category=[category_item_want_rate,category_item_consume_rate]

def add_category_item_features(df):
	item_category_features=[]
	item_category_features.extend(item_category)
	
	feature_data=df[[category_label,item_label]].drop_duplicates([category_label,item_label])

	for f in item_category_features:
		feature_data=feature_data.merge(f(df),on=[category_label,item_label],how='left')
	feature_data.fillna(-1,inplace=True)
	return feature_data
#time feature
def time_feature(df):
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
	print 'start extract user features'
	user_features=add_user_features(raw_data)
	print 'start extract user category features'
	user_category_features=add_user_category_features(raw_data)

	print 'start extract category features'
	category_features=add_category_features(raw_data)
	category_rate_feature=category_rate(category_raw_data)
#合并后的特征
	category_features=category_features.merge(category_rate_feature,on=category_label,how='left')
	print 'start extract category item features'
	category_item_features=add_category_item_features(raw_data)

	print 'fill nan with -1'
	user_features.fillna(-1, inplace=True)
#	category_features.fillna(-1, inplace=True)
	user_category_features.fillna(-1, inplace=True)
	category_features.fillna(-1,inplace=True)
	category_item_features.fillna(-1, inplace=True)

	user_features.to_csv(features_path + 'user_features.csv', index=False)
	category_features.to_csv(features_path + 'category_features.csv', index=False)
	user_category_features.to_csv(features_path + 'user_category_features.csv', index=False)
	category_item_features.to_csv(features_path + 'category_item_features.csv', index=False)

def feature_extract(raw_data_path, item_data, features_path):
    raw_data = pd.read_csv(raw_data_path)
#    category_data = pd.read_csv(category_item_data_path)
    extract_features(raw_data, item_data, features_path)

def preprocess(raw_path,cleaned_path):
    df=pd.read_csv(raw_path)
    print 'original user',len(df)
    user_buyer=set(df[df[aciton_label]==4][user_label])
    df=df[df[user_label].isin(user_buyer)]
    print 'user_buyer',len(df)
   # 人数统计
    rlp = dict(df[user_label].value_counts())
    frame = df[user_label].map(rlp)
    frame.name ='user_action_count'
    action_user = df[[user_label]].join(frame)
    grouped=action_user.groupby(user_label,as_index=False)
    d1=grouped[frame.name].sum()
    normal_user=set(d1[d1[frame.name]<10000][user_label])
    df=df[df[user_label].isin(normal_user)]
    print 'normal_user',len(df)

    df.to_csv(cleaned_path,index=False)

    

if __name__=='__main__':
	item_data = pd.read_csv(item_file_path)
	print 'Train features extracting...'
	feature_extract(train_raw_data_path, item_data, train_feature_data_path)

	print 'Validate features extracting...'
	feature_extract(validate_raw_data_path, item_data, validate_feature_data_path)

	print 'Predict features extracting...'
	feature_extract(predict_raw_data_path, item_data, predict_feature_data_path)

	print 'Done'

