# -*- coding:utf-8 -*-
import pandas as pd
from config import *
import math
import operator
chong=7
crazy=30
verycrazy=60
n=8#前几个相似的用户
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

def user_action_item_rate(df):

	# 用户对每个物品的普通行为次数占用户普通行为所有类目的比重# 字典索引方式可能有问题
	frame=pd.Series(list(map(action_weight,df[action_label])))
	frame.name = 'user_action_item'
	user_want_item = df[[user_label, item_label]].join(frame)
	user_dicts = dict(user_want_item.groupby(user_label)[frame.name].sum())
	user_item_dicts = dict(user_want_item.groupby([user_label, item_label])[frame.name].sum())
	unique_user_item = df[[user_label, item_label]].drop_duplicates([user_label, item_label]).reset_index(drop=True)
	frame = pd.Series(list(map(lambda x, y: user_item_dicts[(x, y)] / (user_dicts[x]+1), unique_user_item[user_label], unique_user_item[item_label])))
	frame.name = 'user_action_item_rate'
	return unique_user_item.join(frame)

#重复购买推荐
#生成数据
df=pd.read_csv('./data/tianchi_fresh_comp_train_user.csv')
frame =pd.Series(list(map(lambda x: 1. if x==4 else 0. ,df[action_label])))
frame.name = 'user_consume_item'
user_want_item = df[[user_label, item_label]].join(frame)
user_dicts = dict(user_want_item.groupby(user_label)[frame.name].sum())
user_item_dicts = dict(user_want_item.groupby([user_label, item_label])[frame.name].sum())

chongfu_buyer={k:v for k,v in user_item_dicts.items() if v >=chong}
crazy_buyer={k:v for k,v in user_dicts.items() if v >=crazy}
very_crazy_buyer={k:v for k,v in user_dicts.items() if v >=verycrazy}

f = open('../tuijian/chongfu'+str(chong)+'.txt','w')  
f.write(str(chongfu_buyer))  
f.close()

f = open('../tuijian/crazy'+str(crazy)+'.txt','w')  
f.write(str(crazy_buyer))  
f.close()  

f = open('../tuijian/verycrazy'+str(verycrazy)+'.txt','w')  
f.write(str(very_crazy_buyer))  
f.close() 

#读取  
df=pd.read_csv('./data/tianchi_fresh_comp_train_user.csv')
f = open('../tuijian/chongfu'+str(chong)+'.txt','r')
a = f.read()
chongfu_buyer=eval(a)
f.close()
f = open('../tuijian/crazy'+str(crazy)+'.txt','r')
a = f.read()
crazy_buyer = eval(a)
f.close()
f = open('../tuijian/verycrazy'+str(verycrazy)+'.txt','r')
a = f.read()
very_crazy_buyer = eval(a)
f.close()

print 'chongfu_buyer is %d'%len(chongfu_buyer)
print 'crazy_buyeri is %d'%len(crazy_buyer)
print 'verycrazy_buyeri is %d'%len(very_crazy_buyer)
#
user=[k[0] for k in chongfu_buyer.keys()]
item=[k[1] for k in chongfu_buyer.keys()]
user_item1=pd.DataFrame({'user_id':user,'item_id':item})
user_item1.astype(str).to_csv('../tuijian/chongfurecomand'+str(chong)+'.csv',index=False)

crazy_user=crazy_buyer.keys()

crazy_user_frame=df[df[user_label].isin(crazy_user)]
crazy_user_frame.reset_index(drop=True,inplace=True)
crazy_user_item_like=user_action_item_rate(crazy_user_frame)

crazy_user_item_like=crazy_user_item_like.set_index(user_label)

# 
crazy_user_items_like_dict=dict()
count=0
for u,i in crazy_user_item_like.iterrows():
	if count==0:
		count+=1
		continue
	crazy_user_items_like_dict.setdefault(u,{}).setdefault(i[item_label],i['user_action_item_rate'])

#用户相似度字典
def UserSimilarity(train):
	# build inverse table for item_users
	item_users = dict()
	for u, items in train.items():
		for i in items.keys():
			if i not in item_users:
				item_users[i] = set()
			item_users[i].add(u)
	#calculate co-rated items between users
	C = dict()
	N = dict()
	for i, users in item_users.items():
		for u in users:
			C1=C.setdefault(u,{})
			N[u]=N.setdefault(u,0)+1
			for v in users:
				if u == v:
					continue
				C1[v]=C1.setdefault(v,0)+ 1 / math.log(1 + len(users))
	#calculate finial similarity matrix W
	W = dict()
	for u, related_users in C.items():
		W[u]={}
		for v, cuv in related_users.items():
			W[u][v]=cuv / math.sqrt(N[u] * N[v])
			#W.setdefault(u,{}).setdefault(v,cuv / math.sqrt(N[u] * N[v]))
	return W
#用户推荐

def Recommend(user, train, W):
	rank = dict()
	interacted_items = train[user].keys()
	K=len(W[user].items())
	if K>=20:
		K=20
	if K!=0:
		for v, wuv in sorted(W[user].items(), key=operator.itemgetter(1),reverse=True)[0:K]:
			for i, rvi in train[v].items():
				if i in interacted_items:
					#we should filter items user interacted before
					continue
				rank[i]=rank.setdefault(i,0) + wuv * rvi
	return rank

W=UserSimilarity(crazy_user_items_like_dict)

#删除没有和其相似的用户，实际上改数据中确实有一个
for u,vv in W.items():
	if len(vv)==0:
		del W[u]
#开始推荐
recomand_ui=dict()
#对多数用户做相似度，对少数用户做推荐
for u in very_crazy_buyer.keys():#W.keys()
	rank=Recommend(u,crazy_user_items_like_dict,W)
	sorted_rank=sorted(rank.items(),key=operator.itemgetter(1),reverse=True)
	need_len=len(sorted_rank)
	if(need_len>n):
		need_len=n
	for ranki in sorted_rank[:need_len]:
		recomand_ui.setdefault(u,[]).append(ranki[0])

lst=[]
for u in recomand_ui.keys():
	for i in recomand_ui[u]:
		lst.append((u,i))

frame=pd.DataFrame()
recomand2=frame.append(pd.DataFrame(lst))
recomand2.columns=[user_label,item_label]

recomand2.astype(int).astype(str).to_csv('../tuijian/crazyrecomandv2_'+str(verycrazy)+'_'+str(n)+'.csv',index=False)

