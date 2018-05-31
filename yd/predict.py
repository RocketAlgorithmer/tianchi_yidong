# -*- coding: utf-8 -*-
from config import *
from sklearn.metrics import roc_auc_score
from scipy import interp 
from sklearn.metrics import roc_curve, auc ,precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xgboost
import pandas as pd
import operator

exec_time='Y052008PM08'
#thr=0.413672
mean_tpr = 0.0	
mean_fpr = np.linspace(0, 1, 100)  
loc=400

def f1_errors(preds,dtrain):
	label=dtrain.get_label()
#	preds=1.0/(1.0+np.exp(-preds))
	pred=[int(i>=0.5) for i in preds]
	tp= sum([int(i==1 and j==1) for i,j in zip(pred,label)])
	precision=float(tp)/sum(pred)
	recall=float(tp)/sum(label)
	print('f1_errors %d,%d,%d'%(sum(label),sum(pred),tp))
	return 2*(precision*recall)/(precision+recall),tp

def maxRecall(preds,dtrain): #preds是结果（概率值），dtrain是个带label的DMatrix
	labels=dtrain.get_label() #提取label
	preds=1-preds
	precision,recall,threshold=precision_recall_curve(labels,preds,pos_label=0)
	pr=pd.DataFrame({'precision':precision,'recall':recall})
	return pr[pr.precision>=0.80].recall.max()

def gen_two_set(data1,pre,data2,thr):
	labeled=data1.join(pre)
	labeled=labeled[labeled[probability_consumed_label]>=thr][[user_label,item_label]].reset_index(drop=True)
	labeled2=data2[data2[action_label]==4][[user_label,item_label]].reset_index(drop=True)
	return labeled,labeled2


def calc_f1(preset,refset):
#	 preset[probability_consumed_label].name='Label'
	refset=refset.drop_duplicates([user_label,item_label]).reset_index(drop=True)
	preset=preset.drop_duplicates([user_label,item_label]).reset_index(drop=True)
	inner=pd.merge(refset,preset,on=[user_label,item_label],how='inner')
	precision=1.0*len(inner)/len(preset)
	recall=1.0*len(inner)/len(refset)
	if (precision+recall)!=0:
		f1_score=2*precision*recall/(precision+recall)
		#print 'precision is %.2f %%'%(precision*100)
		#print 'recall is %.2f %%'%(recall*100)
		#print 'f1 score is %.2f %%'%(f1_score*100)
		print('calc_f1 is %d,%d,%d'%(len(refset),len(preset),len(inner)))
		return f1_score,len(inner),precision
	else:
		return 0,len(inner),precision


# load model and data in
model = xgboost.Booster(model_file=model_path+'_'+exec_time+model_file)
#前面添加one_week_
#train_features = pd.read_csv(train_path + 'train_features.csv').astype(float)
train_labels = pd.read_csv(train_path + 'one_week_labels.csv').astype(float)
#validate_features = pd.read_csv(validate_path + 'train_features.csv').astype(float)
validate_labels = pd.read_csv(validate_path + 'one_week_labels.csv').astype(float)
#predict_features = pd.read_csv(predict_path + 'train_features.csv').astype(float)

#train_matrix = xgboost.DMatrix(train_features.values, label=train_labels.values, feature_names=train_features.columns)
train_matrix=xgboost.DMatrix(train_path+'train.buffer')
#val_matrix = xgboost.DMatrix(validate_features.values, label=validate_labels.values, feature_names=validate_features.columns)
val_matrix=xgboost.DMatrix(validate_path+'val.buffer')
#predict_matrix = xgboost.DMatrix(predict_features.values, feature_names=predict_features.columns)
predict_matrix=xgboost.DMatrix(predict_path+'pred.buffer')
###############################################
print 'calcuate f1 score...'

train_pred_labels = model.predict(train_matrix)#, ntree_limit=model.best_ntree_limit)
val_pred_labels = model.predict(val_matrix)#, ntree_limit=model.best_ntree_limit)


train_pred_frame = pd.Series(train_pred_labels, index=train_labels.index)
train_pred_frame.name = probability_consumed_label
#print 'train_label',train_pred_frame[0:100]
val_pred_frame = pd.Series(val_pred_labels, index=validate_labels.index)
val_pred_frame.name = probability_consumed_label
#print 'val_label',val_pred_frame[0:100]

train_raw_data=pd.read_csv(train_path+'psd_one_week.csv')#.drop_duplicates([user_label,item_label])#train_raw_data_path
print('original train one week data is %d',len(train_raw_data))
val_raw_data=pd.read_csv(validate_path+'psd_one_week.csv')#.drop_duplicates([user_labe,item_label])

#ROC
fpr, tpr, thresholds = roc_curve(train_labels.values,train_pred_frame.values)
#mean_tpr += interp(mean_fpr, fpr, tpr)			 #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数	
#mean_tpr[0] = 0.0								 #初始处为0	 
roc_auc = auc(fpr, tpr)	 
#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来	
plt.plot(fpr, tpr, lw=1, label='ROC %s (area = %0.2f)' % ('train', roc_auc))
print 'thr len is ',len(thresholds),'tpr len is',len(tpr)
print(thresholds[:10])
thr1=thresholds[loc]
print(thr1)
fpr, tpr, thresholds = roc_curve(validate_labels.values,val_pred_frame.values) 
#mean_tpr += interp(mean_fpr, fpr, tpr)			 #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数	
#mean_tpr[0] = 0.0								 #初始处为0	 
roc_auc = auc(fpr, tpr)	 
#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来	
plt.plot(fpr, tpr, lw=1, label='ROC %s (area = %0.2f)' % ('validate', roc_auc)) 
thr2=thresholds[loc]
thr=0.1

plt.xlim([-0.05, 1.05])	 
plt.ylim([-0.05, 1.05])	 
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Receiver operating characteristic example')	
plt.legend(loc="lower right") 
plt.gcf().savefig(model_path+'_'+exec_time+'/roc.png')

plt.figure()
plt.plot(thresholds,tpr)
plt.xlim([-0.05, 1.05])	 
plt.ylim([-0.05, 1.05])
plt.xlabel('thresholds')  
plt.ylabel('True Positive Rate') 
plt.savefig(model_path+'_'+exec_time+'/tp_thr')  


train_dataset=pd.read_csv(train_dataset_path)
train_dataset=train_dataset.drop_duplicates([user_label,item_label]).reset_index(drop=True)
val_dataset=pd.read_csv(validate_dataset_path)
val_dataset=val_dataset.drop_duplicates([user_label,item_label]).reset_index(drop=True)
#太慢
# dic={}
# tmp=[set(thresholds)][0]
# for thr in set(thresholds):
	# if(thr==tmp):
		# continue
	# tmp=thr
	# train_pre,train_ref=gen_two_set(train_raw_data,train_pred_frame,train_dataset,thr)
	# val_pre,val_ref=gen_two_set(val_raw_data,val_pred_frame,val_dataset,thr)

	# f11,_=calc_f1(train_pre,train_ref)
	# f12,_=calc_f1(val_pre,val_ref)
	# dic[thr]=(f11+f12)/2
# the=sorted(dic.items(),key = operator.itemgetter(1),reverse=True)[0][0]
# print(the)
train_pre,train_ref=gen_two_set(train_raw_data,train_pred_frame,train_dataset,thr)
print('len of train preset refset %d,%d'%(len(train_pre),len(train_ref)))
val_pre,val_ref=gen_two_set(val_raw_data,val_pred_frame,val_dataset,thr)
print('len of validate preset refset %d,%d'%(len(val_pre),len(val_ref)))
f11,l1,precision1=calc_f1(train_pre,train_ref)
print('f11_score=%f %%,len1=%d,precision=%f'%((f11*100),l1,precision1))
f12,l2,precision2=calc_f1(val_pre,val_ref)
print('f12_score is %f %%,len2=%d,precision=%f'%((f12*100),l2,precision2))

####################################################
maxrecall1=maxRecall(train_pred_frame,train_matrix)
maxrecall2=maxRecall(val_pred_frame.values,val_matrix)
print('maxrecall1 and maxrecall2 is %f ,%f '%(maxrecall1,maxrecall2))


labels = model.predict(predict_matrix)#, ntree_limit=model.best_ntree_limit)

frame = pd.Series(labels)#, index=predict_features.index)
frame.name = probability_consumed_label

submission=pd.read_csv(predict_raw_data_path)
submission=submission[[user_label,item_label]].join(frame)
submission=submission[submission[frame.name]>thr].drop(frame.name,axis=1).drop_duplicates([user_label,item_label])
submission.astype(str).to_csv('{0}_{1}{2}'.format(submission_path, exec_time, submission_file), index=False)
