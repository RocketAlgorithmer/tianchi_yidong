# -*- coding: utf-8 -*-
from sklearn.metrics import confusion_matrix, roc_auc_score,roc_curve, auc ,precision_recall_curve
from scipy import interp 
from config import *
import numpy as np
import xgboost
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import sys
import json
import operator
#数据换了得改前面和中间两个部分
flag=1	  #是否是第一次生成数据

#exec_time='Y051111AM00'
#thr=0.413672
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100) 
save_stdout = sys.stdout
def maxRecall(preds,dtrain): #preds是结果（概率值），dtrain是个带label的DMatrix
	labels=dtrain.get_label() #提取label
	preds=1-preds
	precision,recall,threshold=precision_recall_curve(labels,preds,pos_label=0)
	pr=pd.DataFrame({'precision':precision,'recall':recall})
	return 'Max Recall:',pr[pr.precision>=0.90].recall.max()

def f1_errors(preds,dtrain):
	label=dtrain.get_label()
	preds=1.0/(1.0+np.exp(-preds))
	pred=[int(i>=0.5) for i in preds]
	tp= sum([int(i==1 and j==1) for i,j in zip(pred,label)])
	precision=float(tp)/sum(pred)
	recall=float(tp)/sum(label)

	return 'f1-score',2*(precision*recall)/(precision+recall)


def gen_two_set(data1,pre,data2,thr):
	labeled=data1.join(pre)
	labeled=labeled[labeled[probability_consumed_label]>thr][[user_label,item_label]]
	labeled2=data2[data2[action_label]==4][[user_label,item_label]]
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
		print 'precision is %.2f %%'%(precision*100)
		print 'recall is %.2f %%'%(recall*100)
		print 'f1 score is %.2f %%'%(f1_score*100)

def calc_auc(df):
	coupon = df[coupon_label].iloc[0]
	y_true = df['Label'].values
	if len(np.unique(y_true)) != 2:
		auc = np.nan
	else:
		y_pred = df[probability_consumed_label].values
		auc = roc_auc_score(np.array(y_true), np.array(y_pred))
	return pd.DataFrame({coupon_label: [coupon], 'auc': [auc]})


def check_average_auc(df):
	grouped = df.groupby(coupon_label, as_index=False).apply(lambda x: calc_auc(x))
	return grouped['auc'].mean(skipna=True)


def create_feature_map(features, fmap):
	outfile = open(fmap, 'w')
	for i, feat in enumerate(features):
		outfile.write('{0}\t{1}\tq\n'.format(i, feat))
	outfile.close()
	
def train(param,num_round=1000,early_stopping_rounds=20):
		exec_time=time.strftime("Y%m%d%I%p%M",time.localtime())
		os.mkdir('{0}_{1}'.format(model_path,exec_time))
		os.mkdir('{0}_{1}'.format(submission_path,exec_time))
		
		train_params=param.copy()
		train_params['num_boost_round']=num_round
		train_params['early_stopping_rounds']=early_stopping_rounds
		json.dump(train_params,open('{0}_{1}{2}'.format(model_path,exec_time,model_params),'wb+'))
		
		print 'get training data'
#已经是去重复的		
		if(flag):
			train_features = pd.read_csv(train_path + 'one_week_train_features.csv').astype(float)
			train_labels = pd.read_csv(train_path + 'one_week_labels.csv').astype(float)
			validate_features = pd.read_csv(validate_path + 'one_week_train_features.csv').astype(float)
			validate_labels = pd.read_csv(validate_path + 'one_week_labels.csv').astype(float)
			predict_features = pd.read_csv(predict_path + 'one_week_train_features.csv').astype(float)

			create_feature_map(train_features.columns.tolist(), '{0}_{1}{2}'.format(model_path, exec_time, model_fmap_file))

			train_matrix = xgboost.DMatrix(train_features.values, label=train_labels.values, feature_names=train_features.columns)
			train_matrix.save_binary(train_path+'train.buffer')
			val_matrix = xgboost.DMatrix(validate_features.values, label=validate_labels.values, feature_names=validate_features.columns)
			val_matrix.save_binary(validate_path+'val.buffer')
			predict_matrix = xgboost.DMatrix(predict_features.values, feature_names=predict_features.columns)
			predict_matrix.save_binary(predict_path+'pred.buffer')
		else:
			train_matrix=xgboost.DMatrix(train_path+'train.buffer')
			val_matrixxgboost.DMatrix(validate_path+'val.buffer')
			predict_matrix=xgboost.DMatrix(predict_path+'pred.buffer')
		watchlist=[(train_matrix,'train'),(val_matrix,'eval')]
		
		print 'Start model trainging...'
		with open('{0}_{1}{2}'.format(model_path, exec_time, model_train_log), 'wb+') as outf:
			sys.stdout = outf
			model = xgboost.train(param, train_matrix, num_boost_round=num_round, evals=watchlist,early_stopping_rounds=early_stopping_rounds)
			
		sys.stdout = save_stdout
		print 'model.best_score: {0}, model.best_iteration: {1}, model.best_ntree_limit: {2}'.format(model.best_score, model.best_iteration, model.best_ntree_limit)
		
		print 'output model data'
		model.save_model('{0}_{1}{2}'.format(model_path, exec_time, model_file))
		model.dump_model('{0}_{1}{2}'.format(model_path, exec_time, model_dump_file))
		
		# importance = model.get_fscore(fmap='{0}_{1}{2}'.format(model_path, exec_time, model_fmap_file))
		# importance = sorted(importance.items(), key=operator.itemgetter(1))
		# df = pd.DataFrame(importance, columns=['feature', 'fscore'])
		# df['fscore'] = df['fscore'] / df['fscore'].sum()
		# df.to_csv('{0}_{1}{2}'.format(model_path, exec_time, model_feature_importance_csv), index=False)
#模型重要度
		print 'plot feature importance'		   
		xgboost.plot_importance(model)
		plt.gcf().set_size_inches(20, 16)
		plt.gcf().set_tight_layout(True)
		plt.gcf().savefig('{0}_{1}{2}'.format(model_path, exec_time, model_feature_importance_file))
		plt.close()
#对数据计算f1得分
		train_pred_labels = model.predict(train_matrix, ntree_limit=model.best_ntree_limit)
		val_pred_labels = model.predict(val_matrix, ntree_limit=model.best_ntree_limit)
		
		train_pred_frame = pd.Series(train_pred_labels, index=train_features.index)
		train_pred_frame.name = probability_consumed_label
		print 'train_label',train_pred_frame[0:100]
		val_pred_frame = pd.Series(val_pred_labels, index=validate_features.index)
		val_pred_frame.name = probability_consumed_label
		print 'val_label',val_pred_frame[0:100]
		
#		 train_true_frame = pd.read_csv(train_path + 'labels.csv')['Label']
#		 val_true_frame = pd.read_csv(validate_path + 'labels.csv')['Label']

		print 'calcuate f1 score...'
		train_raw_data=pd.read_csv(train_path+'psd_one_week.csv')#.drop_duplicates([user_label,item_label])#train_raw_data_path
		val_raw_data=pd.read_csv(validate_path+'psd_one_week.csv')#.drop_duplicates([user_labe,item_label])#validate_raw_data_path
		
		#ROC
		fpr, tpr, thresholds = roc_curve(train_labels.values,train_pred_frame.values)
		#mean_tpr += interp(mean_fpr, fpr, tpr)			 #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数	
		#mean_tpr[0] = 0.0								 #初始处为0	 
		roc_auc = auc(fpr, tpr)	 
		#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来	
		plt.plot(fpr, tpr, lw=1, label='ROC %s (area = %0.2f)' % ('train', roc_auc))
		thr1=thresholds[100]
		fpr, tpr, thresholds = roc_curve(validate_labels.values,val_pred_frame.values) 
		#mean_tpr += interp(mean_fpr, fpr, tpr)			 #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数	
		#mean_tpr[0] = 0.0								 #初始处为0	 
		roc_auc = auc(fpr, tpr)	 
		#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来	
		plt.plot(fpr, tpr, lw=1, label='ROC %s (area = %0.2f)' % ('validate', roc_auc)) 
		thr2=thresholds[100]
		thr=(thr1+thr2)*1./2
		plt.gcf().savefig(model_path+'_'+exec_time+'/roc.png')

		train_dataset=pd.read_csv(train_dataset_path)
		train_dataset=train_dataset.drop_duplicates([user_label,item_label]).reset_index(drop=True)
		val_dataset=pd.read_csv(validate_dataset_path)
		val_dataset=val_dataset.drop_duplicates([user_label,item_label]).reset_index(drop=True)
		train_pre,train_ref=gen_two_set(train_raw_data,train_pred_frame,train_dataset,thr)
		val_pre,val_ref=gen_two_set(val_raw_data,val_pred_frame,val_dataset,thr)
		
		calc_f1(train_pre,train_ref)
		calc_f1(val_pre,val_ref)
		labels = model.predict(predict_matrix, ntree_limit=model.best_ntree_limit)
		print 'pre_label',labels[0:100]
		frame = pd.Series(labels, index=predict_features.index)
		frame.name = probability_consumed_label
		
		plt.figure()
		frame.hist(figsize=(10, 8))
		plt.title('results histogram')
		plt.xlabel('predict probability')
		plt.gcf().savefig('{0}_{1}{2}'.format(submission_path, exec_time, submission_hist_file))
		plt.close()
		
		submission=pd.read_csv(predict_raw_data_path)
		submission=submission[[user_label,item_label]].join(frame)
		submission=submission[submission[frame.name]>thr].drop(frame.name,axis=1).drop_duplicates([user_label,item_label])
		submission.astype(str).to_csv('{0}_{1}{2}'.format(submission_path, exec_time, submission_file), index=False)
		print 'Done'
		
if __name__ == '__main__':
	init_param = {
		'max_depth': 8,
		'eta': 0.01,
		'silent': 1,
		'seed': 13,
		'objective': 'binary:logistic',#'multi:softmax'
		#'num_class':10,
		'eval_metric': 'auc',
		'scale_pos_weight': 1514,#n/p#1514#5
		'subsample': 0.8,
		'colsample_bytree': 0.7,
		'min_child_weight': 10,#100
		'max_delta_step': 20,
		'nthread':50
		}
	train(init_param, num_round=1000, early_stopping_rounds=50)
