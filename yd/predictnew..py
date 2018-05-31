def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
	if useTrainCV:
		xgb_param = alg.get_xgb_params()
		xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
		cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
			metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
		alg.set_params(n_estimators=cvresult.shape[0])

	#Fit the algorithm on the data
	alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')

	#Predict training set:
	dtrain_predictions = alg.predict(dtrain[predictors])
	dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

	#Print model report:
	print "\nModel Report"
	print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
	print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)

	feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
	feat_imp.plot(kind='bar', title='Feature Importances')
	plt.ylabel('Feature Importance Score')