# data file path
train_raw_path='./data/tianchi_fresh_comp_train_user.csv'
train_file_path = './data/preprocessed_train_user.csv'
item_file_path='./data/tianchi_fresh_comp_train_item.csv'
#offline_train_file_path = './data/ccf_data_revised/ccf_offline_stage1_train.csv'
#offline_test_file_path = './data/ccf_data_revised/ccf_offline_stage1_test_revised.csv'

# split data path
#active_user_offline_data_path = './data/data_split/active_user_offline_record.csv'
#active_user_online_data_path = './data/data_split/active_user_online_record.csv'
#offline_user_data_path = './data/data_split/offline_user_record.csv'
#online_user_data_path = './data/data_split/online_user_record.csv'

train_path = './data/data_split/train_data/'
train_feature_data_path = train_path + 'features/'
train_raw_data_path = train_path + 'raw_data.csv'
#train_cleanedraw_data_path=train_path+'cleanedraw_data.csv'
train_subraw_data_path=train_path+'subraw_data.csv'
train_dataset_path = train_path + 'dataset.csv'
train_subdataset_path=train_path+'subdataset.csv'
train_raw_online_data_path = train_path + 'raw_online_data.csv'

validate_path = './data/data_split/validate_data/'
validate_feature_data_path = validate_path + 'features/'
validate_raw_data_path = validate_path + 'raw_data.csv'
#validate_cleaneraw_data_path=validate_path+'cleanedraw_data.csv'
validate_dataset_path = validate_path + 'dataset.csv'
validate_raw_online_data_path = validate_path + 'raw_online_data.csv'

predict_path = './data/data_split/predict_data/'
predict_feature_data_path = predict_path + 'features/'
predict_raw_data_path = predict_path + 'raw_data.csv'
predict_dataset_path = predict_path + 'dataset.csv'
predict_raw_online_data_path = predict_path + 'raw_online_data.csv'

# model path
model_path = './data/model/model'
model_file = '/model'
model_dump_file = '/model_dump.txt'
model_fmap_file = '/model.fmap'
model_feature_importance_file = '/feature_importance.png'
model_feature_importance_csv = '/feature_importance.csv'
model_train_log = '/train.log'
model_params = '/param.json'

val_diff_file = '/val_diff.csv'

# submission path
submission_path = './data/submission/submission'
submission_hist_file = '/hist.png'
submission_file = '/tianchi_mobile_recommendation_predict.csv'

# raw field name
user_label = 'user_id'
item_label = 'item_id'
action_label = 'behavior_type'
user_geohash_label='user_geohash'
category_label='item_category'
action_time_label='time'
probability_consumed_label = 'Probability'

# global values
consume_time_limit = 15

train_feature_start_time = '20141119'
train_feature_end_time = '20141217'
train_dataset_time = '20141218'
#train_dataset_end_time = '20141218'

validate_feature_start_time = '20141118'
validate_feature_end_time = '20141216'
validate_dataset_time = '20141217'
#validate_dataset_end_time = '20160514'

predict_feature_start_time = '20141120'
predict_feature_end_time = '20141218'
predict_dataset_time = '20141219'
#predict_dataset_end_time = '20160731'


