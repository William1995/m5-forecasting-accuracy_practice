import pandas as pd
import numpy as np
from lib import *
import gc
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

def main():
	# Load data
	print("Reading file...")
	calendar = pd.read_csv('Data/calendar.csv')
	sell_prices = pd.read_csv('Data/sell_prices.csv')
	sales_train_validation = pd.read_csv('Data/sales_train_validation.csv')
	submission = pd.read_csv('Data/sample_submission.csv')

	# Reduce memory size
	print("Reducing memory size...")
	calendar = reduce_mem(calendar)
	sell_prices = reduce_mem(sell_prices)
	sales_train_validation = reduce_mem(sales_train_validation)
	submission = reduce_mem(submission)

	# Combine all data into one dataset
	print("Combining data...")
	data = combine_data(calendar, sell_prices, sales_train_validation, submission, nrows = 27500000, merge = True)
	gc.collect()

	# Encoding data
	print("Encoding data...")
	data = data_encoding(data)
	gc.collect()

	# Create new feature
	print("Creating new feature...")
	data = feature_create(data)
	data = reduce_mem(data)
	gc.collect()

	# Train Test split
	x = data[data['date'] <= '2016-04-24']
	y = x.sort_values('date')['demand']
	test = data[(data['date'] > '2016-04-24')]
	x = x.sort_values('date')
	test = test.sort_values('date')
	del data


	# Model parameters setting
	## k-fold using TimeSeriesSplit
	n_fold = 3
	folds = TimeSeriesSplit(n_splits=n_fold)

	## lgb model parameters
	default_params = {"metric": 'rmse',
					  "verbosity": -1,
	}

	params = {'num_leaves': 555,
		  'min_child_weight': 0.034,
		  'feature_fraction': 0.379,
		  'bagging_fraction': 0.418,
		  'min_data_in_leaf': 106,
		  'objective': 'regression', #default
		  'max_depth': -1,
		  'learning_rate': 0.005,
		  "boosting_type": "gbdt", #defaul
		  "bagging_seed": 11,
		  "metric": 'rmse',
		  "verbosity": -1,
		  'reg_alpha': 0.3899,
		  'reg_lambda': 0.648,
		  'random_state': 222,
	}

	# Model training
	columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'year', 'month', 'week', 'day', 'dayofweek', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 
			'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'lag_t28', 'lag_t29', 'lag_t30', 'rolling_mean_t7', 'rolling_std_t7', 'rolling_mean_t30', 'rolling_mean_t90', 
			'rolling_mean_t180', 'rolling_std_t30', 'price_change_t1', 'price_change_t365', 'rolling_price_std_t7', 'rolling_price_std_t30']

	splits = folds.split(x, y)
	y_preds = np.zeros(test.shape[0])
	y_oof = np.zeros(x.shape[0])
	
	feature_importances = pd.DataFrame()
	feature_importances['feature'] = columns
	
	mean_score = []

	print("Start to train...")
	
	for fold_n, (train_index, valid_index) in enumerate(splits):
		print("-" * 20 +"LGB Fold:"+str(fold_n)+ "-" * 20)
		X_train, X_valid = x[columns].iloc[train_index], x[columns].iloc[valid_index]
		y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
		dtrain = lgb.Dataset(X_train, label=y_train)
		dvalid = lgb.Dataset(X_valid, label=y_valid)
		
		clf = lgb.train(params, dtrain, 2500, valid_sets = [dtrain, dvalid], early_stopping_rounds = 50, verbose_eval=100)
		feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
		
		y_pred_valid = clf.predict(X_valid, num_iteration=clf.best_iteration)
		y_oof[valid_index] = y_pred_valid
		
		val_score = np.sqrt(metrics.mean_squared_error(y_pred_valid, y_valid))
		print(f'val rmse score is {val_score}')
		
		mean_score.append(val_score)
		y_preds += clf.predict(test[columns], num_iteration=clf.best_iteration)/n_fold
		del X_train, X_valid, y_train, y_valid
		gc.collect()
	
	print('mean rmse score over folds is', np.mean(mean_score))
	test['demand'] = y_preds

	# Submission format
	subs = submit_format(test, submission)
	subs.to_csv('submission.csv',index = False)

	# Plot feature importance
	feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)
	feature_importances.to_csv('feature_importances.csv')

	plt.figure(figsize=(16, 12))
	sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(20), x='average', y='feature');
	plt.title('20 TOP feature importance over {} folds average'.format(folds.n_splits));

if __name__ == "__main__":
	main()