import pandas as pd
import numpy as np
import gc
from sklearn import preprocessing

def reduce_mem(df):
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	start_mem = df.memory_usage().sum() / 1024**2 
	for col in df.columns:
		col_type = df[col].dtypes
		
		if col_type in numerics:
			c_min = df[col].min()
			c_max = df[col].max()
			
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)  
			
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)	

	end_mem = df.memory_usage().sum() / 1024**2
	#print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
	return df

def combine_data(calendar, sell_prices, sales_train_validation, submission, nrows = 55000000, merge = False):
	# melt sales data to assign format
	sales_train_validation = pd.melt(sales_train_validation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
	sales_train_validation = reduce_mem(sales_train_validation)
	
	# seperate test dataframes
	test1_rows = [row for row in submission['id'] if 'validation' in row]
	test2_rows = [row for row in submission['id'] if 'evaluation' in row]
	test1 = submission[submission['id'].isin(test1_rows)]
	test2 = submission[submission['id'].isin(test2_rows)]
	
	# change column names
	test1.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931', 
					 'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']
	test2.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959', 
					 'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']
	
	# get product info table
	product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
	
	# merge test data with product info table
	test1 = test1.merge(product, how = 'left', on = 'id')
	test2 = test2.merge(product, how = 'left', on = 'id')
	
	# melt test data to assign format
	test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
	test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
	
	# add new label
	sales_train_validation['part'] = 'train'
	test1['part'] = 'test1'
	test2['part'] = 'test2'
	
	# combine all the data
	data = pd.concat([sales_train_validation, test1, test2], axis = 0)
	del sales_train_validation, test1, test2
	
	# sample part of dataset for training
	data = data.loc[nrows:]
	
	# drop some features from calendar
	calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)
	
	# delete test2 for now
	data = data[data['part'] != 'test2']
	
	# whether merge data from other source table
	if merge:
		# merge from calendar
		data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
		data.drop(['d', 'day'], inplace = True, axis = 1)
		
		# merge from the price
		data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')

	else: 
		pass
	
	gc.collect()
	
	return data

def data_encoding(data):
	nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
	for feature in nan_features:
		data[feature].fillna('unknown', inplace = True)
	
	encoder = preprocessing.LabelEncoder()
	data['id_encode'] = encoder.fit_transform(data['id'])
	
	cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
	for feature in cat:
		encoder = preprocessing.LabelEncoder()
		data[feature] = encoder.fit_transform(data[feature])
	
	return data

def feature_create(data):
	# demand features
	data['lag_t28'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
	data['lag_t29'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(29))
	data['lag_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(30))
	data['rolling_mean_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
	data['rolling_std_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())
	data['rolling_mean_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
	data['rolling_mean_t90'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
	data['rolling_mean_t180'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
	data['rolling_std_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())
	
	# price features
	data['lag_price_t1'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
	data['price_change_t1'] = (data['lag_price_t1'] - data['sell_price']) / (data['lag_price_t1'])
	data['rolling_price_max_t365'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
	data['price_change_t365'] = (data['rolling_price_max_t365'] - data['sell_price']) / (data['rolling_price_max_t365'])
	data['rolling_price_std_t7'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
	data['rolling_price_std_t30'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
	data.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1)
	
	# time features
	data['date'] = pd.to_datetime(data['date'])
	data['year'] = data['date'].dt.year
	data['month'] = data['date'].dt.month
	data['week'] = data['date'].dt.week
	data['day'] = data['date'].dt.day
	data['dayofweek'] = data['date'].dt.dayofweek
	
	return data

def submit_format(test, submission):
    predictions = test[['id', 'date', 'demand']]
    predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    evaluation_rows = [row for row in submission['id'] if 'evaluation' in row] 
    evaluation = submission[submission['id'].isin(evaluation_rows)]
    validation = submission[['id']].merge(predictions, on = 'id')
    final_dataset = pd.concat([validation, evaluation])

    return final_dataset
