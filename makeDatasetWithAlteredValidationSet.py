from __future__ import division
import pandas as pd
import numpy as np
#from sklearn.preprocessing import LabelEncoder

LABEL = 'returnQuantity'

def extract_non_probabilities_features(df):

	# Deletions
	# print "\t\t WARNING: Script ini hapus: deviceID, voucherID"
	# del df['deviceID']
	# del df['voucherID']

	# Total price for an order
	order_total_dict = df[['orderID', 'price']].groupby('orderID').sum()['price'].to_dict()
	df['order_total'] = df.orderID.apply(order_total_dict.get).astype(np.float32)
	del order_total_dict

	# Average budget of the customer
	customer_budget_dict = df[['customerID', 'order_total']].groupby('customerID').mean()['order_total'].to_dict()
	df['customer_budget'] = df.customerID.apply(customer_budget_dict.get).astype(np.float32)
	del customer_budget_dict

	# Customer expense ratio	
	total_spent_dict = df[['customerID', 'order_total']].groupby('customerID').sum()['order_total'].to_dict()
	df['total_spent'] = df.customerID.apply(total_spent_dict.get).astype(np.float32)
	del total_spent_dict

	df['expense_ratio'] = (df['customer_budget'] / df['total_spent']).astype(np.float16)

	# Total_spent dihapus. Kalau nggak mau dihapus, comment aja
	del df['total_spent']


	# 2 baris ini untuk mencegah `quantity = 0`, para infaqers
	temp_quantity = df.quantity.copy()
	temp_quantity[temp_quantity==0] = 1

	# unit_price = price / quantity. by @amirahff
	df['unit_price'] = (df.price/temp_quantity).astype(np.float32)
	del temp_quantity

	# Median unit price, the usual unit price
	median_unit_price_dict = df[['articleID', 'unit_price']].groupby('articleID').median().unit_price.to_dict()
	df['median_unit_price']=df.articleID.apply(median_unit_price_dict.get).astype(np.float32)
	del median_unit_price_dict

	df['price_diff'] = (df['unit_price']-df.median_unit_price).astype(np.float32)

	# Price after discount = order_total - voucherAmount
	df['after_voucher'] = df.order_total - df.voucherAmount

	# Orders (as in rank)
	df['order_order']  = df[['customerID', 'orderID']].groupby(['customerID']).cumcount() + 1
	df['choice_order'] = df[['orderID', 'articleID']].groupby(['orderID']).cumcount() + 1
	
	# Reduce float/int precision
	float_64_columns = df.loc[:, df.dtypes == np.float64].columns
	for col in float_64_columns:
		df[col] = df[col].astype(np.float32)

	int_64_columns = df.loc[:, df.dtypes == np.int64].columns
	for col in int_64_columns:
		df[col] = df[col].astype(np.int32)

	return df
        
def extract_prob(df):

	def append_return_prob(df, input_column):
		target_column_name = input_column + '_prob'

		df_temp = df[[input_column, 'returnQuantity','quantity']]
		df_return_probability = df_temp.groupby(input_column)['returnQuantity','quantity'].cumsum()

		df_return_probability[ target_column_name ]  = df_return_probability.returnQuantity / df_return_probability.quantity
		df[ target_column_name ] = df_return_probability[ target_column_name ].replace(np.NaN, 0.5).replace(np.inf, 0.5).apply(lambda x: 1 if x > 1 else x)
		
		del df_return_probability
        
	def append_return_prob_from_multiple_column(df, input_columns):
		column_prefix = ''
		for i in range(0,len(input_columns)):
			column_prefix = column_prefix + input_columns[i][0]+ input_columns[i][1]
		target_column_name = column_prefix + '_prob' 

		df_temp = df[ input_columns + ['returnQuantity','quantity'] ]

		df_return_probability = df_temp.groupby(input_columns).cumsum()
		df_return_probability[ target_column_name ]  = df_return_probability.returnQuantity / df_return_probability.quantity

		df[ target_column_name ] = df_return_probability[ target_column_name ].replace(np.NaN, 0.5).replace(np.inf, 0.5).apply(lambda x: 1 if x > 1 else x)

		del df_return_probability

	oneFeature = ['articleID','colorCode','sizeCode','productGroup','voucherID','customerID','deviceID','paymentMethod']
	#cumprob for one feature
	for x in oneFeature:
		append_return_prob(df, x)
	# oneFeature = ['articleID','colorCode','sizeCode','productGroup','voucherID']
	#cumprob for two features
	for i in range(0,len(oneFeature)):
		for j in range(i+1,len(oneFeature)):
			komb = [oneFeature[i], oneFeature[j]]
			append_return_prob_from_multiple_column(df, komb)
    
	#cumprob for three features
	for i in range(0,len(oneFeature)):
		for j in range(i+1,len(oneFeature)):
			for k in range(j+1,len(oneFeature)):
				komb = [oneFeature[i], oneFeature[j], oneFeature[k]]
				append_return_prob_from_multiple_column(df, komb)
                        
	del oneFeature
    
	return df

def append_prob_to_tests_df(train_df, tests_df):

	def append_return_prob(column, train_df, tests_df):
		df2 = train_df[[column,'returnQuantity','quantity']]
		df_return_probability = df2.groupby(column).sum()
		df_return_probability[ column + '_prob' ]  = df_return_probability.returnQuantity / df_return_probability.quantity
		return_prob_dict = df_return_probability[ column + '_prob' ].to_dict()
		del df_return_probability
		tests_df[ column + '_prob' ] = tests_df[column].apply(return_prob_dict.get).replace(np.NaN, 0.5).replace(np.inf, 0.5).apply(lambda x: 1 if x > 1 else x)
		del return_prob_dict
        
	def append_return_prob_from_multiple_column(input_columns, input_df, target_df):
		column_prefix = ''
		for i in range(0,len(input_columns)):
			column_prefix = column_prefix + input_columns[i][0] + input_columns[i][1]
		target_column_name = column_prefix + '_prob' 

		print "\t\t append_return_prob_from_two_column:", input_columns, ' -> ',target_column_name

		df_temp = input_df[ input_columns + ['returnQuantity','quantity'] ]

		df_return_probability = df_temp.groupby(input_columns).sum()
		df_return_probability[ target_column_name ]  = df_return_probability.returnQuantity / df_return_probability.quantity

		prob_dict  = df_return_probability[target_column_name].to_dict()
		target_df[target_column_name] = target_df[input_columns].apply(tuple, axis=1).apply(prob_dict.get).replace(np.nan, 0.5).replace(np.inf, 0.5)

		del df_return_probability

	oneFeature = ['articleID','colorCode','sizeCode','productGroup','voucherID','customerID','deviceID','paymentMethod']
	#prob for one feature to test
	for x in oneFeature:
		append_return_prob(x, train_df, tests_df)


	# oneFeature = ['articleID','colorCode','sizeCode','productGroup','voucherID']
 
	#cumprob for two features to test
	for i in range(0,len(oneFeature)):
		for j in range(i+1,len(oneFeature)):
			komb = [oneFeature[i], oneFeature[j]]
			append_return_prob_from_multiple_column(komb, input_df=train_df, target_df=tests_df)
    
	#cumprob for three features to test
	for i in range(0,len(oneFeature)):
		for j in range(i+1,len(oneFeature)):
			for k in range(j+1,len(oneFeature)):
				komb = [oneFeature[i], oneFeature[j], oneFeature[k]]
				append_return_prob_from_multiple_column(komb, input_df=train_df, target_df=tests_df)
                        
	del oneFeature
    
	return tests_df

def main():
	from datetime import datetime
	time = lambda: datetime.now().time()

	LABEL = 'returnQuantity'

	print 'ETA: ~18 minutes'

	#
	print time(), 'Loading data.'
	train_df = pd.read_csv('orders_train.txt', sep=';')
	tests_df = pd.read_csv('orders_class.txt', sep=';')
	df = pd.concat([train_df, tests_df], axis=0, ignore_index=True)

	# split at the end of train_df index
	split = train_df.shape[0]

	# delete train_df & tests_df
	#del train_df
	#del tests_df

	#
	print time(), 'Extracting non-probabilities features.'
	df = extract_non_probabilities_features(df)

	#
	print time(), 'Splitting dataframes.'

	train_df = df[:split]
	tests_df = df[split:].drop(LABEL, axis=1)

	#
	print time(), 'Extracting cumprob to train_df. (SLOW)'
	train_df = extract_prob(train_df)

	#
	print time(), 'Extracting prob to tests_df. (SLOW)'
	tests_df = append_prob_to_tests_df(train_df, tests_df)
 
	val_df=train_df[-341099:].copy()

	val_df = append_prob_to_tests_df(train_df[:-341099], val_df)

	#
	print time(), 'Writing to CSVs. ETA 7 minutes'

	train_df.to_csv('train_gue6.csv', index=False)
	tests_df.to_csv('tests_gue6.csv', index=False)
	val_df.to_csv('val_gue6.csv', index=False)
	#
	print time(), 'Done'

if __name__ == "__main__":
	main()