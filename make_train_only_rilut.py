from __future__ import division
import pandas as pd
import numpy as np
#from sklearn.preprocessing import LabelEncoder

LABEL = 'returnQuantity'

def extract_non_probabilities_features(df):

	# Deletions
	print "\t\t WARNING: Script ini hapus: deviceID, voucherID"
	del df['deviceID']
	del df['voucherID']

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

	# def lowerOrHigher(row):
	# 	if row['unit_price']<row['median_unit_price']:
	# 		return -1
	# 	elif row['unit_price']>row['median_unit_price']:
	# 		return 1
	# 	else:
	# 		return 0

	# Months
	# df['mmdd'] = df.orderDate.str[-5:]
	# df['months'] = df.orderDate.str[-5:-3].astype(np.int8)

	return df

def extract_cumprob(df):

	def append_return_cumprob(df, input_column):
		target_column_name = input_column + '_cumprob'

		df_temp = df[[input_column, 'returnQuantity','quantity']]
		df_return_probability = df_temp.groupby(input_column)['returnQuantity','quantity'].cumsum()

		df_return_probability[ target_column_name ]  = df_return_probability.returnQuantity / df_return_probability.quantity
		df[ target_column_name ] = df_return_probability[ target_column_name ].replace(np.NaN, 0.5).replace(np.inf, 0.5).apply(lambda x: 1 if x > 1 else x)
		
		del df_return_probability

	append_return_cumprob(df, 'articleID')
	append_return_cumprob(df, 'colorCode')
	append_return_cumprob(df, 'customerID')
	append_return_cumprob(df, 'sizeCode')

	def append_return_prob_from_two_column(df, input_columns = ['articleID','colorCode']):
		column_prefix = input_columns[0][0] + input_columns[1][0]
		target_column_name = column_prefix + '_prob' 

		print "\t\t append_return_prob_from_two_column:", input_columns, ' -> ',target_column_name

		df_temp = df[ input_columns + ['returnQuantity','quantity'] ]

		df_return_probability = df_temp.groupby(input_columns).sum()
		df_return_probability[ target_column_name ]  = df_return_probability.returnQuantity / df_return_probability.quantity
		
		prob_dict  = df_return_probability[target_column_name].to_dict()
		df[target_column_name] = df[input_columns].apply(tuple, axis=1).apply(prob_dict.get).replace(np.nan, 0.5).replace(np.inf, 0.5)

		del df_return_probability

	append_return_prob_from_two_column(df, ['articleID', 'colorCode'])
	append_return_prob_from_two_column(df, ['articleID', 'sizeCode'])

	return df

def main():
	from datetime import datetime
	time = lambda: datetime.now().time()

	print "ETA: ~12 minutes"

	print time(), 'Loading data. (ETA: 30 seconds)'
	train_df = pd.read_csv('orders_train.txt', sep=';')

	print time(), 'Extracting non-probabilities features. (ETA: 1 minute)'
	train_df = extract_non_probabilities_features(train_df)

	print time(), 'Preprocessing cumprob. (SLOW)'
	train_df = extract_cumprob(train_df)

	print time(), 'Writing to CSV. (ETA: 5 minutes 30 seconds)'
	train_df.to_csv('train_gue.csv', index=False)
	
	print time(), 'Done'

if __name__ == "__main__":
	main()
