# Updated ocassionally

from __future__ import division
import pandas as pd
import numpy as np
#from sklearn.preprocessing import LabelEncoder

LABEL = 'returnQuantity'

def preprocess(df):

	print 'Minimal 12 menitan'
	# Deletions
	print "WARNING: Script ini hapus: deviceID, voucherID"
	del df['deviceID']
	del df['voucherID']

	# Months
	df['mmdd'] = df.orderDate.str[-5:]
	df['months'] = df.orderDate.str[-5:-3].astype(np.int8)

	# Total price for an order
	order_total_dict = df[['orderID', 'price']].groupby('orderID').sum()['price'].to_dict()
	df['order_total'] = df.orderID.apply(order_total_dict.get).astype(np.float32)
	del order_total_dict

	# Average budget of the customer
	customer_budget_dict = df[['customerID', 'order_total']].groupby('customerID').mean()['order_total'].to_dict()
	df['customer_budget'] = df.customerID.apply(customer_budget_dict.get).astype(np.float32)
	del customer_budget_dict

	# Price after rebate = order_total - voucherAmount
	df['after_voucher'] = df.order_total - df.voucherAmount

	# Orders (as in rank)
	df['order_order']  = df[['customerID', 'orderID']].groupby(['customerID']).cumcount() + 1
	df['choice_order'] = df[['orderID', 'articleID']].groupby(['orderID']).cumcount() + 1

	# Return Probabilities

	def append_return_prob(df, column):
		df2 = df[[column,'returnQuantity','quantity']]
		df_return_probability = df2.groupby(column).sum()
		df_return_probability[ column + '_prob' ]  = df_return_probability.returnQuantity / df_return_probability.quantity
		return_prob_dict = df_return_probability[ column + '_prob' ].to_dict()
		del df_return_probability
		df[ column + '_prob' ] = df[column].apply(return_prob_dict.get).replace(np.NaN, 0.5).replace(np.inf, 0.5)
		del return_prob_dict

	def append_return_cumprob(df, column):
		df2 = df[[column,'returnQuantity','quantity']]
		df_return_probability = df2.groupby(column).sum()
		df_return_probability[ column + '_prob' ]  = df_return_probability.returnQuantity / df_return_probability.quantity
		return_prob_dict = df_return_probability[ column + '_prob' ].to_dict()
		del df_return_probability
		df[ column + '_cumprob' ] = df[column].apply(return_prob_dict.get).replace(np.NaN, 0.5).replace(np.inf, 0.5)
		del return_prob_dict

	append_return_prob(df, 'articleID')
	append_return_prob(df, 'colorCode')
	append_return_prob(df, 'customerID')
	append_return_prob(df, 'sizeCode')
	
	float_64_columns = df.loc[:, df.dtypes == np.float64].columns
	for col in float_64_columns:
		df[col] = df[col].astype(np.float16)

	int_64_columns = df.loc[:, df.dtypes == np.int64].columns
	for col in int_64_columns:
		df[col] = df[col].astype(np.int32)

	# Article & Color
	column='ac'
	columns=['articleID','colorCode']
	df2 = df[['articleID','colorCode','returnQuantity','quantity']]
	df_return_probability = df2.groupby(columns).sum()
	df_return_probability[ 'ac' + '_prob' ]  = df_return_probability.returnQuantity / df_return_probability.quantity
	ac_prob_dict=df_return_probability.ac_prob.to_dict()
	df['ac_prob']=df[['articleID','colorCode']].apply(tuple, axis=1).apply(ac_prob_dict.get).replace(np.nan, 0.5).replace(np.inf, 0.5)

	# Article & Size
	columns=['articleID','sizeCode']
	df2 = df[['articleID','sizeCode','returnQuantity','quantity']]
	df_return_probability = df2.groupby(columns).sum()
	df_return_probability[ 'as' + '_prob' ]  = df_return_probability.returnQuantity / df_return_probability.quantity
	as_prob_dict=df_return_probability.as_prob.to_dict()
	df['as_prob']=df[['articleID','colorCode']].apply(tuple, axis=1).apply(ac_prob_dict.get).replace(np.nan, 0.5).replace(np.inf, 0.5)

	#df.productGroup = df.productGroup.astype(np.int8)	

	# Iseng ubah float ke integer.
	# df.price = df.price.astype(np.int32)
	# df.budget = df.budget.astype(np.int32)

	# Is it Wednesday?
	#print("[SLOW] Get weekday")
	#df['weekday'] = df.orderDate.apply(pd.to_datetime).apply(lambda x: x.weekday())
	#df['wednesday'] = 0
	#df['wednesday'][df.weekday==2] = 1

	# Konversi data bertipe kategori/object ke numerik. Komen baris ini hingga blok for kalau tidak ingin konversi data bertipe kategori
	# print("Konversi kategori/object ke numerik:")
	# Cari kolom yang tipenya object, bukan integer maupun float
	# object_columns = df.loc[:, df.dtypes == object].columns
	# for col in object_columns:
	#     print(col)
	#     le = LabelEncoder()
	#     # Konversi deh
	#     df[col] = le.fit_transform(df[col])

	# Unused functions
	# df['cSizeCode'] = df.sizeCode.apply(changeSizeCode)
	# df['season'] = df.months.apply(getSeason)
	# df['priceDiff'] = df['rrp'] - (df['price'] / df['quantity'])
	# df['gpriceDiff'] = df['priceDiff'].apply(grouppriceDiff)

	return df

def main():
	from datetime import datetime
	print datetime.now()
	train_df = pd.read_csv('orders_train.txt', sep=';')
	train_df = preprocess(train_df)
	train_df.to_csv('train_gue.csv', index=False)
	print datetime.now()

if __name__ == "__main__":
	main()
