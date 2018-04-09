import pandas as pd
import numpy as np
import csv
import pickle

flag = 0
if flag == 0:
	data = pd.read_csv('train.csv')
	mean_sales,std_sales = data['Sales'].mean(),data['Sales'].std()
	mean_customers,std_customers = data['Customers'].mean(),data['Customers'].std()
	data['Sales'] = (data['Sales']-data['Sales'].mean())/data['Sales'].std()
	data['Customers'] = (data['Customers']-data['Customers'].mean())/data['Customers'].std()
	with open("mean_std.pkl","w") as f:
		pickle.dump([[mean_sales,std_sales],[mean_customers,std_customers]],f)

else:
	data = pd.read_csv('test.csv')

for i in range(3):
	data['StateHoliday'].replace(chr(97+i),i+1,inplace=True)
data['StateHoliday'].replace('0',0,inplace = True)


def encode_date(date):
	date = date.split('-')
	year = [0]*3
	month = [0]*12
	day = [0]*31
	year[int(date[0])-2013] = 1
	month[int(date[1])-1] = 1
	day[int(date[2])-1] = 1
	return year + month + day

def encode_onehot(column):
	# import pdb;pdb.set_trace()
	min_val = column.min()
	num_val = column.max()-column.min()+1
	final_list = []
	for val in column:
		one_hot = [0]*num_val
		one_hot[val-min_val] = 1
		final_list.append(one_hot)
	return final_list

day_of_week = encode_onehot(data['DayOfWeek'])
state_holiday = encode_onehot(data['StateHoliday'])
train_modified = open('train_modified.csv','w')
writer = csv.writer(train_modified, lineterminator='\n')


for index in range(len(data)):
	data_item = data.iloc[index]
	list_item = []
	list_item.append(data_item['Store'])
	list_item += day_of_week[index]
	list_item += encode_date(data_item['Date'])
	list_item += [data_item['Customers']]
	list_item += [data_item['Open']]
	list_item += [data_item["Promo"]]
	list_item += state_holiday[index]
	list_item += [data_item["SchoolHoliday"]]
	list_item += [data_item['Sales']]
	writer.writerow(list_item)   







