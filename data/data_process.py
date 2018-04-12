import pandas as pd
import numpy as np
import csv
import pickle

# flag = 1
# if flag == 0:
# 	data = pd.read_csv('train.csv')
# 	data['Sales'] = (data['Sales']-data['Sales'].mean())/data['Sales'].std()
# 	data['Customers'] = (data['Customers']-data['Customers'].mean())/data['Customers'].std()

# else:
# 	data = pd.read_csv('test.csv')
# 	data['Open'].replace(float('nan'),1,inplace = True)  


data=  pd.read_csv('store.csv')
for i in range(4):
	data['StoreType'].replace(chr(97+i),i,inplace=True)

for i in range(3):
	data['Assortment'].replace(chr(97+i),i,inplace=True)

data['CompetitionDistance'].fillna(0,inplace=True)
data['CompetitionDistance'] = (data['CompetitionDistance']-data['CompetitionDistance'].mean())/data['CompetitionDistance'].std()

data['CompetitionOpenSinceMonth'].fillna(0,inplace=True)

min_year = data['CompetitionOpenSinceYear'].min()
data['CompetitionOpenSinceYear'].fillna(min_year-1,inplace=True)
data['CompetitionOpenSinceYear'] = data['CompetitionOpenSinceYear']-min_year+1

data['Promo2SinceWeek'].fillna(0,inplace=True)
data['Promo2SinceWeek'] = (data['Promo2SinceWeek']-data['Promo2SinceWeek'].mean())/data['Promo2SinceWeek'].std()

min_year = data['Promo2SinceYear'].min()
data['Promo2SinceYear'].fillna(min_year-1,inplace=True)
data['Promo2SinceYear'] = data['Promo2SinceYear']-min_year + 1

import pdb

year = {'Jan' : 0,'Feb' : 1,'Mar' : 2,'Apr' : 3,'May' : 4,'Jun' : 5,'Jul' : 6,'Aug' : 7,'Sept' : 8,'Oct' : 9,'Nov' : 10,'Dec' : 11}
data['PromoInterval'].fillna("",inplace=True)

promo_interval = []

import pdb;
 # pdb.set_trace()


for val in data['PromoInterval']:
	one_hot = [0]*12
	if len(val) != 0:
		month_list = val.split(',')
		for month in month_list:
			one_hot[year[month]] = 1
	promo_interval.append(one_hot)


def encode_date(date):
	date = date.split('-')
	year = int(date[0])-2013
	month = int(date[1])-1
	day = int(date[2])-1
	return [year, month, day]


store_modified = open('store_modified.csv','w')
writer = csv.writer(store_modified, lineterminator='\n')


for index in range(len(data)):
	data_item = data.iloc[index]
	list_item = []
	list_item.append(data_item['Store'])
	list_item.append(int(data_item['StoreType']))
	list_item.append(data_item['CompetitionDistance'])
	list_item.append(int(data_item["CompetitionOpenSinceMonth"]))
	list_item.append(int(data_item["CompetitionOpenSinceYear"]))
	list_item.append(int(data_item["Promo2"]))
	list_item.append(data_item['Promo2SinceWeek'])
	list_item.append(int(data_item["Promo2SinceYear"]))
	list_item.extend(promo_interval[index])
	writer.writerow(list_item)   







