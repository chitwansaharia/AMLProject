import pandas as pd
import numpy as np
import csv
import pickle

# flag = 1
# if flag == 0:
	# data = pd.read_csv('train.csv')
	# data['Sales'] = (data['Sales']-data['Sales'].mean())/data['Sales'].std()
	# data['Customers'] = (data['Customers']-data['Customers'].mean())/data['Customers'].std()

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

data['CompetitionOpenSinceYear'].fillna(0,inplace=True)
data.loc[data['CompetitionOpenSinceYear'] == 1900, 'CompetitionOpenSinceYear'] = 1
data.loc[data['CompetitionOpenSinceYear']
         == 1961, 'CompetitionOpenSinceYear'] = 2
data.loc[data['CompetitionOpenSinceYear']
         >= 1900, 'CompetitionOpenSinceYear'] -= 1987 


data['Promo2SinceWeek'].fillna(0,inplace=True)
data.loc[data['Promo2SinceWeek'] != 0,'Promo2SinceWeek'] = ((data['Promo2SinceWeek']-1)//4 + 1)


min_year = data['Promo2SinceYear'].min()
data['Promo2SinceYear'].fillna(min_year-1,inplace=True)
data['Promo2SinceYear'] = data['Promo2SinceYear']-min_year + 1


data['PromoInterval'].fillna(0,inplace=True)
data.loc[data['PromoInterval']
         == "Jan,Apr,Jul,Oct", "PromoInterval"] = 1
data.loc[data['PromoInterval']
         == "Feb,May,Aug,Nov", "PromoInterval"] = 2
data.loc[data['PromoInterval']
         == "Mar,Jun,Sept,Dec", "PromoInterval"] = 3


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
	list_item.append(int(data_item['Assortment']))
	list_item.append(data_item['CompetitionDistance'])
	list_item.append(int(data_item["CompetitionOpenSinceMonth"]))
	list_item.append(int(data_item["CompetitionOpenSinceYear"]))
	list_item.append(int(data_item["Promo2"]))
	list_item.append(int(data_item['Promo2SinceWeek']))
	list_item.append(int(data_item["Promo2SinceYear"]))
	list_item.append(int(data_item['PromoInterval']))
	writer.writerow(list_item)   







