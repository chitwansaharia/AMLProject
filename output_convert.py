import pandas as pd
import csv
test_data = pd.read_csv('data/datasets/test_modified.csv',header = None)
train = pd.read_csv('data/train.csv')
mean = train['Sales'].mean()
std = train['Sales'].std()


finaldict = {}

outputs = pd.read_csv('output.csv',header=None)
for _,val in outputs.iterrows():
	val_list = val.tolist()
	val_list[1] += 1
	temp = test_data.loc[test_data[0] == val_list[0]].loc[test_data[4] == val_list[4]].loc[test_data[3] == val_list[3]].index
	if len(temp) != 1:
		print("Error")
		import pdb;pdb.set_trace()
	else:
		finaldict[temp[0]+1] = val_list[-1]*std+mean

temp = map(list,finaldict.items())
with open("submission.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(temp)


		
