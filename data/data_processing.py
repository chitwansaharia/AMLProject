import os
import numpy as np

train_filepath = "data/train.csv"
test_filepath = "data/test.csv"
store_filepath = "data/store.csv"

class DataProcess():

	def __init__(self, ):
		pass

	def read_data(self, data_dir=None):
		train_data = np.loadtxt(train_filepath, dtype=str, skiprows=1, delimiter=',')
		print(train_data.shape)

if __name__ == "__main__":
	data_process = DataProcess()
	data_process.read_data()