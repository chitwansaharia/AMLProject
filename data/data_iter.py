import sys
import numpy  as np
import os, gc
import copy
import logging
import threading
import Queue
import collections
import os
import os
import pdb
import pandas as pd

parent_path = "/home/chitwan/Project/AMLProject/data/"

output_dim = 2

discontinue_index = 122

logger = logging.getLogger(__name__)




class SSFetcher(threading.Thread):
	def __init__(self, parent):
		threading.Thread.__init__(self)
		self.parent = parent


	def run(self):
		diter = self.parent
		if diter.mode == "valid" :
			np.random.shuffle(diter.stores)
			exit = False
			present_batch = []
			time_steps_done = 0
			batch_set = False
			offset = 0 
			while not exit:
				present_batch = diter.stores[offset:offset+diter.batch_size]
				offset += diter.batch_size
				batch_set = True
				time_steps_done = 0
				if len(present_batch) < diter.batch_size:
					exit = True
				while batch_set:
					X_batch = np.zeros((diter.batch_size,diter.max_time_steps,diter.config.input_size),dtype=np.float32)
					Y_batch = np.zeros((diter.batch_size,diter.max_time_steps,output_dim),dtype=np.float32)
					mask = np.zeros((diter.batch_size,diter.max_time_steps))
					refresh = 0
					for b_index in range(min(diter.batch_size,len(present_batch))):
						for t_index in range(diter.max_time_steps):
							data_item = diter.data_dict[present_batch[b_index]][time_steps_done+t_index]
							X_batch[b_index,t_index,:] = diter.store_dict[present_batch[b_index]] + data_item[:-2]
							Y_batch[b_index,t_index,:] = data_item[-2:]
							mask[b_index,t_index] = 1.0
							if time_steps_done + t_index == diter.length-1:
								batch_set = False
								refresh = 1
								break
					time_steps_done += t_index+1
					batch = {}
					batch['inputs'] = X_batch
					batch['outputs'] = Y_batch
					batch['refresh'] = refresh
					batch['mask'] = mask
					refresh = 0
					diter.queue.put(batch)
			diter.queue.put(None)
			return

		elif diter.mode == "train":
			np.random.shuffle(diter.short_stores)
			np.random.shuffle(diter.long_stores)
			long_list_bool = True
			offset = 0
			batch_set = False 
			present_batch = []
			time_steps_done = 0
			while long_list_bool:
				present_batch = diter.long_stores[offset:offset+diter.batch_size]
				offset += diter.batch_size
				batch_set = True
				time_steps_done = 0
				if len(present_batch) < diter.batch_size:
					long_list_bool = False

				while batch_set:
					X_batch = np.zeros((diter.batch_size,diter.max_time_steps,diter.config.input_size),dtype=np.float32)
					Y_batch = np.zeros((diter.batch_size,diter.max_time_steps,output_dim),dtype=np.float32)
					mask = np.zeros((diter.batch_size,diter.max_time_steps))
					refresh = 0
					for b_index in range(min(diter.batch_size,len(present_batch))):
						for t_index in range(diter.max_time_steps):
							data_item = diter.data_dict[present_batch[b_index]][time_steps_done+t_index]
							X_batch[b_index,t_index,:] = diter.store_dict[present_batch[b_index]] + data_item[:-2]
							Y_batch[b_index,t_index,:] = data_item[-2:]
							mask[b_index,t_index] = 1.0
							if time_steps_done + t_index == diter.long_length-1:
								batch_set = False
								refresh = 1
								break
					time_steps_done += t_index+1
					batch = {}
					batch['inputs'] = X_batch
					batch['outputs'] = Y_batch
					batch['refresh'] = refresh
					batch['mask'] = mask
					refresh = 0
					diter.queue.put(batch)

			offset = 0
			present_batch = []
			while not long_list_bool:
				present_batch = diter.short_stores[offset:offset+diter.batch_size]
				offset += diter.batch_size
				batch_set = True
				time_steps_done = 0
				
				if len(present_batch) < diter.batch_size:
					long_list_bool = True
					if len(present_batch) == 0:
						batch_set = False


				while batch_set:
					X_batch = np.zeros((diter.batch_size,diter.max_time_steps,diter.config.input_size),dtype=np.float32)
					Y_batch = np.zeros((diter.batch_size,diter.max_time_steps,output_dim),dtype=np.float32)
					mask = np.zeros((diter.batch_size,diter.max_time_steps))
					refresh = 0
					for b_index in range(min(diter.batch_size,len(present_batch))):
						for t_index in range(diter.max_time_steps):
							data_item = diter.data_dict[present_batch[b_index]][time_steps_done+t_index]
							X_batch[b_index,t_index,:] = diter.store_dict[present_batch[b_index]] + data_item[:-2]
							Y_batch[b_index,t_index,:] = data_item[-2:]
							mask[b_index,t_index] = 1.0
							if time_steps_done + t_index == diter.short_length-1:
								batch_set = False
								refresh = 1
								break
							if time_steps_done + t_index == discontinue_index - 1:
								refresh = 1
								break	
					time_steps_done += t_index+1
					batch = {}
					batch['inputs'] = X_batch
					batch['outputs'] = Y_batch
					batch['refresh'] = refresh
					batch['mask'] = mask
					refresh = 0
					diter.queue.put(batch)
			diter.queue.put(None)
			return
		if diter.mode == "test":
			exit = False
			present_batch = []
			time_steps_done = 0
			batch_set = False
			offset = 0 
			while not exit:
				present_batch = diter.stores[offset:offset+diter.batch_size]
				offset += diter.batch_size
				batch_set = True
				time_steps_done = 0
				if len(present_batch) < diter.batch_size:
					exit = True
				while batch_set:
					X_batch = np.zeros((diter.batch_size,diter.max_time_steps,diter.config.input_size),dtype=np.float32)
					mask = np.zeros((diter.batch_size,diter.max_time_steps))
					refresh = 0
					for b_index in range(min(diter.batch_size,len(present_batch))):
						for t_index in range(diter.max_time_steps):
							data_item = diter.data_dict[present_batch[b_index]][time_steps_done+t_index]
							X_batch[b_index,t_index,:] = diter.store_dict[present_batch[b_index]] + data_item
							mask[b_index,t_index] = 1.0
							if time_steps_done + t_index == diter.length-1:
								batch_set = False
								refresh = 1
								break
					time_steps_done += t_index+1
					batch = {}
					batch['inputs'] = X_batch
					batch['refresh'] = refresh
					batch['stores'] = present_batch
					batch['mask'] = mask
					refresh = 0
					diter.queue.put(batch)
			diter.queue.put(None)
			return





class SSIterator(object):
	def __init__(self,
				 config,
				 mode = "train",
				 dtype="int32"):

		self.config = config
		self.batch_size = config.batch_size
		self.max_time_steps = config.max_time_steps
		self.exit_flag = False
		self.mode = mode
		if self.mode == "train":
			self.load_train_files()
		elif self.mode == "valid":
			self.load_valid_files()
		else:
			self.load_test_files()
		

	def load_train_files(self):
		config = self.config
		file = open(parent_path+"shortlist.txt")
		self.short_stores = file.readlines()
		self.short_stores = [int(store.strip()) for store in self.short_stores]
		file = open(parent_path+"longlist.txt")
		self.long_stores = file.readlines()
		self.long_stores = [int(store.strip()) for store in self.long_stores]
		train_data = pd.read_csv(parent_path+"datasets/train_modified.csv",header = None)
		self.data_dict = {}
		for _,val in train_data.iterrows():
			val_list = val.tolist()
			val_list[1] -= 1
			try:
				self.data_dict[val_list[0]].append(val_list[1:])
			except KeyError:
				self.data_dict[val_list[0]] = [val_list[1:]]
		for item in self.data_dict:
			self.data_dict[item].reverse()
		store_data = pd.read_csv(parent_path+"datasets/store_modified.csv",header = None)
		self.store_dict = {}
		for _,val in store_data.iterrows():
			val_list = val.tolist()
			self.store_dict[val_list[0]] = val_list[1:]
	   
		self.long_length = len(self.data_dict[self.long_stores[0]])
		self.short_length = len(self.data_dict[self.short_stores[0]])


	def load_valid_files(self):
		config = self.config
		self.stores = range(1,1116)
		valid_data = pd.read_csv(parent_path+"datasets/val_modified.csv",header = None)
		self.data_dict = {}
		for _,val in valid_data.iterrows():
			val_list = val.tolist()
			val_list[1] -= 1
			try:
				self.data_dict[val_list[0]].append(val_list[1:])
			except KeyError:
				self.data_dict[val_list[0]] = [val_list[1:]]
		for item in self.data_dict:
			self.data_dict[item].reverse()
		store_data = pd.read_csv(parent_path+"datasets/store_modified.csv",header = None)
		self.store_dict = {}
		for _,val in store_data.iterrows():
			val_list = val.tolist()
			self.store_dict[val_list[0]] = val_list[1:]
		self.length = len(self.data_dict[1])

	def load_test_files(self):
		config = self.config
		data = pd.read_csv(parent_path+"datasets/test_modified.csv",header = None)
		self.data_dict = {}
		for _,val in data.iterrows():
			val_list = val.tolist()
			val_list[1] -= 1
			try:
				self.data_dict[val_list[0]].append(val_list[1:])
			except KeyError:
				self.data_dict[val_list[0]] = [val_list[1:]]
		for item in self.data_dict:
			self.data_dict[item].reverse()
		store_data = pd.read_csv(parent_path+"datasets/store_modified.csv",header = None)
		self.store_dict = {}
		for _,val in store_data.iterrows():
			val_list = val.tolist()
			self.store_dict[val_list[0]] = val_list[1:]
		self.length = len(self.data_dict[1])
		self.stores = self.data_dict.keys()
	   

	def start(self):
		self.exit_flag = False
		self.queue = Queue.Queue(maxsize = 100)
		self.gather = SSFetcher(self)
		self.gather.daemon = True
		self.gather.start()

	def __del__(self):
		if hasattr(self, 'gather'):
			self.gather.exitFlag = True
			self.gather.join()

	def __iter__(self):
		return self

	def next(self):
		if self.exit_flag:
			return None
		
		batch = self.queue.get()
		if not batch:
			self.exit_flag = True
		return batch
