import sys
import numpy  as np
import os, gc
import cPickle
import copy
import logging
import threading
import Queue
import collections
import os
import os
import pdb
import pandas as pd

parent_path = "datasets/"
data_dim = 27
output_dim = 2

discontinue_index = 122

logger = logging.getLogger(__name__)


class SSFetcher(threading.Thread):
	def __init__(self, parent):
		threading.Thread.__init__(self)
		self.parent = parent
		np.random.shuffle(parent.short_stores)
		np.random.shuffle(parent.long_stores)

	def run(self):
		long_list_bool = True
		diter = self.parent
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
				X_batch = np.zeros((min(diter.batch_size,len(present_batch)),diter.max_time_step,data_dim),dtype=np.float32)
				Y_batch = np.zeros((min(diter.batch_size,len(present_batch)),diter.max_time_step,output_dim),dtype=np.float32)
				refresh = 0
				for b_index in range(min(diter.batch_size,len(present_batch))):
					for t_index in range(diter.max_time_step):
						data_item = diter.data_dict[present_batch[b_index]][time_steps_done+t_index]
						X_batch[b_index,t_index,:] = diter.store_dict[present_batch[b_index]] + data_item[:-2]
						Y_batch[b_index,t_index,:] = data_item[-2:]
						if time_steps_done + t_index == diter.long_length-1:
							batch_set = False
							refresh = 1
							break
				time_steps_done += t_index+1
				batch = {}
				batch['inputs'] = X_batch
				batch['outputs'] = Y_batch
				batch['refresh'] = refresh
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

			while batch_set:
				X_batch = np.zeros((min(diter.batch_size,len(present_batch)),diter.max_time_step,data_dim),dtype=np.float32)
				Y_batch = np.zeros((min(diter.batch_size,len(present_batch)),diter.max_time_step,output_dim),dtype=np.float32)
				refresh = 0
				for b_index in range(min(diter.batch_size,len(present_batch))):
					for t_index in range(diter.max_time_step):
						data_item = diter.data_dict[present_batch[b_index]][time_steps_done+t_index]
						X_batch[b_index,t_index,:] = diter.store_dict[present_batch[b_index]] + data_item[:-2]
						Y_batch[b_index,t_index,:] = data_item[-2:]
						if time_steps_done + t_index == diter.short_length-1:
							batch_set = False
							refresh = 1
							break
						if time_steps_done + t_index == discontinue_index + 1:
							refresh = 1
							break	
				time_steps_done += t_index+1
				batch = {}
				batch['inputs'] = X_batch
				batch['outputs'] = Y_batch
				batch['refresh'] = refresh
				refresh = 0
				diter.queue.put(batch)
		diter.queue.put(None)
		return



class SSIterator(object):
	def __init__(self,
				 batch_size,
				 config,
				 max_time_step,
				 seed,
				 use_infinite_loop=False,
				 dtype="int32"):

		self.batch_size = batch_size
		self.max_time_step = max_time_step
		self.config = config
		self.exit_flag = False
		self.load_files()
		

	def load_files(self):
		config = self.config
		file = open("shortlist.txt")
		self.short_stores = file.readlines()
		self.short_stores = [int(store.strip()) for store in self.short_stores]
		file = open("longlist.txt")
		self.long_stores = file.readlines()
		self.long_stores = [int(store.strip()) for store in self.long_stores]
		train_data = pd.read_csv(parent_path+"train_modified.csv",header = None)
		self.data_dict = {}
		for _,val in train_data.iterrows():
			val_list = val.tolist()
			try:
				self.data_dict[val_list[0]].append(val_list[1:])
			except KeyError:
				self.data_dict[val_list[0]] = [val_list[1:]]
		for item in self.data_dict:
			self.data_dict[item].reverse()
		store_data = pd.read_csv(parent_path+"store_modified.csv",header = None)
		self.store_dict = {}
		for _,val in store_data.iterrows():
			val_list = val.tolist()
			self.store_dict[val_list[0]] = val_list[1:]
	   
		self.long_length = len(self.data_dict[self.long_stores[0]])
		self.short_length = len(self.data_dict[self.short_stores[0]])

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
			# print("Okay")
		return batch
