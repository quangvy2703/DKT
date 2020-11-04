import pandas as pd
import os
from tqdm import tqdm
import csv
import random
import pickle as pkl
idx = 0

is_train = False

_skip_rows = 0
n_users_train = 50000
n_users_val = 25000
n_users_test = 50000
_n_rows = 10**6


last_user_id = -1
processed_user = []
user_data = {}
# user_data['group_num']= []
user_data['qid'] = []
user_data['correctness']= []
if is_train:
	data_file = "KT_benchmark_dataset/riiid/train.csv"
else:
	data_file = "KT_benchmark_dataset/riiid/example_test.csv"

if is_train:
	while True:
		trains = []
		_skip_rows = 1 if idx == 0 else idx * _n_rows
		data = pd.read_csv(
	    data_file, 
	    low_memory=False, 
	    nrows=_n_rows, 
	    skiprows=_skip_rows
		)
		
		for i, row in tqdm(data.iterrows(), desc="Reading riiid data from {}-{}".format(idx*_n_rows, (idx + 1) * _n_rows)):
			user_id = row[2]
			if last_user_id != user_id and last_user_id != -1:
				df = pd.DataFrame(user_data)
				if last_user_id in processed_user:
					df.to_csv("KT_benchmark_dataset/riiid/data_train/{}.csv".format(last_user_id),
							  mode='a', header=False, index=False)
				else:
					df.to_csv("KT_benchmark_dataset/riiid/data_train/{}.csv".format(last_user_id), index=False)
					processed_user.append(last_user_id)
				user_data['qid'] = []
				user_data['correctness']= []
				# user_data['group_num'] = []

			last_user_id = user_id if last_user_id is not user_id else last_user_id

			question_id = row[3]
			correctly = row[7]
			group = row[1]
			if correctly != -1:
				user_data['qid'].append(question_id)
				user_data['correctness'].append(correctly)
				# user_data['group_num'].append(group)

		idx += 1

if is_train is False:
	train_files = os.listdir("KT_benchmark_dataset/riiid/data_train")
	test_files = pd.read_csv(
	    data_file, 
	    low_memory=False, 
	    skiprows=1
		)
	total_questions = 0
	last_user_id = -1
	idx = 0
	questions = []
	groups = []
	num_tests = {}
	num_user = test_files.shape[0]
	for i, row in tqdm(test_files.iterrows(), desc="Reading riiid testing data"):
		row_id = row[0]
		group_num = row[1]
		user_id = row[3]
		question_id = row[4]

		questions = {'qid': [question_id], 'correctness': [0]}

		if i == 0:
			last_user_id = user_id
		
		if last_user_id == user_id:
			idx += 1
		else:
			idx = 0
			last_user_id = user_id

		new_records = pd.DataFrame(questions)
		#user_id-group_id-row_id-idx

		if os.path.isfile('KT_benchmark_dataset/riiid/data_train/{}.csv'.format(user_id)) is False:
			print(os.path.isfile('KT_benchmark_dataset/riiid/data_train/{}.csv'.format(user_id)))
			new_records.to_csv('KT_benchmark_dataset/riiid/data_test/{}_{}_{}_{}.csv'.format(user_id, group_num, row_id, idx),
				   			 	index=False)

		else:
			os.system('cp KT_benchmark_dataset/riiid/data_train/{}.csv '
					  'KT_benchmark_dataset/riiid/data_test/{}_{}_{}_{}.csv'.format(user_id, user_id, group_num, row_id, idx))
			new_records.to_csv('KT_benchmark_dataset/riiid/data_test/{}_{}_{}_{}.csv'.format(user_id, group_num, row_id, idx),
					   			mode='a', header=False, index=False)



		# last_user_id = user_id
		# total_questions += len(questions)
		# questions = []
		# groups = []

		# last_user_id = user_id if last_user_id != user_id else last_user_id


		

	print("Num of questions: {} \t Num of converted questions: {}".format(num_user, total_questions))








