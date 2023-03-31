import time
import csv
import os
import numpy as np
import itertools
import fasttreeshap
import parameters


def time_it(func):
	def wrapper(*args, **kwargs):
		start_time = time.time()
		result = func(*args, **kwargs)
		end_time = time.time()
		time_elapsed = end_time - start_time
		return result, np.round(time_elapsed, 2)

	return wrapper


def csv_append_row(out_dir_path, csv_file_name, row):
	csv_file_path = os.path.join(out_dir_path, csv_file_name)

	if not os.path.exists(csv_file_path):
		with open(csv_file_path, 'w', newline='') as csv_file:
			writer = csv.writer(csv_file)
			writer.writerow(list(row.keys()))

	with open(csv_file_path, 'a', newline='') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(list(row.values()))


def round_all_metrics(metrics_dict, decimals):
	for metric_name in metrics_dict.keys():
		metrics_dict[metric_name] = np.round(metrics_dict[metric_name], decimals)
	return metrics_dict


def get_params_grid_combinations(params_grid):
	param_values = itertools.product(*params_grid.values())
	param_grids = [dict(zip(params_grid.keys(), p)) for p in param_values]
	return param_grids
