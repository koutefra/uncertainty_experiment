import utils
import pandas as pd
import os
import numpy as np
import parameters
import random

from metrics import get_unc_auc_plot


def get_data(data_path, fold):
    data = np.load(data_path, allow_pickle=True)[()][fold]
    X_train, y_train = data['X_train'].astype(np.float64), data['y_train'].astype(np.float64)
    n_train = len(X_train)
    random.seed(parameters.RANDOM_STATE)
    train_indices = random.sample(range(n_train), int(parameters.TRAIN_VAL_RATIO * n_train))
    val_indices = np.setdiff1d(np.arange(n_train), train_indices)
    X_val, y_val = X_train[val_indices, :], y_train[val_indices]
    X_train, y_train = X_train[train_indices, :], y_train[train_indices]
    X_test, y_test = data['X_test'].astype(np.float64), data['y_test'].astype(np.float64)
    X_train_val, y_train_val = np.vstack((X_train, X_val)), np.concatenate((y_train, y_val))
    return X_train, y_train, X_train_val, y_train_val, X_val, y_val, X_test, y_test


def create_final_csv_files(out_data_dir_path, out_final_csv_dir_path, dataset_names, method_names, metric_names):
	for metric_name in metric_names:
		csv_col_names = [method_name + '_' + op_type for method_name in method_names for op_type in ['mean', 'se']]
		metric_csv_df = pd.DataFrame(columns=['dataset'] + csv_col_names).set_index('dataset')
		for dataset_name in dataset_names:
			row_values = []
			for method_name in method_names:
				method_dataset_df = pd.read_csv(os.path.join(out_data_dir_path, f'{method_name}_{dataset_name}.csv'))
				row_values.append(np.round(method_dataset_df[metric_name].mean(), 3))
				row_values.append(np.round(method_dataset_df[metric_name].sem(), 3))
			new_series = pd.Series(row_values, index=csv_col_names, name=dataset_name)
			metric_csv_df = metric_csv_df.append(new_series)
		metric_csv_df.to_csv(os.path.join(out_final_csv_dir_path, f'{metric_name}_all.csv'))


def create_final_auc_plots(out_pred_results_dir_path, out_auc_plots_dir_path, dataset_names, method_names):
	for dataset_name in dataset_names:
		methods_results = {}
		for method_name in method_names:
			csv_file_name = f'{method_name}_{dataset_name}.csv'
			prediction_results_df = pd.read_csv(os.path.join(out_pred_results_dir_path, csv_file_name))
			methods_results[method_name] = {}
			methods_results[method_name]['loc'] = np.array(prediction_results_df['loc'])
			methods_results[method_name]['scale'] = np.array(prediction_results_df['scale'])
			methods_results[method_name]['y'] = np.array(prediction_results_df['y_test'])
		_, plot = get_unc_auc_plot(methods_results, rel_error=False, plot_graph=True)
		_, plot_rel = get_unc_auc_plot(methods_results, rel_error=True, plot_graph=True)
		plot.savefig(os.path.join(out_auc_plots_dir_path, f'{dataset_name}_AUC.png'))
		plot_rel.savefig(os.path.join(out_auc_plots_dir_path, f'{dataset_name}_AUC_rel.png'))


def write_result(out_dir_path, dataset_name, method_name, fold, metrics, train_elapsed_sec,
                 predict_elapsed_sec, method_params):
	row = {'fold': fold, 'train_elapsed_sec': train_elapsed_sec, 'predict_elapsed_sec': predict_elapsed_sec}
	row.update(metrics)
	row.update(method_params)
	csv_file_name = f'{method_name}_{dataset_name}.csv'
	utils.csv_append_row(out_dir_path, csv_file_name, row)


def write_result_with_predictions(out_dir_path, dataset_name, method_name, fold, metrics, train_elapsed_sec,
				                  predict_elapsed_sec, method_params, distribution_results_df):
	write_result(out_dir_path, dataset_name, method_name, fold, metrics, train_elapsed_sec,
				 predict_elapsed_sec, method_params)

	distribution_results_file_name = f'distribution_results/{method_name}_{dataset_name}.csv'
	distribution_results_file_path = os.path.join(out_dir_path, distribution_results_file_name)
	if not os.path.exists(distribution_results_file_path):
		distribution_results_df.to_csv(distribution_results_file_path, header=True, index=False)
	else:
		with open(distribution_results_file_path, 'a') as f:
			distribution_results_df.to_csv(f, header=False, index=False)
