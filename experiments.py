from catboost import CatBoostRegressor

import pandas as pd
import location_methods
import uncertainty_methods
import parameters
import os
import metrics
from data_handler import write_result, create_final_csv_files, write_result_with_predictions, get_data, \
	create_final_auc_plots
import numpy as np


loc_methods = [
	['CB', location_methods.CatBoost()]
]


def get_uncertainty_methods(loc_predictor_val, loc_predictor):
	cbu = uncertainty_methods.CBU(
		parameters.CBU_PARAMS_GRID,
		parameters.VARIANCE_CALIBRATION_PARAMS_GRID,
		loc_predictor_val,
		loc_predictor
	)
	ibug = uncertainty_methods.IBUG(
		parameters.IBUG_PARAMS_GRID,
		parameters.VARIANCE_CALIBRATION_PARAMS_GRID,
		loc_predictor_val,
		loc_predictor
	)
	accelerated_ibug = uncertainty_methods.AcceleratedUncertaintyMethod(
		parameters.CB_PARAMS_GRID,
		parameters.VARIANCE_CALIBRATION_PARAMS_GRID,
		loc_predictor_val,
		loc_predictor,
		ibug
	)
	knn = uncertainty_methods.kNN(
		parameters.KNN_PARAMS_GRID,
		parameters.VARIANCE_CALIBRATION_PARAMS_GRID,
		loc_predictor_val,
		loc_predictor
	)
	knn_no_weight = uncertainty_methods.kNN(
		parameters.KNN_NO_WEIGHT_PARAMS_GRID,
		parameters.VARIANCE_CALIBRATION_PARAMS_GRID,
		loc_predictor_val,
		loc_predictor
	)
	accelerated_knn = uncertainty_methods.AcceleratedUncertaintyMethod(
		parameters.CB_PARAMS_GRID,
		parameters.VARIANCE_CALIBRATION_PARAMS_GRID,
		loc_predictor_val,
		loc_predictor,
		knn_no_weight
	)
	cb = uncertainty_methods.CB(
		parameters.CB_PARAMS_GRID,
		parameters.VARIANCE_CALIBRATION_PARAMS_GRID,
		loc_predictor_val,
		loc_predictor
	)
	list_of_methods = [
		['CBU', cbu],
		['IBUG', ibug],
		['accelerated_IBUG', accelerated_ibug],
		['kNN', knn],
		['kNN_no_weight', knn_no_weight],
		['accelerated_kNN', accelerated_knn],
		['CB', cb]
	]
	return list_of_methods


unc_methods_names = ['CBU', 'IBUG', 'accelerated_IBUG', 'kNN', 'kNN_no_weight', 'accelerated_kNN', 'CB']


def run_loc_experiment(data_dir_path, out_dir_path):
	for fold in range(1, parameters.N_FOLDS_DATA + 1, 1):
		for dataset_name in parameters.DATASETS:
			# get data
			data_path = os.path.join(os.path.join(data_dir_path, dataset_name), 'data.npy')
			X_train, y_train, X_train_val, y_train_val, X_val, y_val, X_test, y_test = get_data(data_path, fold)

			for method_name, method in loc_methods:
				# do tuning and get the best model trained only on train data (without validation data)
				(model_val, best_params), tune_elapsed_sec = method.tune(X_train, y_train, X_val, y_val)
				val_model_path = os.path.join(out_dir_path,
				                                  f'models/loc_predictor_val_{method_name}_{dataset_name}_{fold}.dat')
				model_val.save_model(fname=val_model_path)

				# get model trained on train data + validation data
				trained_model, train_elapsed_sec = method.train(X_train_val, y_train_val)
				trained_model_path = os.path.join(out_dir_path,
				                                  f'models/loc_predictor_train_{method_name}_{dataset_name}_{fold}.dat')
				trained_model.save_model(fname=trained_model_path)

				# do predictions, compute metrics and save results
				loc, predict_elapsed_sec = method.predict(X_test)
				measured_metrics = metrics.get_loc_metrics(y_test, loc)
				write_result(os.path.join(out_dir_path, 'data'), dataset_name, method_name, fold, measured_metrics,
				             tune_elapsed_sec + train_elapsed_sec, predict_elapsed_sec, best_params)

	create_final_csv_files(os.path.join(out_dir_path, 'data'), os.path.join(out_dir_path, 'final_tables'),
	                       parameters.DATASETS, [x[0] for x in loc_methods],
	                       metrics.get_loc_metrics_names() + ['train_elapsed_sec', 'predict_elapsed_sec'])


def run_unc_experiment(data_dir_path, loc_predictors_dir_path, out_dir_path):
	for fold in range(1, parameters.N_FOLDS_DATA + 1, 1):
		print(f'Fold = {fold}')
		for dataset_name in parameters.DATASETS:
			print(f'Dataset name = {dataset_name}')
			# get the loc predictors
			val_loc_predictor = _load_loc_cb_model(loc_predictors_dir_path, 'CB', dataset_name, fold, True)
			loc_predictor = _load_loc_cb_model(loc_predictors_dir_path, 'CB', dataset_name, fold, False)
			unc_methods = get_uncertainty_methods(val_loc_predictor, loc_predictor)

			# get data
			data_path = os.path.join(os.path.join(data_dir_path, dataset_name), 'data.npy')
			X_train, y_train, X_train_val, y_train_val, X_val, y_val, X_test, y_test = get_data(data_path, fold)
			loc_val = np.array(val_loc_predictor.predict(X_val))
			loc = np.array(loc_predictor.predict(X_test))

			for method_name, method in unc_methods:
				print(f'Method name = {method_name}')
				data_file_path = os.path.join(out_dir_path, f'data/{method_name}_{dataset_name}.csv')
				if os.path.isfile(data_file_path) and fold <= len(pd.read_csv(data_file_path)):
					continue

				# tune & train & predict
				best_params, tune_elapsed_sec = method.tune(X_train, y_train, X_val, y_val, loc_val)
				_, train_elapsed_sec = method.train(X_train_val, y_train_val)
				scale, predict_elapsed_sec = method.predict(X_test, loc)
				measured_metrics = metrics.get_scale_metrics(y_test, loc, scale)
				distribution_results_df = pd.DataFrame({'y_test': y_test, 'loc': loc, 'scale': scale})
				write_result_with_predictions(os.path.join(out_dir_path, 'data'), dataset_name, method_name, fold,
				                              measured_metrics, tune_elapsed_sec + train_elapsed_sec,
				                              predict_elapsed_sec, best_params, distribution_results_df)
	create_final_csv_files(os.path.join(out_dir_path, 'data'), os.path.join(out_dir_path, 'final_tables'),
	                       parameters.DATASETS, unc_methods_names,
	                       metrics.get_scale_metrics_names() + ['train_elapsed_sec', 'predict_elapsed_sec'])
	create_final_auc_plots(os.path.join(out_dir_path, 'data/distribution_results'), os.path.join(out_dir_path, 'final_tables'),
	                       parameters.DATASETS, unc_methods_names)


def _load_loc_cb_model(loc_predictors_dir_path, loc_method_name, dataset_name, fold, is_validation_model):
	is_validation_model_str = 'val' if is_validation_model else 'train'
	file_name = f'loc_predictor_{is_validation_model_str}_{loc_method_name}_{dataset_name}_{fold}.dat'
	path = os.path.join(loc_predictors_dir_path, file_name)
	loc_predictor = CatBoostRegressor()
	loc_predictor_path = os.path.join(path)
	loc_predictor.load_model(loc_predictor_path)
	return loc_predictor
