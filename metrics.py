import numpy as np
import pandas as pd
import uncertainty_toolbox as uct
import matplotlib.pyplot as plt

import parameters
import utils

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error


def get_loc_metrics(y, loc):
	metrics = {'RMSE': np.sqrt(mean_squared_error(y, loc)), 'R2': r2_score(y, loc), 'MAE': mean_absolute_error(y, loc),
	           'MAPE':  mean_absolute_percentage_error(y, loc)}
	return utils.round_all_metrics(metrics, 4)


def get_loc_metrics_names():
	return ['RMSE', 'R2', 'MAE', 'MAPE']


def get_scale_metrics(y, loc, scale):
	metrics = {'NLL': uct.nll_gaussian(y_pred=loc, y_std=scale, y_true=y),
	           'CRPS': uct.crps_gaussian(y_pred=loc, y_std=scale, y_true=y),
	           'CHECK': uct.check_score(y_pred=loc, y_std=scale, y_true=y),
	           'INTERVAL': uct.interval_score(y_pred=loc, y_std=scale, y_true=y),
	           'RMS_CAL': uct.root_mean_squared_calibration_error(y_pred=loc, y_std=scale, y_true=y),
	           'MA_CAL': uct.mean_absolute_calibration_error(y_pred=loc, y_std=scale, y_true=y),
	           'MISCAL_AREA': uct.miscalibration_area(y_pred=loc, y_std=scale, y_true=y),
	           'SHARPNESS': uct.sharpness(y_std=scale),
	           'UNC_AUC': get_unc_auc(loc, scale, y)[0],
	           'UNC_AUC_REL': get_unc_auc(loc, scale, y, rel_error=True)[0]}
	return utils.round_all_metrics(metrics, 3)


def get_scale_metrics_names():
	return ['NLL', 'CRPS', 'CHECK', 'INTERVAL', 'RMS_CAL', 'MA_CAL', 'MISCAL_AREA',
	        'SHARPNESS', 'UNC_AUC', 'UNC_AUC_REL']


def get_unc_auc(loc, scale, y, rel_error=False):
	abs_error = abs(loc - y)
	if rel_error:
		abs_error = abs_error / np.maximum(loc, parameters.EPSILON)

	df = pd.DataFrame({'scale': scale, 'abs_error': abs_error})
	arr = []
	for i in range(1, 100, 1):
		error_by_scale = df[df['scale'] <= df['scale'].quantile(i / 100)]['abs_error']
		error_by_scale_q_90 = error_by_scale.quantile(0.90)
		arr.append(error_by_scale_q_90)
	auc = sum(arr)
	return auc, arr


def get_unc_auc_plot(methods_results, rel_error=False, plot_graph=False):
	if plot_graph:
		plt.figure(figsize=(15, 10))

	auc_results = {}
	for method_name in methods_results.keys():
		loc = methods_results[method_name]['loc']
		scale = methods_results[method_name]['scale']
		y = methods_results[method_name]['y']
		auc, arr = get_unc_auc(loc, scale, y, rel_error)
		auc_results[method_name] = auc

		if plot_graph:
			plt.plot(arr, label=method_name)

	if plot_graph:
		plt.legend(loc="lower right")
		plt.xlabel('uncertainty quantile [%]')
		plt.ylabel(('rel. abs. ' if rel_error else 'abs. ') + 'error quantile 90%')
		plt.show()
		plot = plt.gcf()
	else:
		plot = None

	return auc_results, plot
