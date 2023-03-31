from abc import ABC, abstractmethod
from ibug import IBUGWrapper
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

import numpy as np
import parameters
import faiss
import fasttreeshap
import utils


class UncertaintyMethod(ABC):
	def __init__(self, model_params_grid, calibration_params_grid, loc_predictor_val, loc_predictor):
		self.best_params = None
		self.trained_model = None

		self.model_params_grid = model_params_grid
		self.calibration_params_grid = calibration_params_grid

		self.model_params_grid_list = utils.get_params_grid_combinations(model_params_grid)
		self.calibration_params_grid_list = utils.get_params_grid_combinations(calibration_params_grid)

		self.loc_predictor_val = loc_predictor_val
		self.loc_predictor = loc_predictor

	@abstractmethod
	def _tuning_init(self, X_train, y_train, X_val, y_val, loc_val):
		pass

	@abstractmethod
	def _tuning_get_scale(self, tuning_init_obj, X_train, y_train, X_val, y_val, loc_val, params):
		pass

	@abstractmethod
	def _tuning_parse_params(self, scale_predictor_val, params):
		pass

	@utils.time_it
	def tune(self, X_train, y_train, X_val, y_val, loc_val):
		best_loss = np.inf
		best_params = {}
		best_scale_val = None
		tuning_init_obj = self._tuning_init(X_train, y_train, X_val, y_val, loc_val)
		for params in self.model_params_grid_list:
			scale_val, scale_predictor_val = self._tuning_get_scale(tuning_init_obj, X_train, y_train, X_val, y_val, loc_val, params)
			scale_val = np.maximum(scale_val, parameters.EPSILON)
			loss = parameters.UNC_VAL_LOSS_FUNCTION(y_pred=loc_val, y_std=scale_val, y_true=y_val)
			if loss < best_loss:
				best_params = self._tuning_parse_params(scale_predictor_val, params)
				best_params['min_scale'] = self._get_min_scale(scale_val)
				best_scale_val = scale_val
				best_loss = loss
		best_delta = self.tune_delta(loc_val, best_scale_val, y_val)
		self.best_params = {**best_params, **best_delta}
		return self.best_params

	def _get_min_scale(self, scale_val):
		candidates_ids = np.where(scale_val > parameters.EPSILON)
		min_scale = np.min(scale_val[candidates_ids]) if len(candidates_ids) != 0 else parameters.EPSILON
		return min_scale

	@abstractmethod
	def train(self, X_train_val, y_train_val):
		pass

	@abstractmethod
	def predict(self, X, loc):
		pass

	def tune_delta(self, loc, scale, y):
		best_loss = np.inf
		best_delta = {}
		for params in self.calibration_params_grid_list:
			op = params['op']
			delta = params['delta']
			multiplier = params['multiplier']
			if op == 'mult' and delta == 0.0:
				continue
			temp_scale = scale + (delta * multiplier) if op == 'add' else scale * (delta * multiplier)
			loss = parameters.UNC_VAL_LOSS_FUNCTION(y_pred=loc, y_std=temp_scale, y_true=y)
			if loss < best_loss:
				best_delta = {'cal_delta': delta * multiplier, 'cal_op': op}
				best_loss = loss
		return best_delta

	@classmethod
	def calibrate(cls, func_predict):
		def do_calibration(self, uncalibrated_scale):
			cal_delta, cal_op = self.best_params['cal_delta'], self.best_params['cal_op']
			scale = uncalibrated_scale * cal_delta if cal_op == 'mult' else uncalibrated_scale + cal_delta
			return scale

		def apply_min_scale(self, scale):
			scale = np.maximum(scale, self.best_params['min_scale'])
			return scale

		def wrapper(self, X, loc):
			uncalibrated_scale = func_predict(self, X, loc)
			scale = apply_min_scale(self, uncalibrated_scale)
			scale = do_calibration(self, scale)
			return scale
		return wrapper


class CBU(UncertaintyMethod):
	def __init__(self, model_params_grid, calibration_params_grid, loc_predictor_val, loc_predictor):
		super().__init__(model_params_grid, calibration_params_grid, loc_predictor_val, loc_predictor)

	def _get_scale(self, model, X):
		predictions = model.predict(X)
		scale = predictions[:, 1]
		scale = np.sqrt(np.array(scale))
		return scale

	def _tuning_init(self, X_train, y_train, X_val, y_val, loc_val):
		return None

	def _tuning_get_scale(self, tuning_init_obj, X_train, y_train, X_val, y_val, loc_val, params):
		scale_predictor = CatBoostRegressor(loss_function='RMSEWithUncertainty', verbose=False,
		                                    depth=params['depth'], learning_rate=params['learning_rate'],
		                                    iterations=params['iterations'], l2_leaf_reg=params['l2_leaf_reg'],
		                                    od_type=params['od_type'], od_wait=params['od_wait'])
		scale_predictor.fit(X_train, y_train, eval_set=(X_val, y_val))
		return self._get_scale(scale_predictor, X_val), scale_predictor

	def _tuning_parse_params(self, scale_predictor_val, params):
		return {'iterations': scale_predictor_val.tree_count_, 'l2_leaf_reg': params['l2_leaf_reg'],
		        'learning_rate': params['learning_rate'], 'depth': params['depth']}

	@utils.time_it
	def train(self, X_train_val, y_train_val):
		scale_predictor = CatBoostRegressor(loss_function='RMSEWithUncertainty', l2_leaf_reg=self.best_params['l2_leaf_reg'],
											learning_rate=self.best_params['learning_rate'], depth=self.best_params['depth'],
											iterations=self.best_params['iterations'], verbose=False)
		scale_predictor.fit(X_train_val, y_train_val)
		self.trained_model = scale_predictor
		return self.trained_model

	@utils.time_it
	@UncertaintyMethod.calibrate
	def predict(self, X, loc):
		return self._get_scale(self.trained_model, X)


class IBUG(UncertaintyMethod):
	def __init__(self, model_params_grid, calibration_params_grid, loc_predictor_val, loc_predictor):
		super().__init__(model_params_grid, calibration_params_grid, loc_predictor_val, loc_predictor)

	def _tuning_init(self, X_train, y_train, X_val, y_val, loc_val):
		max_k = min(max(self.model_params_grid['k']), len(X_train))
		ibug = IBUGWrapper(random_state=parameters.RANDOM_STATE, k=max_k, variance_calibration=False,
						   min_scale=parameters.EPSILON, eps=parameters.EPSILON)
		ibug.fit(self.loc_predictor_val, X_train, y_train)
		# train indices are sorted from smallest to largest distance
		_, _, train_ids, _ = ibug.pred_dist(X_val, return_kneighbors=True)
		return {'train_ids': train_ids}

	def _tuning_get_scale(self, tuning_init_obj, X_train, y_train, X_val, y_val, loc_val, params):
		train_ids = tuning_init_obj['train_ids']
		k = params['k']
		train_ids_k = train_ids[:, -k:]
		train_values_k = np.array(y_train[train_ids_k])
		scale_val_k = np.std(train_values_k, axis=1)
		return scale_val_k, None

	def _tuning_parse_params(self, scale_predictor_val, params):
		return {'k': params['k']}

	@utils.time_it
	def train(self, X_train_val, y_train_val):
		k = self.best_params['k']
		ibug = IBUGWrapper(random_state=parameters.RANDOM_STATE, k=k, variance_calibration=False,
						   min_scale=parameters.EPSILON, eps=parameters.EPSILON)
		ibug.fit(self.loc_predictor, X_train_val, y_train_val)
		self.trained_model = ibug
		return self.trained_model

	@utils.time_it
	@UncertaintyMethod.calibrate
	def predict(self, X, loc):
		_, _, _, train_values = self.trained_model.pred_dist(X, return_kneighbors=True)
		scale = np.std(train_values, axis=1)
		return np.array(scale)


class kNN(UncertaintyMethod):
	def __init__(self, model_params_grid, calibration_params_grid, loc_predictor_val, loc_predictor):
		super().__init__(model_params_grid, calibration_params_grid, loc_predictor_val, loc_predictor)
		self.weights = None
		self.y_train = None
		self.stand_means = None
		self.stand_stds = None

	def _tuning_init(self, X_train, y_train, X_val, y_val, loc_val):
		max_k = max_k = min(max(self.model_params_grid['k']), len(X_train))
		self.weights = self._get_shap_feature_importance(self.loc_predictor_val, X_val)
		means, stds = np.mean(X_train, axis=0), np.std(X_train, axis=0)
		self.stand_means = means
		self.stand_stds = stds
		X_train_scaled = self._apply_weights(self._standardize(X_train, means, stds), self.weights)
		X_val_scaled = self._apply_weights(self._standardize(X_val, means, stds), self.weights)
		index = self._get_knn_index(X_train_scaled)
		# train indices are sorted from smallest to largest distance
		_, train_ids = self._find_knn(index, X_val_scaled, max_k)
		return {'train_ids': train_ids}

	def _tuning_get_scale(self, tuning_init_obj, X_train, y_train, X_val, y_val, loc_val, params):
		train_ids = tuning_init_obj['train_ids']
		k = params['k']
		w = params['w']
		train_ids_k = train_ids[:, :k]
		scale_val_k = np.array(self._get_std_from_neighbors(y_train, loc_val, train_ids_k, w))
		return scale_val_k, None

	def _tuning_parse_params(self, scale_predictor_val, params):
		return {'k': params['k'], 'w': params['w']}

	def _get_shap_feature_importance(self, loc_predictor, X):
		explainer = fasttreeshap.TreeExplainer(loc_predictor)
		shap_values = explainer.shap_values(X)
		shaps_mean = np.mean(np.abs(shap_values), axis=0)
		return shaps_mean

	def _get_std_from_neighbors(self, y_train, y_pred, train_ids, own_weight):
		scale = [np.std(np.concatenate((y_train[my_neigh_ids], np.full(own_weight, y_pred[my_i]))))
		         for my_i, my_neigh_ids in zip(range(len(train_ids)), train_ids)]
		return scale

	def _standardize(self, X, means, stds):
		X_standardized = (X - means) / np.maximum(stds, parameters.EPSILON)
		return X_standardized

	def _apply_weights(self, X, weights):
		X_weighted = X.copy()
		for i in range(X.shape[1]):
			X_weighted[:, i] = X[:, i] * weights[i]
		return X_weighted

	def _get_knn_index(self, X_train):
		X_train_C = X_train.astype('float32').copy('C')
		index = faiss.IndexFlatL2(X_train_C.shape[1])
		index.add(X_train_C)
		return index

	def _find_knn(self, index, X_test, k):
		X_test_C = X_test.astype('float32').copy('C')
		train_dists, train_ids = index.search(X_test_C, k)
		return train_dists, train_ids

	@utils.time_it
	def train(self, X_train, y_train):
		X_train_scaled = self._apply_weights(
			self._standardize(X_train, self.stand_means, self.stand_stds),
			self.weights
		)
		self.trained_model = self._get_knn_index(X_train_scaled)
		self.y_train = y_train
		return self.trained_model

	@utils.time_it
	@UncertaintyMethod.calibrate
	def predict(self, X, loc):
		k = self.best_params['k']
		own_weight = self.best_params['w']
		X_scaled = self._apply_weights(
			self._standardize(X, self.stand_means, self.stand_stds),
			self.weights
		)
		_, train_ids = self._find_knn(self.trained_model, X_scaled, k)
		scale = self._get_std_from_neighbors(self.y_train, loc, train_ids, own_weight)
		return np.array(scale)


class CB(UncertaintyMethod):
	def __init__(self, model_params_grid, calibration_params_grid, loc_predictor_val, loc_predictor):
		super().__init__(model_params_grid, calibration_params_grid, loc_predictor_val, loc_predictor)
		self.train_labels = None
		self.X_train_val_with_pred = None

	def _tuning_init(self, X_train, y_train, X_val, y_val, loc_val):
		X_train_val, y_train_val = np.vstack((X_train, X_val)), np.concatenate((y_train, y_val))
		loc_train_val_unbiased = self._get_unbiased_loc(parameters.N_FOLDS_UNBIASED_TRAIN_ERRORS,
		                                                 X_train_val, y_train_val, self.loc_predictor_val)
		unbiased_loc_train = loc_train_val_unbiased[:len(X_train)]
		unbiased_loc_val = loc_train_val_unbiased[-len(X_val):]

		# create labels for the uncertainty model
		train_val_abs_errors = abs(loc_train_val_unbiased - y_train_val)
		self.train_labels = train_val_abs_errors
		train_abs_errors = abs(unbiased_loc_train - y_train)
		val_abs_errors = abs(unbiased_loc_val - y_val)

		# create data for the uncertainty model
		X_train_val_with_pred = np.hstack((X_train_val, loc_train_val_unbiased.reshape(-1, 1)))
		self.X_train_val_with_pred = X_train_val_with_pred
		X_train_with_pred = np.hstack((X_train, unbiased_loc_train.reshape(-1, 1)))
		X_val_with_pred = np.hstack((X_val, unbiased_loc_val.reshape(-1, 1)))

		return {'unc_train_val_data': (X_train_val_with_pred, train_val_abs_errors),
		        'unc_train_data': (X_train_with_pred, train_abs_errors),
		        'unc_val_data': (X_val_with_pred, val_abs_errors)}

	def _tuning_get_scale(self, tuning_init_obj, X_train, y_train, X_val, y_val, loc_val, params):
		X_train_with_pred, train_abs_errors = tuning_init_obj['unc_train_data']
		X_val_with_pred, val_abs_errors = tuning_init_obj['unc_val_data']
		scale_predictor = CatBoostRegressor(depth=params['depth'], learning_rate=params['learning_rate'],
		                                    iterations=params['iterations'], l2_leaf_reg=params['l2_leaf_reg'],
		                                    od_type=params['od_type'], od_wait=params['od_wait'],
		                                    verbose=False)
		scale_predictor.fit(X_train_with_pred, train_abs_errors, eval_set=(X_val_with_pred, val_abs_errors))
		scale_val = np.array(scale_predictor.predict(X_val_with_pred))
		return scale_val, scale_predictor

	def _tuning_parse_params(self, scale_predictor_val, params):
		return {'iterations': scale_predictor_val.tree_count_, 'l2_leaf_reg': params['l2_leaf_reg'],
		        'learning_rate': params['learning_rate'], 'depth': params['depth']}

	def _get_unbiased_loc(self, n_folds, X, y, loc_predictor):
		kf = KFold(n_splits=n_folds, shuffle=True, random_state=parameters.RANDOM_STATE)
		unbiased_loc = np.zeros(len(y))
		params = loc_predictor.get_all_params()
		for i, (train_index, test_index) in enumerate(kf.split(X)):
			X_train_fold = X[train_index, :]
			y_train_fold = y[train_index]
			X_test_fold = X[test_index, :]
			model_fold = CatBoostRegressor(loss_function=params['loss_function'], iterations=params['iterations'],
										   learning_rate=params['learning_rate'], depth=params['depth'], verbose=False)
			model_fold.fit(X_train_fold, y_train_fold)
			unbiased_loc[test_index] = np.array(model_fold.predict(X_test_fold))
		return unbiased_loc

	@utils.time_it
	def train(self, X_train, y_train):
		scale_predictor = CatBoostRegressor(learning_rate=self.best_params['learning_rate'],
											l2_leaf_reg=self.best_params['l2_leaf_reg'], verbose=False,
											iterations=self.best_params['iterations'], depth=self.best_params['depth'])
		scale_predictor.fit(self.X_train_val_with_pred, self.train_labels)
		self.trained_model = scale_predictor
		return self.trained_model

	@utils.time_it
	@UncertaintyMethod.calibrate
	def predict(self, X, loc):
		X_with_pred = np.hstack((X, loc.reshape(-1, 1)))
		scale = self.trained_model.predict(X_with_pred)
		return np.array(scale)


class AcceleratedUncertaintyMethod(UncertaintyMethod):
	def __init__(self, model_params_grid, calibration_params_grid, loc_predictor_val, loc_predictor,
	             scale_predictor):
		super().__init__(model_params_grid, calibration_params_grid, loc_predictor_val, loc_predictor)
		self.base_scale_predictor = scale_predictor
		self.y_train_accelerated_method = None

	def _tuning_init(self, X_train, y_train, X_val, y_val, loc_val):
		fake_train_loc = np.zeros(len(X_train))
		fake_val_loc = np.zeros(len(X_val))
		ori_train_scale, _ = self.base_scale_predictor.predict(X_train, fake_train_loc)
		ori_val_scale, _ = self.base_scale_predictor.predict(X_val, fake_val_loc)
		self.y_train_accelerated_method = np.hstack((ori_train_scale, ori_val_scale))
		return {'ori_train_scale': ori_train_scale, 'ori_val_scale': ori_val_scale}

	def _tuning_get_scale(self, tuning_init_obj, X_train, y_train, X_val, y_val, loc_val, params):
		scale_predictor = CatBoostRegressor(depth=params['depth'], iterations=params['iterations'],
		                                    learning_rate=params['learning_rate'], verbose=False,
		                                    l2_leaf_reg=params['l2_leaf_reg'],
		                                    od_type=params['od_type'], od_wait=params['od_wait'])
		scale_predictor.fit(X_train, tuning_init_obj['ori_train_scale'], eval_set=(X_val, tuning_init_obj['ori_val_scale']))
		return np.array(scale_predictor.predict(X_val)), scale_predictor

	def _tuning_parse_params(self, scale_predictor_val, params):
		return {'iterations': scale_predictor_val.tree_count_, 'l2_leaf_reg': params['l2_leaf_reg'],
		        'learning_rate': params['learning_rate'], 'depth': params['depth']}

	@utils.time_it
	def train(self, X_train_val, y_train_val):
		scale_predictor = CatBoostRegressor(depth=self.best_params['depth'], iterations=self.best_params['iterations'],
		                                    learning_rate=self.best_params['learning_rate'], verbose=False,
		                                    l2_leaf_reg=self.best_params['l2_leaf_reg'])
		scale_predictor.fit(X_train_val, self.y_train_accelerated_method)
		self.trained_model = scale_predictor
		return self.trained_model

	@utils.time_it
	@UncertaintyMethod.calibrate
	def predict(self, X, loc):
		return np.array(self.trained_model.predict(X))