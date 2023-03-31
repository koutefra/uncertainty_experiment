from abc import ABC, abstractmethod
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

import numpy as np
import parameters
from utils import time_it

class LocationMethod(ABC):
	def __init__(self):
		self.best_params = None
		self.trained_model = None

	@abstractmethod
	def tune(self, X_train, y_train, X_val, y_val):
		pass

	@abstractmethod
	def train(self, X_train, y_train):
		pass

	@abstractmethod
	def predict(self, X):
		pass


class CatBoost(LocationMethod):
	def __init__(self):
		super().__init__()

	@time_it
	def tune(self, X_train, y_train, X_val, y_val):
		best_loss = np.inf
		best_model = None
		param_grids = parameters.get_param_grids('cb')
		for param_grid in param_grids:
			model = CatBoostRegressor(depth=param_grid['depth'], learning_rate=param_grid['learning_rate'],
			                          iterations=param_grid['iterations'], l2_leaf_reg=param_grid['l2_leaf_reg'],
			                          od_type=param_grid['od_type'], od_wait=param_grid['od_wait'], verbose=False)
			model.fit(X_train, y_train, eval_set=(X_val, y_val))
			loc_val = model.predict(X_val)

			loss = np.sqrt(mean_squared_error(y_val, loc_val))

			if loss < best_loss:
				self.best_params = {'iterations': model.tree_count_, 'l2_leaf_reg': param_grid['l2_leaf_reg'],
				                    'learning_rate': param_grid['learning_rate'], 'depth': param_grid['depth']}
				best_model = model
				best_loss = loss

		return best_model, self.best_params

	@time_it
	def train(self, X_train, y_train):
		model = CatBoostRegressor(learning_rate=self.best_params['learning_rate'], depth=self.best_params['depth'],
		                          l2_leaf_reg=self.best_params['l2_leaf_reg'], iterations=self.best_params['iterations'],
		                          verbose=False)
		model.fit(X_train, y_train)
		self.trained_model = model
		return self.trained_model

	@time_it
	def predict(self, X):
		loc = self.trained_model.predict(X)
		return loc
