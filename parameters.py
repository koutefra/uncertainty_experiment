import itertools
import uncertainty_toolbox as uct

EPSILON = 1e-15
RANDOM_STATE = 21
UNC_VAL_LOSS_FUNCTION = uct.nll_gaussian
N_FOLDS_UNBIASED_TRAIN_ERRORS = 10
N_FOLDS_DATA = 10
TRAIN_VAL_RATIO = 0.8

DATASETS = ['ames', 'communities', 'facebook', 'life', 'naval', 'power', 'star', 'yacht',
            'bike', 'concrete', 'meps', 'news', 'protein', 'superconductor', 'wave',
            'california', 'energy', 'kin8nm', 'msd', 'obesity', 'synthetic', 'wine']

IBUG_PARAMS_GRID = {'k': [3, 5, 7, 9, 11, 15, 31, 61, 91, 121, 151, 201, 301, 401, 501, 601, 701]}
KNN_PARAMS_GRID = {'k': [3, 5, 7, 9, 11, 15, 21, 27, 34, 42, 50, 61],
                   'w': [0, 1, 3, 5, 7, 9, 11, 15, 21, 27, 34, 42, 50, 61]}
KNN_NO_WEIGHT_PARAMS_GRID = {'k': [3, 5, 7, 9, 11, 15, 21, 27, 34, 42, 50, 61],
                             'w': [0]}

CB_PARAMS_GRID = {
	'od_type': ['Iter'], 'od_wait': [100], 'iterations': [3000], 'learning_rate': [0.1, 0.01],
	'depth': [4, 6, 8], 'l2_leaf_reg': [0.8, 3, 10]
}
CBU_PARAMS_GRID = CB_PARAMS_GRID

VARIANCE_CALIBRATION_PARAMS_GRID = {
	'op': ['add', 'mult'],
	'delta': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1e0, 1e1, 1e2, 1e3],
	'multiplier': [1.0, 2.5, 5.0]
}
