import numpy as np

# procedure
procedure_params: dict = {'construct_fp': True,
                          'multiprocessing': False, # cv for multiple models
                          'model': 'xgb',
                          'train': True,
                          'hyperparametrize': False, # and trains
                          'predict': True,
                          'feature_reduction': False,
}

# fingerprint
"""

Unicode commands for singles:

phi (flexibility) = '\u03A6'
sigma (symmetry) = '\u03C3'
epsilon^2D (2D eccentricity) = '\u03B5\u00B2\u1d30'
epsilon^3D (3D eccentricity) = '\u03B5\u00B3\u1d30'
q^3D (3D asphericity) = '\U0001D45E\u00B3\u1d30'
w (Wiener index) = '\U0001D464'
m (mass) = '\U0001D45A'
2&6 = number of halogens at 2,6 position of biphenyl (singles)

"""
fp_params: dict = {'data_path': '../../datasets/jain/jain.csv',
                   'smiles_name': 'SMILES',
                   'labels': {'groups': ['X', 'Y', 'YY', 'YYY', 'YYYY', 'YYYYY', 'Z', 'ZZ', 'YZ', 'YYZ', 'YYYZ', '', 'RG', 'AR', 'BR2', 'BR3', 'FU', 'BIP'],
                              'singles': ['\u03A6', '\u03C3', '\u03B5\u00B2\u1d30', '\U0001D45E\u00B3\u1d30', '\U0001D464', '\U0001D45A'],
                          },
                   'fp_type': 'upper',
                   'fp_path': 'upper_jain_3d_mass_w.csv',
                   'd_path': 'upper_jain_3d_mass_w.pkl',
}

# dataset
data_params: dict = {'fp_path': 'upper_jain_3d_mass_w.csv',
                     'target_path': '../../datasets/jain/jain.csv',
                     'target_name': 'Tm (K)',
}

# split train, test
split_params: dict = {'split': {'train': 0.9, 'test': 0.1},
                      'random_state': 123,
                      'tt_indices_path': '',
                      'train_indices': np.array([]),
                      'test_indices': np.array([]),
}

# model and results save/load
save_load_params: dict = {'save_model': True,
                          'load_model': False,
                          'save_model_path': 'model_jain_3d_mass_w',
                          'load_model_path': '',
                          'train_fp_path': 'upper_jain_3d_mass_w.csv',
                          'npz_train_path': 'train',
                          'npz_test_path': 'test',
}

# multiprocessing (e.g., cross validation)
multiprocess_params: dict = {'n_splits': 10,
                             'random_state': 123,
                             'shuffle': True,
}

### ridge regression model params ###

# train params
lr_train_params: dict = {'alpha': 0.5
}

# hyperparameter search
lr_hyper_params: list = [{'alpha': np.logspace(-6, 6, 13)
                      },
]

### xgb model params ###

# initial train params
xgb_train_params: dict = {'learning_rate': 0.3, # default = 0.3
                          'n_estimators': 100, # default = 100
                          'objective': 'reg:squarederror',
}

# hyperparameter search
xgb_hyper_params: list = [{'max_depth': np.arange(1, 11), # (0, inf), default = 3
                           'min_child_weight': [0, 1, 10, 100, 1000], # [0, inf), default = 1
                       },
                          {'gamma': [0, 1, 10, 100, 1000], # [0, inf), default = 0
                       },
                          {'subsample': np.linspace(0.1, 1, 10), # (0, 1], default = 1
                           'colsample_bytree': np.linspace(0.1, 1, 10), # (0, 1], default = 1
                       },
                          {'alpha': [0, 0.001, 0.01, 0.1, 1], # [0,inf), default = 0
                       },
]

# final train params if hyperparameter search
xgb_train_params_final: dict = {'learning_rate': 0.3, # default = 0.3
                                'n_estimators': 100, # default = 100
                                'objective': 'reg:squarederror',
}

# logger configuration
logging_config: dict = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
        'info': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'info.log',
            'mode': 'w',
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'info'],
            'level': 'INFO',
            'propagate': False
        },
        'my.packg': {
            'handlers': ['default'],
            'level': 'WARN',
            'propagate': False
        },
    }
}
