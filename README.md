COMING SOON!

# ARL-UPPER
# Unified Physicochemical Property Estimation Relationships of CCDC US Army Research Laboratory (UPPER-ARL)

UPPER-ARL is a cheminformatics toolkit for featurizing molecules and building quantitative structure-property relationships (QSPRs) for physicochemical properties.  This featurization is supplied as input to linear and nonlinear learning algorithms.

## Table of Contents
[Background and description](#bckgrd_dscrpts)

[Create environment](#create_env)

[Examples](#examples)

* [Featurization](#featurization)

* [Train and test](#train_and_test)

* [Hyperparameter search](#hypersearch)

* [Load model and test](#load_and_test)

[Running code](#run_code)

[Caveats](#caveats)

* [Fragmentation](#fragmentation)

* [Model applicability](#model_applicability)

[Credits](#credits)

[License](#license)

## <a name="bckgrd_dscrpts"/></a>Background and description

Phase transitions such as melting and boiling characterize a material's interaction with its environment. Further, properties such as transition temperature can influence whether a material is suitable for a given application. Examples of phase-driven applications include drug development, melt-casting explosives, or energy harvesting materials. However, a well-known difficulty of materials discovery is that chemical synthesis is costly and time consuming. Therefore, identifying potential candidates may proceed a long and arduous synthesis task. A priori knowledge of transition temperatures would limit the chemical space of candidate compounds, thus expediting discovery.

Predicting physicochemical properties has long been the work of linear regression-based QSPRs. These models have had success but are often limited by their inability to find nonlinear mappings from easy-to-derive descriptors. Recently, there has been a strong interest in applying the tools of data science to chemistry. Nonlinear machine learning (ML) algorithms have made their presence from the atomic scale to the continuum. This work builds on these modern-day predictive tools for transition temperatures. Specifically, this code featurizes compounds using group-constitutive and geometrical descriptors, previously shown to map to enthalpy and entropy--two thermodynamic quantities that drive phase transitions. The descriptors originate from the work of Yalkowsky and coworkers [J. Pharm. 103:2710-2723, 2014](https://onlinelibrary.wiley.com/doi/full/10.1002/jps.24033), known as UPPER (Unified Physicochemical Property Estimation Relationships). A notable advantage of UPPER is that descriptors are derived purely from a compound's SMILES string. Thus, besides relatively simple structural characteristics such as connectivity and hybridization, there are no high-level or numerically-intensive calculations necessary. This featurization can be supplied as input to learning algorithms. The code currently supports ridge regression and eXtreme Gradient Boosting (XGBoost). While it is generally a challenge to train models to limited experimental data, we find that UPPER's concise set of domain-specific descriptors, combined with nonlinear ML algorithms, provide an appealing framework for predicting transition enthalpies, entropies, and temperatures in a diverse set of compounds.

## <a name="create_env"/></a>Create environment
```bash
conda env create -f environment.yml
conda activate upper
```

## <a name="examples"/></a>Examples
Edit params.py for the desired tasks.

### <a name="featurization"/></a>Featurization
Set construct_fp to True.

```python
procedure_params = {'construct_fp': True,
                    'multiprocessing': False, # for cv
                    'model': 'xgb',
                    'train': False,
                    'predict': False,
                    'hyperparametrize': False,
                    'feature_reduction': False,
                    }
```

Set path to dataset (data_path), name of column with SMILES (smiles_name), and where to save featurization (fp_path).

```python
fp_params = {'data_path': '/path/to/data/example.csv',
             'smiles_name': 'SMILES',
             'labels': {'groups': ['X', 'Y', 'YY', 'YYY', 'Z', 'YZ', '', 'RG', 'AR', 'BR2', 'BR3', 'FU', 'BIP'],
                        'singles': ['2&6', '\u03A6', '\u03C3', '\u03B5\u00B3\u1d30', '\U0001D45E\u00B3\u1d30', '\U0001D45A'], # 2&6, phi-flexibility, sigma-symmetry, epsilon(3D)-eccentricity, q(3D)-asphericity, m-mass
                        },
             'fp_type': 'upper',
             'fp_path': '/path/to/fingerprint/example.csv',
             }
```

### <a name="train_and_test"/></a>Train and test
Set train and predict to True.

```python
procedure_params = {'construct_fp': False,
                    'multiprocessing': False, # for cv
                    'model': 'xgb',
                    'train': True,
                    'predict': True,
                    'hyperparametrize': False,
                    'feature_reduction': False,
                    }
```

Set path to featurization (fp_path), path to target property (target_path), and target column name (target_name).

```python
data_params = {'fp_path': '/path/to/fingerprint/example.csv',
               'target_path': '/path/to/dataset/example.csv',
               'target_name': 'Target Name',
               }
```

Set train/test split.

```python
split_params = {'split': {'train': 0.9, 'test': 0.1},
                'random_state': 123,
                'train_indices': np.array([]),
                'test_indices': np.array([]),
                }
```

For this example, set parameters of XGBoost (see [XGBoost's documentation](https://xgboost.readthedocs.io/en/latest/index.html)).

```python
xgb_train_params = {'learning_rate': 0.3,
                    'n_estimators': 100,
                    'objective': 'reg:squarederror',
                    }
```

Set paths where to save trained model and results.

```python
save_load_params = {'save_model': True,
                    'save_model_path': '/path/to/model/example.pkl,
                    'train_fp_path': '/path/to/fingerprint/of/trained/model/example.csv',
                    'npz_train_path': '/path/to/training/results/example.npz',
                    'npz_test_path': '/path/to/testing/results/example.npz',
                    }
```

### <a name="hypersearch"/></a>Hyperparameter search
This example executes a hyperparameter search, trains and tests.

```python
procedure_params = {'construct_fp': False,
                    'multiprocessing': False, # for cv
                    'model': 'xgb',
                    'train': True,
                    'predict': True,
                    'hyperparametrize': True,
                    'feature_reduction': False,
                    }
```

In addition to training/testing parameters, set ranges to search parameters.

```python
xgb_hyper_params = [
                    {'max_depth': np.arange(1, 11), # (0, inf), default = 3
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
```

### <a name="load_and_test"/></a>Load model and test
To load and use a trained model, set predict to True in procedure_params, edit data_params and save_load_params.

```python
save_load_params = {'load_model': True,
                    'load_model_path': '/path/to/model/example.pkl',
                    'train_fp_path': '/path/to/fingerprint/of/trained/model/example.csv',
                    'npz_train_path': '/path/to/training/results/example.npz',
                    'npz_test_path': '/path/to/testing/results/example.npz',
                    }
```

Here, load_model is True.  Paths to model (load_model_path) and training featurization (train_fp_path) are specified.

## <a name="run_code"/></a>Running code
```bash
python input.py
```

Note that params.py and input.py should be in the same directory.

## <a name="caveats"/></a>Caveats

### <a name="featurization"/></a>Fragmentation
During featurization, the code fragments the molecules, assigns them to groups, and indicates whether the number of fragments and assignments are equal.

```bash
2019-10-28 11:29:00,711 - root - INFO - indices of molecules with inconsistent number of frags:
[]
2019-10-28 11:29:00,712 - root - INFO - and their smiles:
[]
```

Molecules with fragments that do not belong to the current groups will be identified.  For such cases, new groups must be added to the code.

### <a name="model_applicability"/></a>Model applicability
A trained model is applicable if the groups of the test set are a subset of those in the training set.  The code will error otherwise.

## <a name="credits"/></a>Credits
For details of UPPER's descriptors, see [J. Pharm. 103:2710-2723, 2014](https://onlinelibrary.wiley.com/doi/full/10.1002/jps.24033).

## <a name="license"/></a>License
This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) license.
