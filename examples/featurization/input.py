from params import fp_params, data_params, procedure_params, logging_config
import sys
sys.path.append('../../')
from src import lc, multiprocessing_logging, log_inputs, construct_fingerprint, prepare_data, run_single_model, run_multiprocessing, single_model_mp
sys.path.remove('../../')

# configure logger
lc.dictConfig(logging_config)
multiprocessing_logging.install_mp_handler()

# log inputs
log_inputs()

if __name__ == '__main__':

    # construct fingerprint
    if procedure_params['construct_fp']:
        construct_fingerprint(**fp_params)

    # machine learning
    if procedure_params['train'] or procedure_params['predict'] or procedure_params['hyperparametrize']:

        # data
        X, y = prepare_data(**data_params)

        # single model
        if not procedure_params['multiprocessing']:
            run_single_model(X = X, y = y)

        # several models across processors
        if procedure_params['multiprocessing']:
            run_multiprocessing(single_model = single_model_mp, X = X, y = y)
