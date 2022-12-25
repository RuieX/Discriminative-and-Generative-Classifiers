import time
import os
import joblib

from loguru import logger
from sklearn.model_selection import GridSearchCV


def model_selector(estimator, properties, scoring, cv, verbose, jobs, x_train, y_train):
    start_time = time.time()
    tuned_model = GridSearchCV(estimator, properties, scoring=scoring, cv=cv,
                               return_train_score=True, verbose=verbose, n_jobs=jobs)
    tuned_model.fit(x_train, y_train)
    logger.info("--- %s seconds ---" % (time.time() - start_time))

    logger.info("Best Score: {:.3f}".format(tuned_model.best_score_))
    logger.info("Best Params: ", tuned_model.best_params_)
    return tuned_model


def save_model(model, model_name):
    if not os.path.exists('../tuned_models'):
        os.mkdir('../tuned_models')
    joblib.dump(model, f'../tuned_models/{model_name}.pkl')


def load_model(model_name):
    return joblib.load(f"../tuned_models/{model_name}.pkl")
