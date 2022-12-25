import time
from loguru import logger
from sklearn.model_selection import GridSearchCV


def model_selection(estimator, properties, cv, verbose, jobs, x_train, y_train):
    start_time = time.time()
    tuned_model = GridSearchCV(estimator, properties, scoring="accuracy", cv=cv,
                               return_train_score=True, verbose=verbose, n_jobs=jobs)
    tuned_model.fit(x_train, y_train)
    logger.info("--- %s seconds ---" % (time.time() - start_time))

    logger.info("Best Score: {:.3f}".format(tuned_model.best_score_))
    logger.info("Best Params: ", tuned_model.best_params_)
    return tuned_model
