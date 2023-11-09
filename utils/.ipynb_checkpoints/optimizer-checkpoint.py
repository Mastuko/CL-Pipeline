import hyperopt
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials, space_eval
from hyperopt.pyll import scope
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
import time

    
    
class HyperOptimizer:
    def __init__(self, model, kwargs):
        '''
        Params:
        -- model: class of model, instantiate with params to a callable object -> class
        -- kwargs: parameters of model, organized as {("param", [typeofparam]):[a, b]->range of param}
        '''
        self.search_space = self.make_space(kwargs)
        self.algo = tpe.suggest
        self.model = model
        self.params = kwargs
        
    def make_space(self, kwargs):
        search_space = {}
        for k, v in kwargs.items():
            if k[1] == "float":
                search_space[k[0]] = hp.uniform(k[0], v[0], v[1])
            else:
                search_space[k[0]] = hp.choice(k[0], v)
        # print(search_space)
        return search_space
    
    def run(self, X, y, metric=mean_squared_error, greater_is_better=True, fold=5, max_evals=100):
        def objective(params):
            print(params)
            inst = self.model(**params)
            loss = cross_val_score(inst, X, y, cv=fold, scoring=make_scorer(metric, greater_is_better=greater_is_better)).mean()
            print("time: {} || Loss: {}".format(time.time(), loss))
            return {'loss': loss, 'status': STATUS_OK}
        best_params = fmin(fn=objective, space=self.search_space, algo=self.algo, max_evals=max_evals)
        return space_eval(self.search_space, best_params)
            