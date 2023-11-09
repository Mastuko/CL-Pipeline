import mlflow
import pickle
from pathlib import Path


def mlflow_logmetrics(EXPERIMENT, METRICS, ITERATIONS, model_name, model, X_train, y_train, X_test, y_test, opt=None, params=None, persistency=""):
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="{}_{}".format(model_name, ITERATIONS)):
        if opt:
            bp = opt.run(X_train, y_train)
            regr = model(**bp)
        elif params:
            regr = model(**params)
        else:
            regr = model()
        if opt:
            print("best_params: {}".format(bp))
        regr.fit(X_train, y_train)
        pred = regr.predict(X_test)
        loss = []
        for m in METRICS:
            loss_item = m(pred, y_test).numpy()
            loss.append(loss_item)
            mlflow.log_metric(str(m), loss_item)
        if opt:
            for key in bp.keys():
                mlflow.log_param(key, bp[key])
        elif params:
            for key in params.keys():
                mlflow.log_param(key, params[key])
        else:
            pass
        if persistency != "":
            Path(f"{persistency}/{EXPERIMENT}").mkdir(parents=True, exist_ok=True)
            with open(f"{persistency}/{EXPERIMENT}/{model_name}.obj", "wb") as f:
                pickle.dump(regr, f)
        return regr, loss