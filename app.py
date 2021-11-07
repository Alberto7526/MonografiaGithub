import typer
import model
import metrics
import division_dataset as div_d
from functools import lru_cache
import yaml
import typing as t
import pandas as pd
import data
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from datetime import datetime, timezone
import os
import joblib
import shutil
import numpy as np

app = typer.Typer()


@app.command()
def train(config_file: str):
    estimator_config = _load_config(config_file, "estimator")
    data_config = _load_config(config_file, "data")
    X = _get_dataset(data_config)
    X_train = X['train'][0]
    y_train = X['train'][1] 
    estimator = model.build_estimator(estimator_config)
    estimator.fit(X_train, y_train)
    output_dir = _load_config(config_file, "export")["output_dir"] 
    _save_versioned_estimator(estimator, config_file, output_dir)

@app.command()
def test(config_file: str):
    model_path = os.path.join("models", "2021-11-06 15-18", "model.joblib")
    og_dataset_path = os.path.join('Datasets', 'sales_train.csv')
    model = joblib.load(model_path)
    data_config = _load_config(config_file, "data")
    X = _get_dataset(data_config)
    X_test = X['test'][0]
    y_test = X['test'][1]
    prediction = model.predict(X_test)
    print('prediction', prediction)
    df = pd.read_csv(og_dataset_path)
    error = metrics.custom_error(y_test,prediction,df)
    content = {
        'cnt_error': float(error['cnt_error'][0]),
        'total_money': float(error['total_money'][0]),
    }
    print(content)
    output_dir = _load_config(config_file, "metrics")['export']['filepath']
    with open(output_dir, "w") as f:
        yaml.dump(content, f)

# @app.command()
# def predict(config_file: str, shop_id: int, category_id: int):
#     print('shop id:', shop_id, 'category id:', category_id)
#     data_config = _load_config(config_file, "data")
#     X = _get_dataset(data_config)
#     # ts = X.loc[(X['shop_id'] == shop_id) & X['favorite_color'].isin(array)]
#     print(X)

def _get_dataset(data_config):
    file_path = data_config['filepath']
    return data.get_dataset(filepath=file_path)

def _load_config(filepath: str, key: str):
    content = _load_yaml(filepath)
    config = content[key]
    return config


@lru_cache(None) 
def _load_yaml(filepath: str) -> t.Dict[str, t.Any]:
    with open(filepath, "r") as f:
        content = yaml.load(f)
    return content


def _save_yaml(config_file: str, filepath: str):
    estimator_config = _load_config(config_file, "estimator")
    data_config = _load_config(config_file, "data")
    content = {'Estimator': estimator_config,'data': data_config}
    with open(filepath, "w") as f:
        yaml.dump(content, f)

def _save_versioned_estimator(estimator: BaseEstimator, config_file: str, output_dir: str):
    version = datetime.now(timezone.utc).strftime("%Y-%m-%d %H-%M")
    model_dir = os.path.join(output_dir, version)
    os.makedirs(model_dir, exist_ok=True)
    try:
        joblib.dump(estimator, os.path.join(model_dir, "model.joblib"))
        _save_yaml(config_file, os.path.join(model_dir, "params.yml"))
    except Exception as e:
        typer.echo(f"Coudln't serialize model due to error {e}")
        shutil.rmtree(model_dir)
##############################################################################################3
@app.command()
def find_hyperparams(config_file: str):
    search_config = _load_config(config_file, "search")
    estimator_config = search_config[0]['estimator']
    grid_config = search_config[1]
    param_grid = grid_config['grid'][0]

    data_config = _load_config(config_file, "data")
    X = _get_dataset(data_config)
    x_train = X['train'][0]
    y_train = X['train'][1]

    estimator = model.build_estimator_search(estimator_config)   
    model_cv = GridSearchCV(estimator,param_grid,return_train_score=False)
    model_cv = model_cv.fit(x_train,y_train)
    result = pd.DataFrame(model_cv.cv_results_)  
    output_dir = _load_config(config_file, "export_best")["output_dir"] 
    _save_best_estimator(model_cv.best_estimator_,
                        model_cv.best_params_,
                        model_cv.best_score_, 
                        config_file, output_dir)
    

def _save_best_estimator(estimator: BaseEstimator, best_params: dict, best_score: float, config_file: str, output_dir: str):
    version = datetime.now(timezone.utc).strftime("%Y-%m-%d %H-%M")
    model_dir = os.path.join(output_dir, version)
    os.makedirs(model_dir, exist_ok=True)
    try:
        joblib.dump(estimator, os.path.join(model_dir, "model.joblib"))
        _save_best_params_yaml(config_file,best_params,best_score,os.path.join(model_dir, "params.yml"))
    except Exception as e:
        typer.echo(f"Coudln't serialize model due to error {e}")
        shutil.rmtree(model_dir)

def _save_best_params_yaml(config_file: str, best_params: dict, best_score: float, filepath: str):
    search_config = _load_config(config_file, "search")
    estimator_config = search_config[0]['estimator']
    grid_config = search_config[1]
    grid_params = grid_config['grid'][0]
    data_config = _load_config(config_file, "data")
    content = {'Estimator': estimator_config,
                'data': data_config,
                'grid': grid_params,
                'best_params': best_params,
                'best_score': float(best_score)}
    with open(filepath, "w") as f:
        yaml.dump(content, f)
###################################################################
@app.command()
def data_preparation(file_path: str):
    df = pd.read_csv(file_path)
    df = div_d.transform_data_shop_category(df)
    try:
        df.to_csv('./NewDataset/New_dataset.csv')
    except:
        pass




if __name__ == "__main__":
    app()
    #python app.py .\config.yml 
    #python app.py train .\config.yml
    #python app.py find-hyperparams .\config.yml
    #python app.py data-preparation .\Datasets\sales_train.csv
