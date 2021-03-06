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
import matplotlib.pyplot as plt

app = typer.Typer()


@app.command()
def train(config_file: str):
    estimator_config = _load_config(config_file, "estimator")
    data_config = _load_config(config_file, "data")
    X = _get_dataset(data_config, 'train')
    X1_train = X[0]
    X2_train = X[1]
    y_train = X[2] 
    if estimator_config[0]['additional_info'][0]['multi_input'] == True:
        inputs = [X1_train, X2_train]
    else:
        inputs = X2_train[:,:,0]

    output = y_train
    if estimator_config[0]['additional_info'][0]['transpose_y'] == True:
        output = np.ravel(y_train)
    estimator = model.build_estimator(estimator_config)
    estimator.fit(inputs, output)
    output_dir = _load_config(config_file, "export")["output_dir"] 
    _save_versioned_estimator(estimator, config_file, output_dir)

@app.command()
def test(config_file: str):
    model_config = _load_config(config_file, "model")
    model = joblib.load(model_config['filepath'])
    data_config = _load_config(config_file, "data")
    X = _get_dataset(data_config, 'test')
    x1_test = X[0]
    x2_test = X[1]
    y_test = X[2]
    if model_config[0]['multi_input'] == True:
        inputs = [x1_test, x2_test]
    else:
        inputs = x2_test[:,:,0]
    
    prediction = model.predict(inputs)
    error = metrics.eval(x1_test,y_test,prediction)
    content = {
        'opportunity cost': float(error[0]),
        'maintenance cost': float(error[1]),
    }
    output_dir = model_config['dir']
    output_file = _load_config(config_file, "metrics")['export']['filepath']
    with open(os.path.join(output_dir, output_file), "w") as f:
        yaml.dump(content, f)
    plt.plot(y_test)
    plt.plot(prediction)
    plt.show()

@app.command()
def predict(config_file: str, shop_id: int, category_id: int, predict_last: bool):
    data_config = _load_config(config_file, "data")['filepath']
    model_path = _load_config(config_file, "model")['filepath']

    # Find the corresponding shop/category info:
    df = pd.read_csv(data_config)
    row = df[df['id']==str((shop_id, category_id))]
    relevant_columns = ['shop_id', 'item_category_id']
    
    # Just for comparing purpose
    for i in (range(-3,0) if predict_last else range(-4,-1)):
        relevant_columns.append(row.columns[i])
    columns_to_drop = list(
        [column for column in row.columns if column not in relevant_columns])
    row = row.drop(columns=columns_to_drop)

    # Use the model:
    model = joblib.load(model_path)
    prediction = model.predict(row)
    print(prediction[0])
    return prediction[0]

def _get_dataset(data_config, key):
    file_path = data_config['filepath']
    return data.get_dataset(filepath=file_path)[key]

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
    X = _get_dataset(data_config, 'train')
    X1_train = X[0]
    X2_train = X[1]
    y_train = X[2] 
    if estimator_config[0]['additional_info'][0]['multi_input'] == True:
        inputs = [X1_train, X2_train]
    else:
        inputs = X2_train[:,:,0]

    output = y_train
    if estimator_config[0]['additional_info'][0]['transpose_y'] == True:
        output = np.ravel(y_train)
    estimator = model.build_estimator_search(estimator_config)   
    model_cv = GridSearchCV(estimator,param_grid,return_train_score=False)
    model_cv = model_cv.fit(inputs,output)
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
