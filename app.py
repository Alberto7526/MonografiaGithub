import typer
import model
from functools import lru_cache
import yaml
import typing as t
import pandas as pd
import data
from sklearn.base import BaseEstimator
from datetime import datetime, timezone
import os
import joblib
import shutil

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

if __name__ == "__main__":
    app()
    #python app.py .\config.yml 
