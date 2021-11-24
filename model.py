import pandas as pd
import typing as t
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RANSACRegressor

EstimatorConfig = t.List[t.Dict[str, t.Any]]

def build_estimator(config: EstimatorConfig):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for step in config:
        name = step["name"]
        params = step["params"]
        estimator = estimator_mapping[name](**params)
        steps.append((name,estimator))
    model = Pipeline(steps)
    return model

def build_estimator_search(config: EstimatorConfig):
    estimator_mapping = get_estimator_mapping()
    name = config[0]["name"]
    params = config[0]["params"]
    estimator = estimator_mapping[name](**params)
    return estimator


def get_estimator_mapping(): 
    return {
        "logistic-regressor": LogisticRegression,
        "baseline": SalesPerCategory,
        "RANSACRegressor": RANSACRegressor 
    }

class SalesPerCategory():
    def fit(self, X, y):
        self.items = self.load_items()
        new_df = pd.read_csv('./Datasets/sales_train.csv')
        new_df = new_df.iloc[0:int(len(new_df)*0.7)]
        new_df['y'] = new_df['item_cnt_day']
        self.result = pd.merge(new_df,self.items[['item_id','item_category_id']],how='inner')
        self.aux = self.result.groupby(['date_block_num','item_category_id'],as_index=False).sum()
        self.result_mean= self.aux.groupby(['item_category_id'],as_index=False).mean()
        return self


    def predict(self, X):
        """Predicts the mode computed in the fit method."""
        X = pd.read_csv('./Datasets/sales_train.csv')
        X = X.iloc[int(len(X)*0.7):]
        X['y'] = X['item_cnt_day']
        r = pd.merge(X,self.items[['item_id','item_category_id']],how='inner')
        r_f = pd.merge(r,self.result_mean[['item_category_id']],how='inner')
        y_pred = r_f['y']  
        return y_pred
    
    def load_items(self):
        df = pd.read_csv("./Datasets/items.csv")
        return df

'''

'''