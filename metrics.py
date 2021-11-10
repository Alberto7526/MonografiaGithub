"""
In this module we store functions to measuer the performance of our model.

"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, make_scorer
import app

def get_metric_function(name: str, **params):
    mapping = get_metric_name_mapping()

    def fn(y_true, y_pred):
        return mapping[name](y_true, y_pred, **params)

    return fn


def get_metric_name_mapping():
    return {
        'mean absolute error': mean_absolute_error,
        'custom prediction error': custom_error}

def custom_error(y_true, y_pred,aditional_info = False):
    """
    A custom metric that is related to the business, the lower the better.
    
    """ 
    items = pd.read_csv("./Datasets/items.csv")
    dataset = pd.read_csv("./Datasets/sales_train.csv")
    dataset = pd.merge(dataset,items[['item_id','item_category_id']],how='inner')
    dataset = dataset.groupby(['item_category_id'], as_index=False).mean()
    dataset = dataset.drop(columns=('item_id'))
    dataset = dataset.rename(columns={'item_cnt_day':'item_cnt_month'})
    df = dataset[['item_category_id','item_price']]
    sales = {'item_category':[],'cnt_error':[],'total':[],'message':[]}
    for item_y_true,item_y_pred,data in zip(y_true.values,y_pred,np.array(df)):
        diff = float(item_y_true-item_y_pred)
        if diff >= 0:
            sales['message'].append('Si')
        else:
            sales['message'].append('No')
        sales['item_category'].append(data[0])
        sales['cnt_error'].append(abs(diff))
        sales['total'].append(abs(diff)*data[1])
    if(aditional_info):
        return sales
    else:
        return {'cnt_error':sum(sales['cnt_error']),'total_money':sum(sales['total'])}