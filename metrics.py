import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, make_scorer
import matplotlib.pyplot as plt
import app
from keras import backend

def get_metric_function(name: str, **params):
    mapping = get_metric_name_mapping()

    def fn(y_true, y_pred):
        return mapping[name](y_true, y_pred, **params)

    return fn


def get_metric_name_mapping():
    return {
        'root mean absolute error': rmse,
        'custom prediction error': eval}

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def calculate_cost_of_sales(cost,expense,quantity,initial_inventory=0):
    '''
    cost_per_item = (cost_per_item+expense_per_item)*(quanrity_per_item+initial_inventory_per_item)
    '''
    return float((cost+expense)*(quantity+initial_inventory))

def calculate_expense(expense,quantity,initial_inventory=0):
    '''
    experse_per_item = expense_per_item*(quantity_per_item + initial_invrntory_per_item)
    '''
    return float(expense*(quantity+initial_inventory))

def eval(x,y_true,y_pred):
    '''
    
    '''
    price = x[:,2]
    base_price = price*0.87
    salary_per_employee = 5964
    employees = 2
    rental = 8500
    transportation = 500
    total_expense = (salary_per_employee*employees)+rental+transportation
    expense_per_item = (total_expense/(sum(y_pred)))*60 # number_shops
    cost_sales_pred = []
    expense_pred = []
    for i,k in zip(base_price,y_pred):
        cost_sales_pred.append(calculate_cost_of_sales(i,expense_per_item,k))
        expense_pred.append(calculate_expense(expense_per_item,k))
    total_price = []
    for i, k in zip(price,y_pred):
        total_price.append(i*k)    
    gain_pred = []
    for i,k in zip(total_price,cost_sales_pred):
        gain_pred.append(i-k)
    error = []
    oportunidad =0
    mantenimiento = 0
    for i, k in zip(y_true,y_pred):
        error.append(i-k)
    for i in range(len(error)):
        if error[i]<0:
            oportunidad += float(gain_pred[i])
        else:
            mantenimiento += expense_pred[i]  
    print('Costo de oportunidad promedio: ',oportunidad/len(y_pred))
    print('Costo de mantenimiento promedio: ',mantenimiento/len(y_pred))
    return oportunidad,mantenimiento