'''
In this module we're going to load, store and  prepare the dataset for machine learning experiments.
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def reader(filepath):
    '''
    In this function we load the dataset 

    Parameters:
    filepath:  File path where 

    Return:
    df: dataframe
    '''
    try:
        df = pd.read_csv(filepath)
    except:
        df_old = pd.read_csv("./Datasets/Sales_train.csv")
        df = create_new_dataset(df_old)
    return df


def get_dataset(filepath, process = True,cross_validation = True):
    '''
    This is the main function, here we process and split the  dataset 
    '''
    df = reader(filepath)
    if process:
        df = clean_dataset(df)
        df = new_feature(df) 
        split_mapping = dataset_split(df)
        return split_mapping
    else:
        dataY = df['item_cnt_day']
        dataX = df.drop(columns=['item_cnt_day'])
        threshold = round(len(dataX)*(1-0.3))
        X_train = dataX[0:threshold]
        y_train = dataY[0:threshold]
        X_test = dataX[threshold:]
        y_test = dataY[threshold:]
        split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
        return split_mapping

def new_feature(df):
    items = pd.read_csv("./Datasets/items.csv")
    items = items[['item_id','item_category_id']]
    df = pd.merge(df, items, on='item_id',  how='left')
    df = add_average_sale_per_shop(df)
    df = add_average_items_per_shop(df)
    df = add_average_items_per_category(df)
    df = add_total_items_per_category(df)
    return df




def clean_dataset(df):
    '''
    Cleaning our dataset
    '''
    df = delete_outliers(df)
    return df

def delete_outliers(df):
    '''
    Remove outliers
    '''
    df = df[df.item_cnt_day<1000]
    df = df[~((df.item_cnt_day>400) & (df.shop_id==24))]
    df = df[~((df.item_cnt_day>200) & (df.shop_id==47))]
    df = df[df.item_price<50000]
    try:
        df = df.drop(columns=["date"])
    except:
        pass

    return df

def dataset_split(df, look_back=3,size_test = 0.3):
    '''
    Organize the dataset by time windows using a look_back
    and then divide the data into train and test.

    Parameters:
    df:  the dataset
    look_back:  

    Return:
    df: dataframe
    '''
    y = df.item_cnt_day
    X = df.drop(columns=["item_cnt_day"])
    dataX, Y = [], []
    for i in range(len(y)-look_back):
        a = y[i:(i+look_back)]
        dataX.append(a)
        Y.append(y[i + look_back])
    dataY = np.array(Y)
    threshold = round(len(dataX)*(1-size_test))
    X_train = dataX[0:threshold]
    y_train = dataY[0:threshold]
    X_test = dataX[threshold:]
    y_test = dataY[threshold:]
    split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return split_mapping

def transform_data(dataset):
    '''
    In this function we take the dataset and generate
    a dictionary containing all sales per month for each product. 

    Parameters:
    dataset:  the dataset 

    Return:
    items: dictionary where 
            keys is the items(products) and values is a list with all sales per month 
    '''
    dataset = dataset.groupby(['date_block_num','item_id'], as_index=False).sum()
    items = {}
    for i in dataset.item_id.unique():
      if not (i in items):
        items[i] = [0]*34
    for i in dataset.index:
      item = dataset.item_id[i]
      #shop = d.shop_id[i]
      block = dataset.date_block_num[i]
      items[item][block] = dataset.item_cnt_day[i]
    return (items)

def items_new(dictionary_items):
    '''
    In this function we take the dictionary generated from "transform_data" function
    and classify each product according to its average sales during the last 34 months.

    Parameters:
    dictionary_items: dictionary where 
                    keys is the items(products) and values is a list with all sales per month 

    Return:
    Dataset: Dataset with two columns 'item' and 'new' 
    '''
    items = {'item_id':[],'new':[]}
    for i in dictionary_items:
        items['item_id'].append(i)
        if np.mean(dictionary_items[i])<=5:
            items['new'].append(1) # new item
        else:
            items['new'].append(0)
    return pd.DataFrame(items)

def create_new_dataset(df):
    '''
    This function will only be called once, 
    it allows to split and save the dataset
    taking into account whether it is considered as new products or not.

    Parameters:
    df: the original dataset

    Return:
    df_new: New dataset
    '''
    df = clean_dataset(df)
    items = transform_data(df)
    items_df = items_new(items)
    new_df = pd.merge(df,items_df[['item_id','new']],how='inner')
    filter = new_df['new']==1
    df_new_items = new_df[filter]
    filter = new_df['new']==0
    df_new = new_df[filter]
    df_new = df_new.drop(columns=["new"])
    df_new_items = df_new_items.drop(columns=["new"])
    print(df_new.shape, df_new_items.shape)
    df_new.to_csv('./NewDataset/New_dataset.csv')
    df_new_items.to_csv('./NewDataset/New_dataset_items.csv')
    return df_new

def add_average_sale_per_shop(df):
    '''
    Adds an average of the selling price of items per shop

    Parameters:
    df: the dataset

    Return:
    df_c: copy of the dataset but with a new column "average_price_per_shop" 
    '''
    df_c = df.copy()
    average_sale_price_per_shop = df.groupby("shop_id").mean().to_dict()["item_price"]
    global_average = 0
    df_c['average_price_per_shop'] = df['shop_id'].apply(
        lambda x: round(average_sale_price_per_shop[x],2) if x in average_sale_price_per_shop else global_average)
    return df_c

def add_average_items_per_shop(df):
    '''
    Adds an average of the sold items per shop

    Parameters:
    df: the dataset

    Return:
    df_c: copy of the dataset but with a new column "average_items_per_shop" 
    '''
    df_c = df.copy()
    average_items_per_shop = df.groupby("shop_id").mean().to_dict()["item_cnt_day"]
    global_average = 0
    df_c['average_items_per_shop'] = df['shop_id'].apply(
        lambda x: round(average_items_per_shop[x],2) if x in average_items_per_shop else global_average)
    return df_c

def add_average_items_per_category(df):
    '''
    Adds an average of the sold items per category

    Parameters:
    df: the dataset

    Return:
    df_c: copy of the dataset but with a new column "average_items_per_category" 
    '''
    df_c = df.copy()
    global_average = 0
    average_items_per_category = df_c.groupby("item_category_id").mean().to_dict()["item_cnt_day"]
    
    df_c['average_items_per_category'] = df_c['item_category_id'].apply(
        lambda x: round(average_items_per_category[x],2) if x in average_items_per_category else global_average)
    return df_c

def add_total_items_per_category(df):
    '''
    Adds the total amount of sold items per category

    Parameters:
    df: the dataset

    Return:
    df_c: copy of the dataset but with a new column "total_items_per_category" 
    '''
    df_c = df.copy()
    global_average = 0
    average_items_per_category = df_c.groupby("item_category_id").sum().to_dict()["item_cnt_day"]
    
    df_c['total_items_per_category'] = df_c['item_category_id'].apply(
        lambda x: round(average_items_per_category[x],2) if x in average_items_per_category else global_average)
    return df_c