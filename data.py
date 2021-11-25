'''
In this module we're going to load, store and  prepare the dataset for machine learning experiments.
'''
from numpy.lib.shape_base import split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import division_dataset as div_data


def get_dataset(filepath,look_back=3):
    '''
    This is the main function, here we process and split the  dataset 
    '''
    df = pd.read_csv(filepath)
    df = df.drop(['Unnamed: 0'],axis=1)
    shops = [0,1,7,10,18,24,25,27,33,42,49,53,56,59]
    for i in shops:
        df = df.drop(df[df['shop_id']==i].index)
    categories = [0,1,9,12,29,35,42,44,53,54,56,57,60,64,65,66,69,70,71,]
    for i in categories:
        df = df.drop(df[df['item_category_id']==i].index)    
    print(df.head())
    
    split_mapping = split_dataset(df,look_back=look_back)
    return split_mapping



def split_dataset(df,look_back):    
    s = div_data.split_data(df,look_back)    
    #split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return s

def split_looking_back(x,y,look_back):
    dataX = x
    datay = pd.DataFrame()
    X = pd.DataFrame()
    for i in range(0,(len(y.columns)-look_back-3),look_back+1):
        for k in range(look_back+1):
            if k==look_back:
                a = y.loc[:,['month_'+str(i+k)]]
                a = a.rename(columns={'month_'+str(i+k):'y'})
                datay = pd.concat([datay, a], axis=0,)
            else:
                a = y.loc[:,['month_'+str(i+k)]]
                a = a.rename(columns={'month_'+str(i+k):'x_'+str(k)})
                dataX = pd.concat([dataX, a], axis=1,)
        X = pd.concat([X, dataX], axis=0,)
        dataX = x
    train = X,datay
    dataX_test = x
    for i in range((len(y.columns)-(look_back+1)),len(y.columns),1):
        if i==(len(y.columns)-1):
            a = y.loc[:,['month_'+str(i)]]
            a = a.rename(columns={'month_'+str(i):'y'})
            datay_test = a.copy()
        else:
            a = y.loc[:,['month_'+str(i)]]
            a = a.rename(columns={'month_'+str(i):'x_'+str(i)})
            dataX_test = pd.concat([dataX_test, a], axis=1,)
    test = dataX_test, datay_test 
    split_mapping = {"train": (train), "test": (test)}
    return split_mapping

def new_feature(df):
    # items = pd.read_csv("./Datasets/items.csv")
    # items = items[['item_id','item_category_id']]
    # df = pd.merge(df, items, on='item_id',  how='left')
    df = add_average_sale_per_shop(df)
    df = add_average_items_per_shop(df)
    df = add_average_items_per_category(df)
    df = add_average_price_per_category(df)
    # df = add_total_items_per_category(df)
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

def add_average_price_per_category(df):
    '''
    Adds an average of the selling price of items per category

    Parameters:
    df: the dataset

    Return:
    df_c: copy of the dataset but with a new column "average_price_per_category" 
    '''
    df_c = df.copy()
    average_sale_price_per_category = df.groupby("item_category_id").mean().to_dict()["item_price"]
    global_average = 0
    df_c['average_price_per_category'] = df['item_category_id'].apply(
        lambda x: round(average_sale_price_per_category[x],2) if x in average_sale_price_per_category else global_average)
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