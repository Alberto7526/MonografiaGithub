import pandas as pd
import numpy as np
import data
import matplotlib.pyplot as plt


def transform_data_shop_category(dataset):
    items = pd.read_csv("./Datasets/items.csv")
    new_dataset = dataset.copy()
    new_dataset = pd.merge(new_dataset,items[['item_id','item_category_id']],how='inner')
    new_dataset = data.new_feature(new_dataset)
    grouped_dataset = new_dataset.groupby(['date_block_num','shop_id','item_category_id'], as_index=False)
    new_dataset = grouped_dataset.agg({
        'item_cnt_day': 'sum',
        'average_price_per_shop': 'mean',
        'average_items_per_shop': 'mean',
        'average_items_per_category': 'mean'
    })

    # new_dataset = new_dataset.drop(['item_id','item_price'],axis=1)
    new_dataset = new_dataset.rename(columns={'item_cnt_day':'item_cnt_month'})
    sales = {
        'id':[],
        'shop_id':[],
        'item_category_id':[],
        'average_price_per_shop': [],
        'average_items_per_shop': [],
        'average_items_per_category': []
    }

    for i in range(34):
        sales['month_'+str(i)]=[]
    for i in new_dataset.index:
        shop_id = new_dataset.shop_id[i]
        item_category_id = new_dataset.item_category_id[i]
        average_price_per_shop = new_dataset.average_price_per_shop[i]
        average_items_per_shop = new_dataset.average_items_per_shop[i]
        average_items_per_category = new_dataset.average_items_per_category[i]
        month = new_dataset.date_block_num[i]
        sales_per_month = new_dataset.item_cnt_month[i]
        a = (shop_id,item_category_id)
        if a in sales['id']:
            indice = sales['id'].index(a)
            sales['month_'+str(month)][indice] = sales_per_month
        else:
            sales['id'].append(a)
            sales['shop_id'].append(shop_id)
            sales['item_category_id'].append(item_category_id)
            sales['average_price_per_shop'].append(average_price_per_shop)
            sales['average_items_per_shop'].append(average_items_per_shop)
            sales['average_items_per_category'].append(average_items_per_category)
            for j in range(34):
                if j==month:
                    sales['month_'+str(j)].append(sales_per_month)
                else:
                    sales['month_'+str(j)].append(0)

    new_dataset = pd.DataFrame(sales)
    new_dataset = delete_outliers(new_dataset)
    try:
        new_dataset.to_csv('./NewDataset/New_dataset.csv')
    except:
        pass
    return new_dataset

def transform_data_shop(dataset):
    dataset = dataset.groupby(['date_block_num','shop_id'], as_index=False).sum()
    dataset = dataset.drop(columns=('item_id'))
    dataset = dataset.rename(columns={'item_cnt_day':'item_cnt_month'})
    sales = {'shop_id':[]}

    for i in range(34):
        sales['month_'+str(i)]=[]
    for i in dataset.index:
        shop_id = dataset.shop_id[i]
        month = dataset.date_block_num[i]
        sales_per_month = dataset.item_cnt_month[i]
        if shop_id in sales['shop_id']:
            indice = sales['shop_id'].index(shop_id)
            sales['month_'+str(month)][indice] = sales_per_month
        else:
            sales['shop_id'].append(shop_id)
            for j in range(34):
                if j==month:
                    sales['month_'+str(j)].append(sales_per_month)
                else:
                    sales['month_'+str(j)].append(0)

    dataset = pd.DataFrame(sales)

    return dataset


def transform_data_category(dataset):
    items = pd.read_csv("./Datasets/items.csv")
    dataset = pd.merge(dataset,items[['item_id','item_category_id']],how='inner')
    dataset = dataset.groupby(['date_block_num','item_category_id'], as_index=False).sum()
    dataset = dataset.drop(columns=('item_id'))
    dataset = dataset.rename(columns={'item_cnt_day':'item_cnt_month'})
    sales = {'item_category_id':[]}

    for i in range(34):
        sales['month_'+str(i)]=[]
    for i in dataset.index:
        item_category_id = dataset.item_category_id[i]
        month = dataset.date_block_num[i]
        sales_per_month = dataset.item_cnt_month[i]
        if item_category_id in sales['item_category_id']:
            indice = sales['item_category_id'].index(item_category_id)
            sales['month_'+str(month)][indice] = sales_per_month
        else:
            sales['item_category_id'].append(item_category_id)
            for j in range(34):
                if j==month:
                    sales['month_'+str(j)].append(sales_per_month)
                else:
                    sales['month_'+str(j)].append(0)

    dataset = pd.DataFrame(sales)
    return dataset

def delete_outliers(dataset_):
    dataset_ = dataset_.drop(['id'],axis=1)
    for i in range(34):
        dataset_.loc[(dataset_['month_'+str(i)] >=1500 ),'month_'+str(i)]=1500

    return dataset_


if __name__ == "__main__":
    df = pd.read_csv('./Datasets/sales_train.csv')
    transform_data_shop_category(df)
