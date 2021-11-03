import pandas as pd
import numpy as np 
import data
import matplotlib.pyplot as plt
'''
try:
    df = pd.read_csv('.\\NewDataset\\New_dataset_with_new_feature.csv')
except:
    df = pd.read_csv('.\\NewDataset\\New_dataset.csv')
    df = data.clean_dataset(df)
    df = data.new_feacture(df) 

    df = df.rename(columns={'item_cnt_day':'item_cnt_month'})
    df = df.drop(columns=(['Unnamed: 0']))
    df.to_csv('./NewDataset/New_dataset_with_new_feature.csv')

y = df.item_cnt_month

plt.plot(y)
plt.show()
'''

def transform_data_shop_category(dataset):
    items = pd.read_csv("./Datasets/items.csv")
    dataset = pd.merge(dataset,items[['item_id','item_category_id']],how='inner')
    dataset = dataset.groupby(['date_block_num','shop_id','item_category_id'], as_index=False).sum()
    dataset = dataset.drop(columns=('item_id'))
    dataset = dataset.rename(columns={'item_cnt_day':'item_cnt_month'})
    print('.......', len(dataset))
    sales = {'id':[],'shop_id':[],'item_category_id':[]}

    for i in range(34):
        sales['month_'+str(i)]=[]
    for i in dataset.index: 
        shop_id = dataset.shop_id[i]
        item_category_id = dataset.item_category_id[i]
        month = dataset.date_block_num[i]
        sales_per_month = dataset.item_cnt_month[i]
        a = (shop_id,item_category_id)
        if a in sales['id']:
            indice = sales['id'].index(a)
            sales['month_'+str(month)][indice] = sales_per_month
        else:
            sales['id'].append(a)
            sales['shop_id'].append(shop_id)
            sales['item_category_id'].append(item_category_id)
            for j in range(34):
                if j==month: 
                    sales['month_'+str(j)].append(sales_per_month)
                else:
                    sales['month_'+str(j)].append(0)

    dataset = pd.DataFrame(sales)
    
    return dataset

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
    print('.......', len(dataset))
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


df = pd.read_csv('.\\Datasets\\sales_train.csv')
df = transform_data_category(df)
df.to_csv('./NewDataset/prueba_category.csv')

for i in range(5):
    plt.subplot(5,1,(i+1))
    plt.plot(df['month_'+str(i)])
plt.figure()
plt.plot(df['month_33'])
plt.show()