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
        'average_price_per_category': 'mean',
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
        'average_price_per_category': [],
        'average_items_per_shop': [],
        'average_items_per_category': []
    }

    for i in range(34):
        sales['month_'+str(i)]=[]
    for i in new_dataset.index:
        shop_id = new_dataset.shop_id[i]
        item_category_id = new_dataset.item_category_id[i]
        average_price_per_shop = new_dataset.average_price_per_shop[i]
        average_price_per_category = new_dataset.average_price_per_category[i]
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
            sales['average_price_per_category'].append(average_price_per_category)
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
        new_dataset.to_csv('./NewDataset/New_dataset_features.csv')
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
    '''
    fig, ax = plt.subplots(figsize=(15,7))
    flierprops = dict(marker='o', markerfacecolor='gray', markersize=6,
                    linestyle='none', markeredgecolor='black')
    boxprops=dict(color='gold')
    ax.boxplot(dataset_,boxprops=boxprops,flierprops=flierprops)
    ax.set_title('OUTLIERS',fontsize=20)
    ax.set_xlabel('SHOPS-CATEGORIES',fontsize=18)
    ax.set_ylabel('SALES',fontsize=18)
    plt.show()
    '''
    
    for i in range(34):
        dataset_.loc[(dataset_['month_'+str(i)] >=1500 ),'month_'+str(i)]=1500

    return dataset_

def split_rnn(df, look_back):
    x1_ = df[['shop_id','item_category_id','average_price_per_category','average_price_per_shop','average_items_per_shop','average_items_per_category']]
    x1 = pd.DataFrame()
    x2 = pd.DataFrame()
    y = pd.DataFrame()
    for i in range(32-look_back):
        x2_aux = pd.DataFrame()
        for k in range(look_back):
            x2_ = df[['month_'+str(i+k)]]
            x2_ = x2_.rename(columns={'month_'+str(i+k):'month_'+str(k)})
            x2_aux = pd.concat([x2_aux,x2_],axis=1)
        y_ = df[['month_'+str(i+look_back)]]
        y_ = y_.rename(columns={'month_'+str(i+look_back):'month'})
        x1 = pd.concat([x1,x1_],axis=0)
        x2 = pd.concat([x2,x2_aux],axis=0)
        y = pd.concat([y,y_],axis=0)
    x1_test = pd.DataFrame()
    x2_test = pd.DataFrame()
    y_test = pd.DataFrame()
    for i in range(32-look_back,34-look_back,1):
        print(i)
        x2_aux = pd.DataFrame()
        for k in range(look_back):
            x2_t = df[['month_'+str(i+k)]]
            x2_t = x2_.rename(columns={'month_'+str(i+k):'month_'+str(k)})
            x2_aux = pd.concat([x2_aux,x2_],axis=1)
        y_t = df[['month_'+str(i+look_back)]]
        y_t = y_t.rename(columns={'month_'+str(i+look_back):'month'})
        x1_test = pd.concat([x1_test,x1_],axis=0)
        x2_test = pd.concat([x2_test,x2_aux],axis=0)
        y_test = pd.concat([y_test,y_t],axis=0)
    train = (np.array(x1),np.array(x2).reshape(-1,look_back,1),np.array(y))
    test = (np.array(x1_test),np.array(x2_test).reshape(-1,look_back,1),np.array(y_test))
    return {'train':train,'test':test}




if __name__ == "__main__":
    df = pd.read_csv('./Datasets/sales_train.csv')
    transform_data_shop_category(df)
