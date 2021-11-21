import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import LSTM,Dense,Input,Concatenate,add
from tensorflow.keras.models import Model
from data import * 

'''
def split_looking_back(x,y,look_back):
    dataX1,dataX2, Y = [],[],[]
    for i in range(len(y)-look_back):
        a = y[i:(i+look_back)]
        dataX1.append(a)
        Y.append(y[i + look_back])
        dataX2.append(x[i])
    dataY = np.array(Y)
    return np.array(dataX1).reshape(-1,look_back,1),np.array(dataX2).reshape(-1,1),np.array(dataY)

def blockTimeSeriesSplit(X,y, margin=0, n_splits=10):
    n_samples = len(X)
    k_fold_size = n_samples // n_splits
    indices = np.arange(n_samples)
    for i in range(n_splits):
        start = i * k_fold_size
        stop = start + k_fold_size
        validation = int(0.8 * (stop - start)) + start
        yield indices[start: validation], indices[validation + margin: stop]


X = get_dataset(filepath='./NewDataset/New_dataset.csv')
x_train = X['train'][0]
y_train = X['train'][1] 

cv_plit = blockTimeSeriesSplit(x_train,y_train,n_splits=2)



estimator = RANSACRegressor(LinearRegression(),residual_threshold=20.0,loss='absolute_loss')
param_grid = {'max_trials':[30,40,50,70,80,100,120,150],
            'min_samples':[20,100,200,300]}
#model = GridSearchCV(estimator, param_grid, cv = cv_plit, return_train_score=False)
model = GridSearchCV(estimator, param_grid, return_train_score=False)
print(model)
model = model.fit(x_train,y_train)
result = pd.DataFrame(model.cv_results_)
print(result)
print(result.shape)
print(result.mean_test_score)
print(model.best_score_)
print(model.best_params_)
'''




x1,x2,y = split_looking_back(y_train,y_train,4) 

print('x1',x1.shape)
print('x2',x2.shape)
print('y',y.shape)
 
input1 = Input(shape=x1.shape[1:])
input2 = Input(shape=x2.shape[1])

layer1 = LSTM(10)(input1)
layer2 = Dense(5)(input2) 
layer3 = Concatenate()([layer1,layer2])
layer3 = Dense(10)(layer3)
output = Dense(1)(layer3)
model = Model(inputs=[input1,input2],outputs=[output])

print(model.summary())  

model.compile(optimizer='adam',
            loss='MeanSquaredError',
            metrics=['MeanSquaredError'])

model.fit([x1,x2],y)
'''

categoria, tienda,ventas mes de todos los meses (0-33)
33? ()


comandos:

encontrar hyperparametros

evaluar

train 

preparar datos y exportar el nuevo_Dataset

predecir --> ungresa categoria y tienda y predice las ventas  (carga el modelo x, 
                busca las features definidas para el modelo y predice)



'''
