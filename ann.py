from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
from keras.layers import Dropout
import math


#Defining ANN model
def model_ANN(inputs, outputs, layers, neurons, batch, opt, ker, epo, drop, act, x_t, y_t, x_v, y_v):

    model = Sequential()
    
    if layers == 1:  
        model.add(Dense(neurons, input_dim= inputs, kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(outputs, kernel_initializer=ker,activation='linear'))
        n = (neurons,)
        
       
    elif layers ==2:    
        model.add(Dense(neurons[0], input_dim= inputs, kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(neurons[1], kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(outputs, kernel_initializer=ker,activation='linear'))
        n = (neurons[0], neurons[1])
        
    
    elif layers ==3:    
        model.add(Dense(neurons[0], input_dim= inputs, kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(neurons[1], kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(neurons[2], kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(outputs, kernel_initializer=ker,activation='linear'))
        n = (neurons[0], neurons[1], neurons[2])
        
        
    elif layers ==4:    
        model.add(Dense(neurons[0], input_dim= inputs, kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(neurons[1], kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(neurons[2], kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(neurons[3], kernel_initializer=ker, activation=act))
        model.add(Dropout(drop))
        model.add(Dense(outputs, kernel_initializer=ker,activation='linear'))
        n = (neurons[0], neurons[1], neurons[2], neurons[3])
    
    print('No Inputs:', inputs, '  No Outputs: ', outputs)
    print('Layers: ', layers, '  Neurons: ', n )
    print('Batch: ', batch, '  Optimizer: ', opt , '  Initializer: ' , ker )
    print('Epochs: ', epo, '  Dropout: ', drop, ' Activation: ', act )
    
    model.compile(loss='mean_squared_error', optimizer= opt, metrics=['mae'])
    history = model.fit(x_t, y_t, epochs= epo, batch_size= batch, verbose=1, validation_data=(x_v, y_v))
     
    predictions = model.predict(x_t)
    predictions_v =  model.predict(x_v)

#ANN Performance metrics list
    MSE_scaled = mean_squared_error(y_t, predictions)
    RMSE = math.sqrt(MSE_scaled)
    MSE_scaled_val = mean_squared_error(y_v, predictions_v)
    RMSE_val = math.sqrt(MSE_scaled_val)
    R2 = r2_score(y_t, predictions)
    R2_v = r2_score(y_v, predictions_v)
    mae = history.history['mae'][-1]
    val_mae = history.history['val_mae'][-1]
    print("Results-- RMSE: %.2f RMSEv: %.2f R2: %.2f R2v: %.2f MAE: %.2f " % (RMSE, RMSE_val, R2, R2_v, mae))
    
    return RMSE, RMSE_val, mae, val_mae, R2, R2_v
    
  