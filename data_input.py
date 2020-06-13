import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gc

def data(training):
    dataframe = pd.read_csv("Data.csv", sep=',', header = None)
    dataset = dataframe.values
    
    #Locate input variables in the dataset
    X = dataset[:,0:64]
    #Locate output variables in the dataset
    Y = dataset[:,-3:]
    
        
    #split X, Y into a train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=training, random_state=42)
     
    del(dataframe, X, Y)
    gc.collect()
    
        # created scaler
    scaler = StandardScaler()
        # fit scaler on training dataset
    scaler.fit(Y_train)
        # transform training dataset
    Y_train = scaler.transform(Y_train)
        # transform test dataset
    Y_test = scaler.transform(Y_test)
    
    return X_train, X_test, Y_train, Y_test
 