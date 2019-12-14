
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from tensorflow import keras
import tensorflow as tf
import argparse
import sys
from scipy.spatial import distance

# In[ ]:


def main():
    # In[4]:


    # Read the file with the data. Test code is with .xlsx file. This needs to be changed in the production version with the read of an
    # xml file. Details TBD with Danaos Software Engineers.

    #df = pd.read_csv(args.Input_Filepath)
    df = pd.read_csv(args.Input_Filepath)

    # df2 =pd.read_csv(args.Training_Data)

    # ShipData_For_Prediction = 'C:/Users/manos/OneDrive/Desktop/Danaos/Deliverables/Debugging v1/ShipData_For_Prediction.csv'
    # df = pd.read_csv(ShipData_For_Prediction)


    # Load Parameters for Scalers (from the training script)

    scaler_x_filename = args.Scaler_X
    Scaler_x = joblib.load(scaler_x_filename)

    scaler_y_filename = args.Scaler_Y
    Scaler_y = joblib.load(scaler_y_filename)

    # Scaler_y= MinMaxScaler()
    # scaler_y_data_=np.load("C:/Users/manos/OneDrive/Desktop/Danaos/my_scaler_y.npy")
    # scaler_y_data_
    # Scaler_y.set_params(scaler_y_data_)

    # Target

    # df = pd.read_excel(args.Input_Filepath)

    # Print the head of the dataframe for confirmation that the data are structured correctly, and loaded correctly.
    # df.head()
    ##vessel_code=df['vessel_code']


    # Print the high level information of the dataframe, with statistics on the number of entries, data types etc.

    # df.info()


    # In[5]:


    # Define the index names of the vectors, and also the column names. Print for verification

    index_names = df.index
    columns_names = df.columns
    #print("Data Table has:", len(index_names), "data entries")
    #print("Data Table has columns:", columns_names, "as vectors")

    # In[7]:


    # Clarify for the user which are the predictor vectors. Present them in the form of a dataframe.

    # print ("Predictors Vectors Are: ")

    # Splitting the dataset. Separating the Predictor Vectors. These are the parameters that affect the prediction of the Target.
    Predictors = df.loc[:, columns_names[1:len(columns_names) - 2].values]
    actual = df.loc[:, columns_names[len(columns_names)-1:len(columns_names)  ].values]
    # Predictors.info()
    # Predictors.head(1)


    # In[8]:


    # Normalise the dataset. This is making all the values of each feature vector on a common scale.
    # There are different types of normalisation that can be applied to the data.
    # Examples are L1 - Lease Absolute Deviations - Sum of absolute values (on each row) = 1, it is insensitive to outliers
    # L2 : Least squares - sum of sqares (on each row) =1 ; takes outliners in consideration during training

    data = pd.read_csv('./MT_DELTA_MARIA_data_1.csv')
    seriesX, targetY, unseenFeaturesX, unseenFeaturesY = readLarosDataFromCsvNew(data)
    seriesX = seriesX[150000:151000]
    targetY = targetY[150000:151000]


    df_normalised = Scaler_x.fit_transform(Predictors)
    df_normalised = pd.DataFrame(df_normalised)
    x_scaled = df_normalised
    # x_scaled.head()
    #x_scaled = Scaler_x.fit_transform(seriesX)

    # In[9]:


    model = keras.models.load_model(args.Input_Model)

    # In[10]:


    pred = model.predict(x_scaled)
    pred = Scaler_y.inverse_transform(pred)

    # In[11]:

    errors=[]
    actual = actual.values
    pd_print = pd.DataFrame(pred)
    for i in range(0,len(pred)):
        #print (str(pred[i][0])+" ")
        errors.append(abs(pred[i] - actual[i]))
    print("MAE : " + str(np.mean(errors)))
    ########################
    ########################
    partitionsX=[]
    partitionsY=[]
    for cl in range(0,50):
        data = pd.read_csv('cluster_'+str(cl)+'_.csv')
        partitionsX.append(readClusteredLarosDataFromCsvNew(data))

    for cl in range(0,50):
        data = pd.read_csv('cluster_foc'+str(cl)+'_.csv')
        partitionsY.append(readClusteredLarosDataFromCsvNew(data))

    import matplotlib.pyplot as plt

    #data = pd.read_csv('./tdk_error_1.csv')
    #dtTdk = data.values[0:, :].astype(float)
    #data1= pd.read_csv('./m_error_1.csv')
    #dtMine = data1.values[0:, :].astype(float)
    #x = np.linspace(0, 10^3, 300)
    #for i in range(0,len(data)):
    #plt.plot(x,dtMine,'-',color='orange')
    #plt.plot(x, dtTdk,'-',color='black')
    #plt.ylabel('Absolute Error on 10^3 instances')
    #plt.title("Absolute Error on 10^3 instances")
    #plt.show()
    x=0

    #print("FOC Predictions: " + str(np.round(pred[0][0], 3)) + " and " + str(np.round(pred[1][0], 3)) + " with given vector of parameters: " + str(Predictors._get_values[0,:]))

    lErrors=[]
    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler()
    scalerY = StandardScaler()
    scalerX = scalerX.fit(np.concatenate(partitionsX))
    scalerY = scalerY.fit(np.concatenate(partitionsY).reshape(-1, 1))
    try:
        for i in range(36, len(pred)):
            #vector = seriesX[i]
            vector = Predictors.values[i]
            vector = vector.reshape(-1,5)
            #vector = np.array([vector[0],vector[2],vector[3],vector[4]]).reshape(-1, 4)
            #vector[:, 1] = abs(vector[:, 1])
            #Predictor =  np.array([Predictors.values[:,0],Predictors.values [:,2],Predictors.values [:,3],Predictors.values [:,4]]).reshape(-1, 4)
            #vctr = np.array([vector[0][1],vector[0][3]]).reshape(-1, 2)
            ind, fit = getBestPartitionForPoint(vector, partitionsX)
            #if fit < 10:
            currModeler = keras.models.load_model('estimatorCl_' + str(ind) + '.h5')
            #else:
            #currModeler = keras.models.load_model('estimatorCl_Gen_.h5')

            scaled = scalerX.fit_transform(vector)

            prediction = (scalerY.inverse_transform(currModeler.predict(scaled)))#  + scalerY.inverse_transform(currModeler1.predict(scaled))) / 2
            #prediction = (currModeler.predict(vector))  # + currModeler1.predict(vector)) / 2
            tf.keras.backend.clear_session()
            lErrors.append(abs(prediction - actual[i]))
            # x=unseenX[:,0].reshape(-1,unseenX.shape[0])
            # prediction =modeler._models[ 0 ].predict(unseenX.reshape(2,2860))
            # print np.mean(abs(prediction - unseenY))
        print("MAE MINE: " + str(np.mean(lErrors)))
    except:
        t=0
    import csv
    with open('./tdk_error_1.csv', mode='w') as data:
        for k in range(0, len(errors)):
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow([errors[k]])

    with open('./m_error_1.csv', mode='w') as data:
        for k in range(0, len(lErrors)):
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow([lErrors[k]])

    return pred

    pd_print.to_csv('Prediction_from_new_values.csv')

def getFitForEachPartitionForPoint(point, partitions):
        # For each model
        fits = []
        for m in range(0, len(partitions)):
            # If it is a better for the point
            dCurFit = getFitnessOfPoint(partitions, m, point)
            fits.append(dCurFit)

        return fits

def readLarosDataFromCsvNew(data):

        dataNew=data.drop(['DateTime','SpeedOvg'],axis=1)
        #'M/E FOC (kg/min)'

        #dataNew=dataNew[['Draft', 'SpeedOvg']]
        dtNew = dataNew.values[ 0:, : ].astype(float)

        dtNew = dtNew[~np.isnan(dtNew).any(axis=1)]

        trim = np.array(dtNew[:,1] - dtNew[:,2])
        seriesX = np.array(np.append(dtNew[:,0].reshape(-1,1),np.asmatrix([trim,dtNew[:,3],dtNew[:,4],dtNew[:,5]]).T,axis=1))
        #seriesX=dtNew[0:160000,0:6]
        seriesXnew = seriesX[:,0:6]

        UnseenSeriesX = seriesX[172000:173000]

        FOC = dtNew[:,7]

        unseenFOC = dtNew[172000:173000,7]


        return seriesXnew ,FOC , UnseenSeriesX , unseenFOC

def getFitnessOfPoint(partitions, cluster, point):
    return distance.euclidean(np.mean(partitions[cluster], axis=0) , point)
        #1 / (1 + np.linalg.norm(np.mean(partitions[cluster], axis=0) - point))



def getBestPartitionForPoint(point, partitions):
    # mBest = None
    mBest = None
    dBestFit = sys.maxsize
    # For each model
    for m in range(0, len(partitions)):
        # If it is a better for the point
        dCurFit = getFitnessOfPoint(partitions, m, point)
        if dCurFit < dBestFit:
            # Update the selected best model and corresponding fit
            dBestFit = dCurFit
            mBest = m

    if mBest == None:
        return 0, 0
    else:
        return mBest, dBestFit


def readClusteredLarosDataFromCsvNew(data):
    # Load file
    dtNew = data.values[0:, 0:5].astype(float)
    dtNew = dtNew[~np.isnan(dtNew).any(axis=1)]
    seriesX = dtNew
    UnseenSeriesX = dtNew
    return UnseenSeriesX



if __name__=="__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("Input_Model", help="Input Model")
    parser.add_argument("Scaler_X", help="Scaler X")
    parser.add_argument("Scaler_Y", help="Scaler Y")
    parser.add_argument("Input_Filepath", help="Input Filepath")
    args=parser.parse_args()

    #print(args.Input_Model)
    #print(args.Scaler_X)
    #print(args.Scaler_Y)
    #print(args.Input_Filepath)
    main()





# In[2]:


# Helpfull functions that are used, or will need to be used in the module execution

# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd


# Convert all missing values in the specified column to the median
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)


# Convert all missing values in the specified column to the default
def missing_default(df, name, default_value):
    df[name] = df[name].fillna(default_value)


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(
        target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    # Regression
    return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)

# Regression chart.
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    plt.figure(figsize=(30,20))
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')    
    plt.ylabel('output')
    plt.legend()
    plt.show()

# Remove all rows where the specified column is +/- sd standard deviations
def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean())
                          >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)


# Encode a column to a range between normalized_low and normalized_high.
def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1,
                         data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])

    df[name] = ((df[name] - data_low) / (data_high - data_low))         * (normalized_high - normalized_low) + normalized_low




