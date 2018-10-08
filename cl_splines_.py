from pyearth import Earth
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import  parameters as param_mars_dist

SEED = 1000



# Given a set of instances, returns labels per cluster
def getClustersForTraining(dataX, Y, nClusters, labels):
    clustersX=[]
    clustersY=[]
    for n in range(0,nClusters):
        clustersX.append([])
        clustersY.append([ ])

    for i in range(0,len(labels)):
            clustersX[labels[i]].append(dataX[i,0])
            clustersY[labels[i]].append(Y[i])

    return clustersX,clustersY

def trainingModel(clX,clY,trSplit,nParamsSearch,nsplits):
    splits = nsplits
    trSplit.n_splits=splits
    def report(results, n_top=1):
        for i in range(1, n_top + 1):
            candidate = np.flatnonzero(results[ 'rank_test_score' ] == i)
            return results[ 'params' ][ candidate[ 0 ] ]


    knots=[]
    models=list()
    model_params=[]
    models_basis=[]
    models_coef=[]
    n_iter_search=nParamsSearch
    model = Earth()

    trains_x = []
    error_mse = []

    for i in range(0,len(clX)):
        X=np.array(clX[i])
        Y=np.array(clY[i])
        offset = 0
        split = 0
        for train_index, test_index in trSplit.split(X):

            #print("TRAIN:", train_index, "TEST:", test_index)
            train_X, test_X = X[ train_index ], X[ test_index ]
            train_y, test_y = Y[ train_index ], Y[ test_index ]
            split += 1
            #train_X = train_X[ offset: ]
            #train_y = train_y[ offset: ]

            #random_search3 = RandomizedSearchCV(model, param_distributions=param_mars_dist.param_mars_dist, n_iter=n_iter_search)
            #try:
            #    random_search3.fit(train_X, train_y)
            #    model.set_params(**report(random_search3.cv_results_))
            #except:
            #    print "Error"
                #print train_X, train_y

            trains_x.append(train_X)

            model.fit(np.nan_to_num(train_X), np.nan_to_num(train_y))
            y_hat = model.predict(np.array(test_X).reshape(-1, 1))
            predictions = list()
            predictions.append(y_hat)
            error = mean_squared_error(y_hat, test_y)


            error_mse.append(error)

            flag = False
            if error > 30:
                # print(model.summary())
                # pyplot.plot(test_y)
                # pyplot.plot(np.array(y_hat), color='red')
                # pyplot.show()
                flag = True
                offset = int(len(X) * split / splits)
                knots.append(split)
                print(offset)

        model_params.append(model.get_params())
        models_basis.append(model.basis_)
        models_coef.append(model.coef_)
        models.append(model)
    return models,models_basis,models_coef,error_mse

def importUnseen(path):
    data = pd.read_csv(
        path, nrows=20000)
    X = data.values[ 0:, 3:23 ]
    unseen = X[ 4200:4900, 1 ]
    rpm_unseen = X[ 4200:4900, 4 ]
    return  unseen,rpm_unseen


def predictLabelUnseen(unseenX,model):
    y=model.predict(unseenX)
    return y

def predictRPM(X,Y, model):
    lenx=0
    offset=10
    mseU=[]
    Xnew = [ ]
    prev = [ [ X[ i ], np.mean(X[i-20:i]) ] for i in range(20, len(X)) ]
    Xnew.append(prev)
    Xnew = np.array(Xnew).reshape(-1, 2)

    while lenx+offset <= len(Xnew):

        chunkX = Xnew[lenx:offset+lenx]
        chunkY = Y[lenx:offset+lenx]
        label = predictLabelUnseen(chunkX,model)

        c=max(set(label), key=list(label).count)
        newModel = models[c]
        newModel.coef_ = models_coef[c]
        newModel.basis_ = models_basis[c]
        #newModel.set_params(**newModel.get_params())

        preds=newModel.predict(chunkX[:,0])
        mse = mean_squared_error(preds,chunkY)
        mseU.append(mse)
        lenx=offset+lenx

    return np.mean(mseU)

def main():
    Xnew,Y=importSeriesDataFromFile('./kaklis_autoregr_rpm_speed.csv')
    labels,centroids , kmeans =clusteringV(Xnew,30)
    clTrX,clTrY=getClustersForTraining(Xnew, Y, 30, labels)

    # error_mse=[]
    # trains_x=[]

    tscv = TimeSeriesSplit()
    models,models_basis,models_coef,error=trainingModel(clTrX,clTrY,tscv,4,1)

    unseen , rpm_unseen = importUnseen('./kaklis.csv')
    mse=predictRPM(unseen,rpm_unseen)

    print ("Mean squared error: %4.2f"%(mse))

if __name__ == "__main__":
    main()