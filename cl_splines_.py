dateTimesV = [ ]
for row in dtNew[ :, 0 ]:
    date = row.split(" ")[ 0 ]
    hhMMss = row.split(" ")[ 1 ]
    newDate = date + " " + ":".join(hhMMss.split(":")[ 0:2 ])
    dt = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
    dateTimesV.append(dt)

data = pd.read_csv('./export telegram.csv', sep='\t')
DdtNew = data.values[ 0:, : ]
dateTimesW = [ ]
ws = [ ]

for i in range(0, len(DdtNew[ :, 1 ])):
    if DdtNew[ i, 0 ] == 'T003':
        date = DdtNew[ i, 1 ].split(" ")[ 0 ]
        month = date.split('/')[ 1 ]
        day = date.split('/')[ 0 ]
        year = date.split('/')[ 2 ]

        hhMMss = DdtNew[ i, 1 ].split(" ")[ 1 ]
        newDate = year + "-" + month + "-" + day + " " + ":".join(hhMMss.split(":")[ 0:2 ])
        dt = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
        # if DdtNew[ i, 7 ] != 'No data' and DdtNew[ i, 8 ] != 'No data':
        dateTimesW.append(dt)

draftsVmapped = [ ]
ballastLadenFlagsVmapped = [ ]
for i in range(0, len(dateTimesV)):

    datesWs = [ ]

    date = str(dateTimesV[ i ]).split(" ")[ 0 ]
    month = date.split('-')[ 1 ]

    day = date.split('-')[ 2 ]

    year = date.split('-')[ 0 ]
    dt = dateTimesV[ i ]

    filteredDTWs = [ d for d in dateTimesW if month == str(d).split(' ')[ 0 ].split('-')[ 1 ] and day ==
                     str(d).split(' ')[ 0 ].split('-')[ 2 ] and year == str(d).split(' ')[ 0 ].split('-')[ 0 ] ]

    date2numFiltered = {}
    sorted = [ ]

    for k in range(0, len(filteredDTWs)):
        # dtDiff[filteredDTWs[k]]=abs((filteredDTWs[k]-dt).total_seconds() / 60.0)
        date = str(filteredDTWs[ k ]).split(" ")[ 0 ]
        month = date.split('-')[ 1 ]
        month = month[ 1 ] if month[ 0 ] == '0' else month
        day = date.split('-')[ 2 ]
        day = day[ 1 ] if day[ 0 ] == '0' else day
        year = date.split('-')[ 0 ]
        hhMMss = str(filteredDTWs[ k ]).split(" ")[ 1 ]
        # newDate = month + "/" + day + "/" + year + " " + ":".join(hhMMss.split(":")[ 0:2 ])
        newDate = day + "/" + month + "/" + year + " " + ":".join(hhMMss.split(":"))
        datesWs.append(newDate)

        date2numFilteredList = [ ]

        sorted.append(newDate)
        # date2numFiltered.append(matplotlib.dates.date2num(k))

        # dtFiltered[ filteredDTWs[ k ] ] = matplotlib.dates.date2num(k)
        # dt2num = matplotlib.dates.date2num(dt)
        # date2numFiltered.append(dt2num)
    date = str(dt).split(" ")[ 0 ]
    month = date.split('-')[ 1 ]
    month = month[ 1 ] if month[ 0 ] == '0' else month
    day = date.split('-')[ 2 ]
    day = day[ 1 ] if day[ 0 ] == '0' else day
    year = date.split('-')[ 0 ]
    hhMMss = str(dt).split(" ")[ 1 ]
    # newDate = month + "/" + day + "/" + year + " " + ":".join(hhMMss.split(":")[ 0:2 ])
    newDate = day + "/" + month + "/" + year + " " + ":".join(hhMMss.split(":"))
    sorted.append(newDate)
    sorted.sort()
    for i in range(0, len(sorted)):
        if sorted[ i ] == newDate:
            try:
                if i > 0:
                    if i == len(sorted) - 1:
                        previous = sorted[ i - 1 ]
                        blFlag = np.array(data.loc[ data[ 'telegram_date' ] == previous ].values)[ 0 ][ 49 ]
                        if blFlag == 'B':
                            ballastLadenFlagsVmapped.append('B')
                        else:
                            ballastLadenFlagsVmapped.append('L')
                    else:
                        try:
                            previous = sorted[ i - 1 ]
                            next = sorted[ i + 1 ]

                            # date1 = {k: v for k, v in dtFiltered.items() if v == previous}
                            # date2 = {k: v for k, v in dtFiltered.items() if v == next}
                            blFlag1 = np.array(data.loc[ data[ 'telegram_date' ] == previous ].values)[ 0 ][ 49 ]
                            blFlag2 = np.array(data.loc[ data[ 'telegram_date' ] == next ].values)[ 0 ][ 49 ]
                        except:
                            x = 0
                        if blFlag1 == 'L' and blFlag2 == 'L':
                            ballastLadenFlagsVmapped.append('L')
                        elif blFlag1 == 'L' and blFlag2 == 'B':
                            ballastLadenFlagsVmapped.append('L')
                        elif blFlag1 == 'B' and blFlag2 == 'L':
                            ballastLadenFlagsVmapped.append('B')
                        else:
                            ballastLadenFlagsVmapped.append('B')

                elif i == 0:
                    next = sorted[ i + 1 ]
                    blFlag = np.array(data.loc[ data[ 'telegram_date' ] == next ].values)[ 0 ][ 49 ]
                    if blFlag == 'B':
                        ballastLadenFlagsVmapped.append('B')
                    else:
                        ballastLadenFlagsVmapped.append('L')
                elif i == len(sorted) - 1:
                    previous = sorted[ i - 1 ]
                    blFlag = np.array(data.loc[ data[ 'telegram_date' ] == previous ].values)[ 0 ][ 49 ]
                    if blFlag == 'B':
                        ballastLadenFlagsVmapped.append('B')
                    else:
                        ballastLadenFlagsVmapped.append('L')
            except:
                ballastLadenFlagsVmapped.append('N')
                # firstMin = min(dtDiff.items(), key=lambda x: x[1])
    # del dtDiff[ firstMin[ 0 ] ]
    # secMin = min(dtDiff.items(), key=lambda x: x[ 1 ] )

    drafts = [ ]
    for date in datesWs:
        try:
            drftAFT = str(np.nan_to_num(np.array(data.loc[ data[ 'telegram_date' ] == date ].values)[ 0 ][ 27 ]))
            drftFORE = str(np.nan_to_num(np.array(data.loc[ data[ 'telegram_date' ] == date ].values)[ 0 ][ 28 ]))
            try:
                drftAFT = float(drftAFT.split(',')[ 0 ] + "." + drftAFT.split(',')[ 1 ]) if drftAFT.__contains__(
                    ',') else float(drftAFT)
                drftFORE = float(
                    drftFORE.split(',')[ 0 ] + "." + drftFORE.split(',')[ 1 ]) if drftFORE.__contains__(
                    ',') else float(drftFORE)
            except:
                x = 0
            drft = (drftAFT + drftFORE) / 2
            drafts.append(float(drft))

            y = np.array(drafts)
            x = matplotlib.dates.date2num(filteredDTWs)
            f = sp.Earth()
            f.fit(x.reshape(-1, 1), y.reshape(-1, 1))
            pred = f.predict(np.array(matplotlib.dates.date2num(dt)).reshape(-1, 1))
            draftsVmapped.append(pred)
        except:
            if np.nan_to_num(np.array(data.loc[ data[ 'telegram_date' ] == date ].values)[ 0 ][ 27 ]).__len__() == 0 or \
                    np.nan_to_num(
                        np.array(data.loc[ data[ 'telegram_date' ] == date ].values)[ 0 ][ 28 ]).__len__() == 0:
                draftsVmapped.append([ 0 ])

##########

windSpeedVmapped = [ ]
windDirsVmapped = [ ]

dateTimesV = [ ]
for row in dtNew[ :, 0 ]:
    date = row.split(" ")[ 0 ]
    hhMMss = row.split(" ")[ 1 ]
    newDate = date + " " + ":".join(hhMMss.split(":")[ 0:2 ])
    dt = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
    dateTimesV.append(dt)

data = pd.read_excel('./data/Persefone/Persefone TrackReport.xls')
WdtNew = data.values[ 0:, : ]
dateTimesW = [ ]
ws = [ ]

for i in range(0, len(WdtNew[ :, 1 ])):
    date = WdtNew[ i, 1 ].split(" ")[ 0 ]
    month = date.split('.')[ 1 ]
    day = date.split('.')[ 0 ]
    year = date.split('.')[ 2 ]

    hhMMss = WdtNew[ i, 1 ].split(" ")[ 1 ]
    newDate = year + "-" + month + "-" + day + " " + ":".join(hhMMss.split(":")[ 0:2 ])
    dt = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
    if WdtNew[ i, 7 ] != 'No data' and WdtNew[ i, 8 ] != 'No data':
        dateTimesW.append(dt)

for i in range(0, len(dateTimesV)):

    dtFiltered = {}
    datesWs = [ ]

    date = str(dateTimesV[ i ]).split(" ")[ 0 ]
    month = date.split('-')[ 1 ]
    day = date.split('-')[ 2 ]
    year = date.split('-')[ 0 ]
    dt = dateTimesV[ i ]

    filteredDTWs = [ d for d in dateTimesW if month == str(d).split(' ')[ 0 ].split('-')[ 1 ] and day ==
                     str(d).split(' ')[ 0 ].split('-')[ 2 ] and year == str(d).split(' ')[ 0 ].split('-')[ 0 ] ]
    date2numFiltered = {}

    for k in range(0, len(filteredDTWs)):
        date = str(filteredDTWs[ k ]).split(" ")[ 0 ]
        month = date.split('-')[ 1 ]
        day = date.split('-')[ 2 ]
        year = date.split('-')[ 0 ]
        hhMMss = str(filteredDTWs[ k ]).split(" ")[ 1 ]
        newDate = day + "." + month + "." + year + " " + ":".join(hhMMss.split(":")[ 0:2 ])

        datesWs.append(newDate)

    windSpeeds = [ ]
    windDirs = [ ]
    for date in datesWs:
        ws = np.array(data.loc[ data[ 'Date' ] == date + " Z" ].values)[ 0 ][ 7 ]
        wd = np.array(data.loc[ data[ 'Date' ] == date + " Z" ].values)[ 0 ][ 8 ]
        if ws != 'No data':
            windSpeeds.append(float(ws.split('m')[ 0 ]))
        if wd != 'No data':
            windDirs.append(float(wd[ :len(wd) - 1 ]))
    try:
        y = np.array(windSpeeds)
        x = matplotlib.dates.date2num(filteredDTWs)
        f = sp.Earth(max_degree=2)
        f.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        pred = f.predict(np.array(matplotlib.dates.date2num(dt)).reshape(-1, 1))
        windSpeedVmapped.append(pred)
    except:
        windSpeedVmapped.append(None)

    try:
        y = np.array(windDirs)
        x = matplotlib.dates.date2num(filteredDTWs)
        f = sp.Earth(max_degree=2)

        f.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        pred = f.predict(np.array(matplotlib.dates.date2num(dt)).reshape(-1, 1))
        # f=interp1d(list(x), list(y), kind='cubic')
        windDirsVmapped.append(pred)
    except:
        windDirsVmapped.append(None)
# pred = f(matplotlib.dates.date2num(dt))
# data = pd.read_excel('./Penelope TrackReport.xls')
# WdtNew = data.values[ 0:, : ]
# print(pred)
with open('./windSpeedMapped.csv', mode='w') as dataw:
    for k in range(0, len(windSpeedVmapped)):
        data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow([ windSpeedVmapped[ k ] if windSpeedVmapped[ k ] == None else windSpeedVmapped[ k ][ 0 ] ])
with open('./windDirMapped.csv', mode='w') as dataw:
    for k in range(0, len(windDirsVmapped)):
        data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow([ windDirsVmapped[ k ] if windDirsVmapped[ k ] == None else windDirsVmapped[ k ][ 0 ] ])

with open('./draftsMapped.csv', mode='w') as dataw:
    for k in range(0, len(draftsVmapped)):
        data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow([ draftsVmapped[ k ][ 0 ] ])

with open('./blFlagsMapped.csv', mode='w') as dataw:
    for k in range(0, len(ballastLadenFlagsVmapped)):
        data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow([ ballastLadenFlagsVmapped[ k ][ 0 ] ])


###################################################################################
from pyearth import Earth
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import  parameters as param_mars_dist
import random
from scipy.spatial import Delaunay

SEED = 1000

def importSeriesDataFromFile(path,n):
    data = pd.read_csv(
        path, nrows=20000)
    dt = data.values[ 0:, 3:23 ]
    X=dt[ 2000:12000, 1 ]
    Y = dt[ 2000:12000, 4 ]
    Xnew=[]
    prev=[[X[i],np.mean(X[i-n:i])] for i in range(n,len(X))]
    Xnew.append(prev)
    Xnew = np.array(Xnew).reshape(-1,2)
    return Xnew,Y[n:]

def clusteringV(dataX,nClusters):
    # Number of clusters
    #dataX=dataX[:,0:2]
    kmeans = KMeans(n_clusters=nClusters)
    # Fitting the input data
    kmeans = kmeans.fit(dataX)
    # Getting the cluster labels
    labels = kmeans.predict(dataX)
    # Centroid values
    centroids = kmeans.cluster_centers_
    #plt.scatter(centroids[:,0],c="red")
    plt.scatter(dataX[:,0],dataX[:,1], c=labels)
    plt.title("K-means with "+str(nClusters)+" clusters")
    plt.xlabel("V(t)")
    plt.ylabel("V(t,t-1,...,t-20)")
    plt.show()
    return labels , centroids , kmeans


# Given a set of instances, returns labels per cluster
def getClustersForTraining(dataX, Y, nClusters, labels):
    clustersX=[]
    clustersY=[]
    clustersXind = [ ]
    clustersYind = [ ]
    for n in range(0,nClusters):
        clustersX.append([])
        clustersY.append([ ])
        clustersXind.append([])
        clustersYind.append([ ])

    for i in range(0,len(labels)):
            clustersX[labels[i]].append(dataX[i,0])
            clustersY[labels[i]].append(Y[i])
            clustersXind[labels[i]].append(i)
            clustersYind[ labels[ i ] ].append(i)

    return clustersX,clustersY ,clustersXind,clustersYind

def tri_indices(simplices):
    #simplices is a numpy array defining the simplices of the triangularization
    #returns the lists of indices i, j, k

    return ([triplet[c] for triplet in simplices] for c in range(3))

def plotly_trisurf(x, y, z, simplices):
    #x, y, z are lists of coordinates of the triangle vertices
    #simplices are the simplices that define the triangularization;
    #simplices  is a numpy array of shape (no_triangles, 3)
    #insert here the  type check for input data

    points3D=np.vstack((x,y,z)).T
    tri_vertices=map(lambda index: points3D[index], simplices)# vertices of the surface triangles
    zmean=[np.mean(tri[:,2]) for tri in tri_vertices ]# mean values of z-coordinates of

                                                      #triangle vertices
    min_zmean=np.min(zmean)
    max_zmean=np.max(zmean)

    I,J,K=tri_indices(simplices)
    return  tri_vertices

def sign (p1,p2,p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def  PointInTriangle (pt,v1,v2, v3):

    b1 = sign(pt, v1, v2) < 0.0
    b2 = sign(pt, v2, v3) < 0.0
    b3 = sign(pt, v3, v1) < 0.0

    return ((b1 == b2) and (b2 == b3))


def showTriangle(x,y,z,error):
    #vertices = plotly_trisurf(x, y, z, tri.simplices)
    #polytops = list()

    k = np.vstack(np.array([[ x ], [ y ], [ z ]]))
    t = plt.Polygon(k, fill=False)
    plt.gca().add_patch(t)
    plt.xlim(np.min(k[0:,0]-10),np.max(k[0:,0])+10)
    plt.ylim(np.min(k[0:,1]-10),np.max(k[0:,1])+10)
    plt.title("Triangle with error bound " +str(round(error,2)))
    plt.xlabel("Velocity")
    plt.ylabel("RPM +- " +str(round(error,2)))
    plt.show()

def getTriangle(X,Y,usedV,errorBound):
        X=X[:,0]
        indinXnotinU = filter(lambda x: x not in usedV, range(0, len(X)))
        if usedV==[]:
            pointV1 = random.sample(X,1)
            index = random.sample(list([ num for num in range(0, len(X)) if X[ num ] > pointV1 ]), 1)
            pointV2 = X[index]
            pointRpm2 = Y[index]

        else:
            pointV1= random.sample(X[indinXnotinU],1)
            #p = [filter(lambda x: x not in usedV,range(0,len(X)))]
            x=X[indinXnotinU]
            y=Y[indinXnotinU]
            index=random.sample(list([ num for num in x if num > pointV1 ]), 1)
            pointV2 = index
            pointRpm2=index

        pointRpm1 = np.mean([Y[n] for n in range(0,len(Y)) if X[n]==pointV1])

        #points =np.vstack([[pointV1[0],pointRpm1],[pointV2[0],(pointRpm2+errorBound)[0]],[pointV2[0],(pointRpm2-errorBound)[0]]])
        #tri = Delaunay(points)
        showTriangle([pointV1[0],pointRpm1],[pointV2[0],(pointRpm1+errorBound)],[pointV2[0],(pointRpm1-errorBound)],errorBound)
        d=[PointInTriangle([ X[ i ], Y[ i ] ], [pointV1[0],pointRpm1],[pointV2[0],(pointRpm1+errorBound)],[pointV2[0],(pointRpm1-errorBound)])
        for i in indinXnotinU]

        indexInTr=[i for i in range(0,len(indinXnotinU)) if d[i]==True]
        return indexInTr


def trainingModel(clX,clY,trSplit,nParamsSearch,nsplits,Xtr,Ytr,clindx,clindY):
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
        indX = np.array(clindx[i])
        indy = np.array(clindY[i])
        offset = 0
        split = 0
        for train_index, test_index in trSplit.split(X):

            #print("TRAIN:", train_index, "TEST:", test_index)
            train_X, test_X = X[ train_index ], X[ test_index ]
            train_y, test_y = Y[ train_index ], Y[ test_index ]
            split += 1
            tr_ind = indX[train_index]
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
            if error>100:
                pointsTri = getTriangle(Xtr,Ytr,tr_ind,error)
                X=list(X)
                Y = list(Y)
                for x in  Xtr[0:,0][pointsTri]: X.append(x)
                for y in Ytr[ pointsTri ]: Y.append(y)
                X=np.array(X)
                Y=np.array(Y)
                for train_index, test_index in trSplit.split(X):
                    # print("TRAIN:", train_index, "TEST:", test_index)
                    train_X, test_X = X[ train_index ], X[ test_index ]
                    train_y, test_y = Y[ train_index ], Y[ test_index ]
                    model.fit(np.nan_to_num(train_X), np.nan_to_num(train_y))
                    y_hat = model.predict(np.array(test_X).reshape(-1, 1))
                    error1 = mean_squared_error(y_hat, test_y)

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
        path, nrows=40000)
    X = data.values[ 0:, 3:23 ]
    unseen = X[ 30000:30700, 1 ]
    rpm_unseen = X[ 30000:30700, 4 ]
    return  unseen,rpm_unseen


def predictLabelUnseen(unseenX,model):
    y=model.predict(unseenX)
    return y

def predictRPM(X,Y, model,models,models_basis,models_coef,n):
    lenx=0
    offset=10
    mseU=[]
    Xnew = [ ]
    prev = [ [ X[ i ], np.mean(X[i-n:i]) ] for i in range(n, len(X)) ]
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
        newModel.set_params(**newModel.get_params())

        preds=newModel.predict(chunkX[:,0])
        mse = mean_squared_error(preds,chunkY)
        mseU.append(mse)
        lenx=offset+lenx

    return np.mean(mseU)

def main():
    Xnew,Y=importSeriesDataFromFile('./kaklis.csv',20)
    labels,centroids , kmeans =clusteringV(Xnew,20)
    clTrX,clTrY,clindX,clindY=getClustersForTraining(Xnew, Y, 20, labels)

    # error_mse=[]
    # trains_x=[]

    tscv = TimeSeriesSplit()
    models,models_basis,models_coef,error=trainingModel(clTrX,clTrY,tscv,4,1,Xnew,Y,clindX,clindY)

    unseen , rpm_unseen = importUnseen('./kaklis.csv')
    mse=predictRPM(unseen,rpm_unseen,kmeans,models,models_basis,models_coef,20)

    print ("Mean squared error: %4.2f"%(mse))

if __name__ == "__main__":
    main()