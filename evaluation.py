import numpy as np
import dataModeling as dt
import tensorflow as tf
import sklearn.ensemble as skl
#import statsmodels.api
from statsmodels.formula.api import ols
import pandas as pd
#import scikit_posthocs as sp
from scipy import stats
from scipy.interpolate import BPoly as Bernstein
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import warnings
import math
from scipy import spatial
import pyearth as sp
from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt
import csv
import pyearth as sp
import sklearn.svm as svr

class Evaluation:
    def evaluate(self, train, unseen, model,genericModel):
        return 0.0

class MeanAbsoluteErrorEvaluation (Evaluation):
    '''
    Performs evaluation of the datam returning:
    errors: the list of errors over all instances
    meanError: the mean of the prediction error
    sdError: standard deviation of the error
    '''
    def evaluate(self, unseenX, unseenY, modeler,genericModel):
        lErrors = []
        for iCnt in range(np.shape(unseenX)[0]):
            pPoint = unseenX[iCnt].reshape(1, -1)#[0] # Convert to matrix
            trueVal = unseenY[iCnt]
            prediction = modeler.getBestModelForPoint(pPoint).predict(pPoint)

            lErrors.append(abs(prediction - trueVal))
        errors = np.asarray(lErrors)

        return errors, np.mean(errors), np.std(lErrors)

    def evaluateNN(self, unseenX, unseenY, modeler,output,xs):
        lErrors = []
        self.session = tf.Session()
        saver = tf.train.Saver()
       # tf.saved_model.loader.load(
       #     self.session,
       #     [ tf.saved_model.tag_constants.TRAINING ],
       #     './save')

        for iCnt in range(np.shape(unseenX)[0]):

            saver.restore(self.session, "./save/test_" + str(iCnt) + ".ckpt")
            model = modeler.getBestModelForPoint(pPoint)
            pPoint = unseenX[iCnt].reshape(1, -1)#[0] # Convert to matrix
            trueVal = unseenY[iCnt]
            prediction = self.session.run(output, feed_dict={xs: pPoint})
            lErrors.append(abs(prediction - trueVal))
            errors = np.asarray(lErrors)

        return errors, np.mean(errors), np.std(lErrors)

    def sign(self, p1, p2, p3):
        return (p1[ 0 ] - p3[ 0 ]) * (p2[ 1 ] - p3[ 1 ]) - (p2[ 0 ] - p3[ 0 ]) * (p1[ 1 ] - p3[ 1 ])

    def distance(self, a, b):
        return math.sqrt((a[ 0 ] - b[ 0 ]) ** 2 + (a[ 1 ] - b[ 1 ]) ** 2)

    def is_between(self, a, c, b):
        return self.distance(a, c) + self.distance(c, b) == self.distance(a, b)

    def PointInTriangle(self, pt, v1, v2, v3):

        b1 = self.sign(pt, v1, v2) < 0.0 #or self.is_between(v1, pt, v2)
        b2 = self.sign(pt, v2, v3) < 0.0 #or self.is_between(v2, pt, v3)
        b3 = self.sign(pt, v3, v1) < 0.0 #or self.is_between(v3, pt, v1)

        return ((b1 == b2) and (b2 == b3))

    def plotly_trisurf(self, x, y, z, simplices):
        # x, y, z are lists of coordinates of the triangle vertices
        # simplices are the simplices that define the triangularization;
        # simplices  is a numpy array of shape (no_triangles, 3)
        # insert here the  type check for input data

        points3D = np.vstack((x, y, z)).T
        tri_vertices = map(lambda index: points3D[ index ], simplices)  # vertices of the surface triangles
        zmean = [ np.mean(tri[ :, 2 ]) for tri in tri_vertices ]  # mean values of z-coordinates of

        # triangle vertices
        min_zmean = np.min(zmean)
        max_zmean = np.max(zmean)

        I, J, K = self.tri_indices(simplices)
        return tri_vertices

    def tri_indices(self, simplices):
        # simplices is a numpy array defining the simplices of the triangularization
        # returns the lists of indices i, j, k

        return ([ triplet[ c ] for triplet in simplices ] for c in range(3))

    def ccw(self,A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # Return true if line segments AB and CD intersect
    def intersect(self,A, B, C, D):
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def evaluateTriInterpolant(self, unseenX, unseenY,trainX,trainY,dataX,dataY, tri,modeler):
        lErrors = [ ]
        #triData=np.vstack((trainX[:,0],trainY))
        #triData=np.array(triData).reshape(-1,2)
        listX=[]
        if len(trainX) > 1:
            dataXnew = dataX
            dataYnew = dataY
        else:
            dataXnew = trainX[ 0 ]
            dataYnew = trainY[ 0 ]

        #########
        dataXnew3d=[]
        #for i in range(0,len(dataXnew)):
            #dataXnew3d.append([dataXnew[i][0],dataXnew[i][1]*dataYnew[i]])
        #dataXnew=dataXnew3d
        triNew = Delaunay(dataXnew, qhull_options='Q14' )

        ##############preprocessing scheme

        rpmVarPerTri = []
        meanRPMvar = np.var(dataYnew)
        meanRPM = np.mean(dataYnew)
        rpmDeviationPercentagefromMeanVarRPm = []
        count = 0
        delIndexes=[]
        delSimplices=[]
        flagedVertices=[]

        for i in range(0, len(dataXnew)):
            simplex = triNew.find_simplex(dataXnew[ i ])
            # for k in range(0, len(triNew.vertices)):
            k = triNew.vertices[ simplex ]

            rpm1 = dataYnew[ k ][ 0 ]
            rpm2 = dataYnew[ k ][ 1 ]
            rpm3 = dataYnew[ k ][ 2 ]

            V1 = dataXnew[ k ][ 0 ]
            V2 = dataXnew[ k ][ 1 ]
            V3 = dataXnew[ k ][ 2 ]

            varPerTri = np.var([ rpm1, rpm2, rpm3 ])
            meanPerTri = np.mean([ rpm1, rpm2, rpm3 ])
            rpmVarPerTri.append(varPerTri)
            rpmDeviationFromMeanVar = ((meanRPMvar - varPerTri) / meanRPMvar) * 100

            rpmDeviationFromMeanValue = ((meanRPM - meanPerTri) / meanRPM) * 100
            rpmDeviationPercentagefromMeanVarRPm.append(rpmDeviationFromMeanValue)
            if varPerTri > 5:
                delSimplices.append(int(simplex))
                flagedVertices.append(int(simplex))
                count += 1
                # triNew.vertices=np.delete(triNew.vertices,k,axis=0)
                from sklearn.cluster import DBSCAN

                _dataInit = np.array([ rpm1, rpm2, rpm3 ]).reshape(-1, 1)

                outlier_detection = DBSCAN(min_samples=1, eps=3)
                clustersInit = outlier_detection.fit_predict(_dataInit)
                Outclusters = 0 if list(clustersInit).count(0) == 1 else 1
                partitionsXInit = [ ]
                for curLbl in np.unique(clustersInit):
                    # Create a partition for X using records with corresponding label equal to the current
                    partitionsXInit.append(np.asarray(_dataInit[ clustersInit == curLbl ]))
                if len(partitionsXInit) > 1:
                    outlier_ind = list(clustersInit).index(Outclusters)
                    if outlier_ind == 0:
                        if (dataXnew[ i ] == V1).all() == True:
                            delIndexes.append(i)

                    elif outlier_ind == 1:
                        if (dataXnew[ i ] == V2).all() == True:
                            delIndexes.append(i)

                    else:
                        if (dataXnew[ i ] == V3).all() == True:
                            delIndexes.append(i)
        ##################################3
        dataXnew = np.delete(dataXnew,delIndexes,axis=0)
        #triNew.vertices=np.delete(triNew.vertices,delSimplices,axis=0)
        dataYnew = np.delete(dataYnew, delIndexes)
        percentageOfBigvar = (len(triNew.vertices) - float(count)) / float(len(triNew.vertices)) * 100
        #print("Mean variance of DT triangles: " +str(np.mean(np.array(rpmVarPerTri))))
        #Qj Qm Qv Q2 Q15 d G PD0:1.0 Qg

        #zx = range(0, len(dataXnew[ :, 0 ]))

        #vertices = self.plotly_trisurf(dataXnew[ :, 0 ], zx, dataXnew[ :, 1 ], triNew.simplices)

        #self.plotly_trisurf(dataX[ tri.vertices[ 0 ] ][ :, 0 ], zx[ 0:3 ], dataX[ tri.vertices[ 0 ] ][ :, 1 ],
         #tri.simplices)

        #plt.plot(unseenX[ 0:, 0 ], unseenX[ 0:, 1 ], 'o', markersize=8,color='blue')


        #for v in vertices:
            #k = np.array([ [ v[ 0 ][ 0 ], v[ 0 ][ 2 ] ], [ v[ 1 ][ 0 ], v[ 1 ][ 2 ] ], [ v[ 2 ][ 0 ], v[ 2 ][ 2 ] ] ])
            #t = plt.Polygon(k, fill=False,color='red',linewidth=3)
            #plt.gca().add_patch(t)
        #plt.show()
        x=1
        triNew = Delaunay(dataXnew, qhull_options='Q14')
        #hull = ConvexHull(dataXnew)
        #for simplex in hull.simplices:
            #plt.plot(dataXnew[ simplex, 0 ], dataXnew[ simplex, 1 ], 'k-', linewidth=3)
        ######################

        for i in range(0, len(triNew.vertices)):
            k = triNew.vertices[ i ]
            # if simplex == 8901:
            # x = 0
            rpm1 = dataYnew[ k ][ 0 ]
            rpm2 = dataYnew[ k ][ 1 ]
            rpm3 = dataYnew[ k ][ 2 ]

            V1 = dataXnew[ k ][ 0 ]
            V2 = dataXnew[ k ][ 1 ]
            V3 = dataXnew[ k ][ 2 ]

            if varPerTri > 10:
                # delSimplices.append(int(i))
                flagedVertices.append(int(i))

        ###build triangulations on clusters
        #triangulationsClusters = [ ]
        #for i in range(0, len(trainX)):
            #try:
                #triangulationsClusters.append(Delaunay(trainX[ i ]))
            #except Exception,e:
                #print(str(e))
                #x=0

        # for v in vertices:
        # k = np.array([ [ v[ 0 ][ 0 ], v[ 0 ][ 2 ] ], [ v[ 1 ][ 0 ], v[ 1 ][ 2 ] ], [ v[ 2 ][ 0 ], v[ 2 ][ 2 ] ] ])
        # t = plt.Polygon(k, fill=False,color='red',linewidth=3)
        # plt.gca().add_patch(t)
        # plt.show()

        x=9
        errorsTue=[]
        errorsFalse=[]
        countTrue = 0
        countFalse = 0
        ##########
        ########
        ###################
        for iu in range(0,len(unseenX)):

            flg = False
            trueVal = unseenY[ iu ]
            candidatePoint = unseenX[ iu ]

            preds=[]


            for k in range(0 ,len(triNew.vertices)):
                #simplicesIndexes
                #ki=triNew.find_simplex(ki)
                V1 = dataXnew[triNew.vertices[k]][0]
                V2 = dataXnew[triNew.vertices[k]][1]
                V3 = dataXnew[triNew.vertices[k]][2]

                #plt.plot(candidatePoint[ 0 ], candidatePoint[ 1 ], 'o', markersize=8)
                #plt.show()
                x=1
                b = triNew.transform[ k, :2 ].dot(candidatePoint - triNew.transform[ k, 2 ])
                W1 = b[ 0 ]
                W2 = b[ 1 ]
                W3 = 1 - np.sum(b)

                if W1>0 and W2>0 and W3>0:

                    flg  = True

                    rpm1 = dataYnew[ triNew.vertices[ k ] ][ 0 ]
                    rpm2 = dataYnew[ triNew.vertices[ k ] ][ 1 ]
                    rpm3 = dataYnew[ triNew.vertices[ k ] ][ 2 ]

                    flgR=False
                    flgV=False
                    ##efaptomena trigwna panw sto trigwno sto opoio anikei to Point estimate
                    tri1 =np.array(triNew.vertices[
                    [ i for i, x in enumerate(list(triNew.vertices[ :, 0 ] == triNew.vertices[ k, 0 ] )) if x ] ])
                    #tri1 = np.delete(tri1,triNew.vertices[k])

                    tri11=[]
                    k_set = set(triNew.vertices[k])

                    for tr in tri1:
                        a_set = set(tr)
                        if len(k_set.intersection(a_set)) == 2 and\
                                 (tr !=triNew.vertices[k]).any():
                            tri11.append(tr)

                    tri2 =np.array( triNew.vertices[
                        [ i for i, x in enumerate(list(triNew.vertices[ :, 1 ] == triNew.vertices[ k, 1 ])) if x ] ])
                    #tri2 = np.delete(tri2, triNew.vertices[ k ])
                    tri22 = [ ]
                    tri11=np.array(tri11)
                    for tr in tri2:
                        b_set = set(tr)
                        if len(k_set.intersection(b_set)) == 2 \
                                and \
                                (tr !=triNew.vertices[k]).any():
                            tri22.append(tr)

                    tri3 =np.array( triNew.vertices[
                        [ i for i, x in enumerate(list(triNew.vertices[ :, 2 ] == triNew.vertices[ k, 2 ])) if x ] ])
                    #tri3 = np.delete(tri3, triNew.vertices[ k ])
                    tri33 = [ ]
                    tri22=np.array(tri22)
                    for tr in tri3:
                        c_set = set(tr)
                        if len(k_set.intersection(c_set)) == 2 \
                                and \
                                (tr !=triNew.vertices[k]).any():
                            tri33.append(tr)

                    tri33 = np.array(tri33)
                    #####

                    ######
                    #attched_vertex = np.concatenate([tri11,tri22,tri33])

                    rpms=[]
                    vs=[]
                    if tri11!=[]:
                        for tr in tri11:
                            vi=[ x for x in tr if x not in triNew.vertices[k] ]
                            rpms.append( dataYnew[ vi ])
                            vs.append( dataXnew[ vi ])

                    if tri22!=[]:
                        for tr in tri22:
                            vi=[ x for x in tr if x not in triNew.vertices[k] ]
                            rpms.append( dataYnew[ vi ])
                            vs.append(dataXnew[ vi ])

                    if tri33!=[]:
                        for tr in tri33:
                            vi=[ x for x in tr if x not in triNew.vertices[k] ]
                            rpms.append( dataYnew[ vi ])
                            vs.append(dataXnew[ vi ])

                    if len(rpms) < 3:
                        flgR = True
                        rpms.append(np.array([0]))
                    try:
                        rpm12 = rpms[0]
                        rpm23=rpms[1]
                        rpm31 = rpms[2]
                    except :
                        #print(str(e))
                        x=0

                    if len(vs) < 3:
                        flgV = True
                        vs.append([np.array([0,0])])

                    v12 = vs[0][0]
                    v23 = vs[1][0]
                    v31 =  vs[2][0]

                    x=0

                    ##barycenters (W12,W23,W13) of neighboring triangles and solutions of linear 3x3 sustem in order to find gammas.
                    try:

                        ###################################################
                        eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                                         [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

                        eq2 = np.array([ v12[ 0 ] - V3[ 0 ], v12[ 1 ] - V3[ 1 ] ])
                        solutions = np.linalg.solve(eq1, eq2)

                        W12_1 = solutions[ 0 ]
                        W12_2 = solutions[ 1 ]
                        W12_3 = 1 - solutions[ 0 ] - solutions[ 1 ]

                        cond_numer1 = np.linalg.cond(eq1)
                        ###########################

                        eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                                         [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

                        eq2 = np.array([ v23[ 0 ] - V3[ 0 ], v23[ 1 ] - V3[ 1 ] ])
                        solutions = np.linalg.solve(eq1, eq2)

                        W23_1 = solutions[ 0 ]
                        W23_2 = solutions[ 1 ]
                        W23_3 = 1 - solutions[ 0 ] - solutions[ 1 ]
                        cond_numer2 = np.linalg.cond(eq1)
                        ####################################################

                        eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                                         [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

                        eq2 = np.array([ v31[ 0 ] - V3[ 0 ], v31[ 1 ] - V3[ 1 ] ])
                        solutions = np.linalg.solve(eq1, eq2)

                        W31_1 = solutions[ 0 ]
                        W31_2 = solutions[ 1 ]
                        W31_3 = 1 - solutions[ 0 ] - solutions[ 1 ]
                        cond_numer3 = np.linalg.cond(eq1)
                    ####################################
                    ##############################
                    except :
                        #print(str(e))
                        x=0

                    #############################
                    ##end of barycenters
                    flgExc=False
                    try:

                        eq1=0
                        eq2=0
                        initrpm1 = rpm1
                        initrpm2 = rpm2
                        initrpm3 = rpm3
                        neighboringVertices1 = [ ]
                        neighboringVertices2 = [ ]

                        for u in range(0, 2):
                            try:
                                neighboringVertices1.append(triNew.vertex_neighbor_vertices[ 1 ][
                                                            triNew.vertex_neighbor_vertices[ 0 ][
                                                                triNew.vertices[ k ][ u ] ]:
                                                            triNew.vertex_neighbor_vertices[ 0 ][
                                                                triNew.vertices[ k ][ u ] + 1 ] ])
                            except:
                                break
                        neighboringTri = triNew.vertices[
                            triNew.find_simplex(dataXnew[np.concatenate(np.array(neighboringVertices1)) ]) ]
                        if (W1 > 0 and W2 > 0 and W3 > 0):

                            nRpms = [ ]
                            nGammas = [ ]
                            rpm31 = rpm31 if rpm31 != 0 else (rpm1 + rpm2 + rpm3 + rpm23 + rpm12) / 5

                            B1 = (math.pow(W12_1, 2) * rpm1 + math.pow(W12_2, 2) * rpm2 + math.pow(W12_3, 2) * rpm3)
                            nRpms.append(rpm12[0] - B1)

                            B1 = (math.pow(W23_1, 2) * rpm1 + math.pow(W23_2, 2) * rpm2 + math.pow(W23_3, 2) * rpm3)
                            nRpms.append(rpm23[0] - B1)

                            B1 = (math.pow(W31_1, 2) * rpm1 + math.pow(W31_2, 2) * rpm2 + math.pow(W31_3, 2) * rpm3)
                            nRpms.append(rpm31[0] - B1)

                            nGammas.append(np.array([ 2 * W12_1 * W12_2, 2 * W12_1 * W12_3, 2 * W12_2 * W12_3 ]))
                            nGammas.append(np.array([ 2 * W23_1 * W23_2, 2 * W23_1 * W23_3, 2 * W23_2 * W23_3 ]))
                            nGammas.append(np.array([ 2 * W31_1 * W31_2, 2 * W31_1 * W31_3, 2 * W31_2 * W31_3 ]))

                            #ki = np.array([ V1, V2,
                                            #V3 ])
                            #t5 = plt.Polygon(ki, fill=False, color='blue', linewidth=3)
                            #plt.gca().add_patch(t5)
                            #neighboringTri =triNew.vertices[ triNew.find_simplex(trainX[index])]
                            rpms=[]
                            for s in neighboringTri:
                                V1n = dataXnew[ s ][ 0 ]
                                V2n = dataXnew[ s ][ 1 ]
                                V3n = dataXnew[ s ][ 2 ]

                                rpm1n = dataYnew[ s ][ 0 ]
                                rpm2n = dataYnew[ s ][ 1 ]
                                rpm3n = dataYnew[ s ][ 2 ]

                                rpms.append([rpm1n,rpm2n,rpm3n])
                                ###barycentric coords of neighboring points in relation to initial triangle
                                eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                                                     [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

                                eq2 = np.array([ V1n[ 0 ] - V3[ 0 ], V1n[ 1 ] - V3[ 1 ] ])
                                solutions = np.linalg.solve(eq1, eq2)

                                W1n = solutions[ 0 ]
                                W2n = solutions[ 1 ]
                                W3n = 1 - solutions[ 0 ] - solutions[ 1 ]


                                B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
                                nRpms.append(rpm1n - B1)

                                nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))
                                ####################################

                                eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                                                     [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

                                eq2 = np.array([ V2n[ 0 ] - V3[ 0 ], V2n[ 1 ] - V3[ 1 ] ])
                                solutions = np.linalg.solve(eq1, eq2)

                                W1n = solutions[ 0 ]
                                W2n = solutions[ 1 ]
                                W3n = 1 - solutions[ 0 ] - solutions[ 1 ]

                                B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
                                nRpms.append(rpm2n - B1)

                                nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))
                                ##################################################

                                eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                                                     [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

                                eq2 = np.array([ V3n[ 0 ] - V3[ 0 ], V3n[ 1 ] - V3[ 1 ] ])
                                solutions = np.linalg.solve(eq1, eq2)

                                W1n = solutions[ 0 ]
                                W2n = solutions[ 1 ]
                                W3n = 1 - solutions[ 0 ] - solutions[ 1 ]

                                B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
                                nRpms.append(rpm3n - B1)
                                nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))

                                #ki = np.array([ V1n, V2n,
                                #V3n ])
                                #t5 = plt.Polygon(ki, fill=False, color='yellow', linewidth=3)
                                #plt.gca().add_patch(t5)
                            #plt.show()
                            ####solve least squares opt. problem
                            # nRPms : y [1xn matrix]
                            # nGammas : x [3xn matrix]

                            nGammas = np.array(nGammas)
                            nRpms = np.array(nRpms)
                            from sklearn.linear_model import LinearRegression
                            lr = LinearRegression()
                            sr = sp.Earth()
                            #rf=skl.RandomForestRegressor()
                            #svm=svr.SVR(kernel='linear')
                            #svmApprxs=svm.fit(nGammas.reshape(-1, 3), nRpms.reshape(-1, 1))
                            #rfApprxs=rf.fit(nGammas.reshape(-1, 3), nRpms.reshape(-1, 1))
                            splApprx = sr.fit(nGammas.reshape(-1, 3), nRpms.reshape(-1, 1))
                            leastSqApprx = lr.fit(nGammas.reshape(-1, 3), nRpms.reshape(-1, 1))
                            XpredN = ((math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) +
                                      2 * W1 * W2 * leastSqApprx.coef_[ 0 ][ 0 ] +
                                      2 * W2 * W3 * leastSqApprx.coef_[ 0 ][ 1 ] +
                                      2 * W1 * W3 * leastSqApprx.coef_[ 0 ][ 2 ])


                            x = 1
                            if abs(XpredN - trueVal) > 5:
                                x=1



                        if  (len(tri11)==0 or len(tri22)==0 or len(tri33)==0) and flgV==False:
                            rpm31 = rpm31 if rpm31 !=0 else (rpm1+rpm2+rpm3+rpm23+rpm12)/5
                            Xpred = (math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) + \
                                2 * W1 * W2 *rpm12 + 2 * W2 * W3 * rpm23 + 2 * W1 * W3 * rpm31


                            Xpred=XpredN
                            #Xpred = [ W2 * rpm2 + W1 * rpm1 + W3 * rpm3 ]

                        elif flgV==True or flgR==True:

                            ##efaptomena trigwna <3 => trigwno pou perikliei to pointEstimate sta oria tou triangulation
                            # 2 x 2 grammiko sustima exiswsewn in order to find gammas or take the mean of neighboring points
                            ##I suneutheiakes koryfes geitonikwn trigonwn

                            ###############Approximate true val with trained ML model
                            Xpred = (math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) + \
                                2 * W1 * W2 *rpm12 + 2 * W2 * W3 * rpm23 + 2 * W1 * W3 * ((rpm1+rpm2+rpm3+rpm23+rpm12)/5)
                            Xpred = [ W2 * rpm2 + W1 * rpm1 + W3 * rpm3 ]
                            #Xpred=XpredN
                            #Xpred=modeler.getBestModelForPoint(candidatePoint.reshape(-1, 2)).predict(
                                #candidatePoint.reshape(-1, 2))
                            x=1
                            Xpred=XpredN


                            #Xpred = (modeler.getBestModelForPoint(candidatePoint.reshape(-1,2)).predict(candidatePoint.reshape(-1,2))\
                            #+(rpm1+rpm2+rpm3)/3)/2
                            #Xpred=(math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) + \
                                        #2 * W1 * W2 *rpm12 + 2 * W2 * W3 * rpm23 + 2 * W1 * W3 * rpm31
                            ################
                        elif W1==0 or W2==0 or W3==0:
                                #abs((np.mean([rpm1,rpm2,rpm3])-np.mean([rpm12,rpm23,rpm31])))>15 \
                                #or abs((np.mean([ rpm1, rpm2, rpm3 ]) - np.mean([ rpm12, rpm23, rpm31 ]))) ==0\


                                Xpred =   ((math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) + \
                                    2 * W1 * W2 * rpm12 + 2 * W2 * W3 * rpm23 + 2 * W1 * W3 * rpm31)
                                Xpred = [ W2 * rpm2 + W1 * rpm1 + W3 * rpm3 ]
                                Xpred = XpredN
                                # +(rpm1+rpm2+rpm3)/3)/2
                        #####   EKTIMISI MONO ME TETRAGWNIKOUS OROUS??
                        else:
                                eq1 = np.array([ [ 2 * W12_1 * W12_2, 2 * W12_2 * W12_3, 2 * W12_1 * W12_3 ],
                                             [ 2 * W23_1 * W23_2, 2 * W23_2 * W23_3, 2 * W23_1 * W23_3 ],
                                             [ 2 * W31_1 * W31_2, 2 * W31_2 * W31_3, 2 * W31_1 * W31_3 ] ])
                                eq2 = np.array([ rpm12 -
                                math.pow(W12_2, 2) * rpm1 - math.pow(W12_1, 2) * rpm2 - math.pow(W12_3, 2) * rpm3,

                                                 rpm23 - math.pow(W23_2, 2) * rpm1 - math.pow(W23_1, 2) * rpm2 - math.pow(
                                                 W23_3, 2) * rpm3,

                                                 rpm31 - math.pow(W31_2, 2) * rpm1 - math.pow(W31_1, 2) * rpm2 - math.pow(
                                                 W31_3, 2) * rpm3 ])
                                cond_numerGen = np.linalg.cond(eq1)
                                if cond_numerGen > 2500:

                                    Xpred = ((math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) + \
                                    2 * W1 * W2 * rpm12 + 2 * W2 * W3 * rpm23 + 2 * W1 * W3 * rpm31)
                                    Xpred = [ W2 * rpm2 + W1 * rpm1 + W3 * rpm3 ]

                                else:
                                    solutions = np.linalg.solve(eq1, eq2)
                                    Xpred = (math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) + \
                                        2 * W1 * W2 * solutions[0] + 2 * W2 * W3 * solutions[1] + 2 * W1 * W3 * solutions[2]
                                    #Xpred = [ W2 * rpm2 + W1 * rpm1 + W3 * rpm3 ]
                                    if np.var([rpm12,rpm23,rpm31])>10:
                                        Xpred=[W2 * rpm2 + W1 * rpm1 + W3 * rpm3]
                                    else:
                                        Xpred =((math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) + \
                                        2 * W1 * W2 * rpm12 + 2 * W2 * W3 * rpm23 + 2 * W1 * W3 * rpm31) if Xpred < 0 or (solutions[0]<0 or solutions[1]<0 or solutions[2]<0) else Xpred
                                        x=0
                                Xpred=XpredN
                                        #Xpred = [ W2 * rpm2 + W1 * rpm1 + W3 * rpm3 ]

                    except :
                        flgExc = True

                    break
            if flg == False:
                distList = [ ]
                preds=[]
                for k in range(0, len(triNew.vertices)):
                    #################################################################################
                    #########################################
                    V1 = dataXnew[ triNew.vertices[ k ] ][ 0 ]
                    V2 = dataXnew[ triNew.vertices[ k ] ][ 1 ]
                    V3 = dataXnew[ triNew.vertices[ k ] ][ 2 ]

                    rpm1 = dataYnew[ triNew.vertices[ k ] ][ 0 ]
                    rpm2 = dataYnew[ triNew.vertices[ k ] ][ 1 ]
                    rpm3 = dataYnew[ triNew.vertices[ k ] ][ 2 ]

                    distV1 = math.sqrt(
                        math.pow(candidatePoint[ 0 ] - V1[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V1[ 1 ], 2))
                    distV2 = math.sqrt(
                        math.pow(candidatePoint[ 0 ] - V2[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V2[ 1 ], 2))
                    distV3 = math.sqrt(
                        math.pow(candidatePoint[ 0 ] - V3[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V3[ 1 ], 2))

                    distList.append(np.mean([ distV1, distV2, distV3 ]))

                minIndexDist = distList.index(np.min(distList))
                k = minIndexDist
                neighboringVertices1 = [ ]
                #####################################
                for u in range(0, 2):
                    try:
                        neighboringVertices1.append(triNew.vertex_neighbor_vertices[ 1 ][
                                                    triNew.vertex_neighbor_vertices[ 0 ][
                                                        triNew.vertices[ k ][ u ] ]:
                                                    triNew.vertex_neighbor_vertices[ 0 ][
                                                        triNew.vertices[ k ][ u ] + 1 ] ])
                    except:
                        break
                    neighboringTri = triNew.vertices[
                        triNew.find_simplex(dataXnew[ np.concatenate(np.array(neighboringVertices1)) ]) ]
                    # simplex = triNew.vertices[ triNew.find_simplex(
                    # [ dataXnew[ triNew.vertices[ k ] ][ 0 ], dataXnew[ triNew.vertices[ k ] ][ 1 ],
                    # dataXnew[ triNew.vertices[ k ] ][ 2 ] ]) ]

                for s in neighboringTri:
                    V1 = dataXnew[ s ][ 0 ]
                    V2 = dataXnew[ s ][ 1 ]
                    V3 = dataXnew[ s ][ 2 ]

                    rpm1 = dataYnew[ s ][ 0 ]
                    rpm2 = dataYnew[ s ][ 1 ]
                    rpm3 = dataYnew[ s ][ 2 ]

                    distV1 = math.sqrt(
                        math.pow(candidatePoint[ 0 ] - V1[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V1[ 1 ], 2))
                    distV2 = math.sqrt(
                        math.pow(candidatePoint[ 0 ] - V2[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V2[ 1 ], 2))
                    distV3 = math.sqrt(
                        math.pow(candidatePoint[ 0 ] - V3[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V3[ 1 ], 2))

                    W1x = 1 / distV1 if distV1 != 0 else 0
                    W2x = 1 / distV2 if distV2 != 0 else 0
                    W3x = 1 / distV3 if distV3 != 0 else 0

                    ##Barycentric coordinates

                    W1 = (V2[ 1 ] - V3[ 1 ]) * (candidatePoint[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (
                            candidatePoint[ 1 ] - V3[ 1 ]) \
                         / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

                    W2 = (V3[ 1 ] - V1[ 1 ]) * (candidatePoint[ 0 ] - V3[ 0 ]) + (V1[ 0 ] - V3[ 0 ]) * (
                            candidatePoint[ 1 ] - V3[ 1 ]) \
                         / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

                    W3 = 1 - W1 - W2
                    ##############################################################
                    prediction = (W1 * rpm1 + W2 * rpm2 + W3 * rpm3) / (W1 + W2 + W3)
                    pred1S = (W1x * rpm1 + W2x * rpm2 + W3x * rpm3) / (W1x + W2x + W3x)
                    preds.append(pred1S)
                #################################################################################
                #########################################
                V1 = dataXnew[ triNew.vertices[ k ] ][ 0 ]
                V2 = dataXnew[ triNew.vertices[ k ] ][ 1 ]
                V3 = dataXnew[ triNew.vertices[ k ] ][ 2 ]

                rpm1 = dataYnew[ triNew.vertices[ k ] ][ 0 ]
                rpm2 = dataYnew[ triNew.vertices[ k ] ][ 1 ]
                rpm3 = dataYnew[ triNew.vertices[ k ] ][ 2 ]

                distV1 = math.sqrt(
                    math.pow(candidatePoint[ 0 ] - V1[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V1[ 1 ], 2))
                distV2 = math.sqrt(
                    math.pow(candidatePoint[ 0 ] - V2[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V2[ 1 ], 2))
                distV3 = math.sqrt(
                    math.pow(candidatePoint[ 0 ] - V3[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V3[ 1 ], 2))

                W1x = 1 / distV1 if distV1 != 0 else 0
                W2x = 1 / distV2 if distV2 != 0 else 0
                W3x = 1 / distV3 if distV3 != 0 else 0

                ##Barycentric coordinates

                W1 = (V2[ 1 ] - V3[ 1 ]) * (candidatePoint[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (
                        candidatePoint[ 1 ] - V3[ 1 ]) \
                     / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

                W2 = (V3[ 1 ] - V1[ 1 ]) * (candidatePoint[ 0 ] - V3[ 0 ]) + (V1[ 0 ] - V3[ 0 ]) * (
                        candidatePoint[ 1 ] - V3[ 1 ]) \
                     / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

                W3 = 1 - W1 - W2
                #############################

                prediction = (W1 * rpm1 + W2 * rpm2 + W3 * rpm3) / (W1 + W2 + W3)
                pred1 = (W1x * rpm1 + W2x * rpm2 + W3x * rpm3) / (W1x + W2x + W3x)
                # preds.append(prediction)
                preds.append(pred1)
                LinearPred = W1 * rpm1 + W2 * rpm2 + W3 * rpm3
                pred = (math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3)
                #####################################################################

            #if flg==False and all(i >= 0 for i in preds):
                #meanPreds=np.mean(preds) #if flg==False else pred
            if flg==True:
                meanPreds = Xpred
                errorsTue.append(abs(meanPreds - trueVal))
                if abs(meanPreds-trueVal)>2 :
                    print("Index of point : "+ str(iu)+" "+ "Index of vertice"+ str(k))
                countTrue+=1
            else:
                #meanPreds = modeler.getBestModelForPoint(candidatePoint.reshape(-1,2)).predict(candidatePoint.reshape(-1,2))
                meanPreds=np.mean(preds)
                errorsFalse.append(abs(meanPreds - trueVal))
                countFalse+=1
                    #np.mean(preds)
            lErrors.append(abs(meanPreds - trueVal))
            if abs(meanPreds - trueVal) > 10:
                x= 0

        errors = np.asarray(lErrors)
        print(np.mean(np.nan_to_num(errors)))
        print("Errors True: "+str(np.mean(errorsTue))+ "  Errors False: "+str(np.mean(errorsFalse))\
                                                                              +" Count True: " +str(countTrue)\
             +" Count False " +str(countFalse))
        print(np.mean(np.nan_to_num(errors)))
        return errors, np.mean(errors), np.std(lErrors)

    def readClusteredLarosDataFromCsvNew(self,data):
        # Load file
        dtNew = data.values[ 0:, 0:5 ].astype(float)
        dtNew = dtNew[ ~np.isnan(dtNew).any(axis=1) ]
        seriesX = dtNew
        UnseenSeriesX = dtNew
        return UnseenSeriesX

    def extractFunctionsFromSplines(self,x0, x1,modelId):
        piecewiseFunc = []
        #self.count = self.count + 1
        csvModels=['./model_'+str(modelId)+'_.csv']
        for csvM in csvModels:
            #if csvM != './model_Gen_.csv':
                #continue
            # id = csvM.split("_")[ 1 ]
            # piecewiseFunc = [ ]

            with open(csvM) as csv_file:
                data = csv.reader(csv_file, delimiter=',')
                for row in data:
                    # for d in row:
                    if [w for w in row if w == "Basis"].__len__() > 0:
                        continue
                    if [w for w in row if w == "(Intercept)"].__len__() > 0:
                        #self.interceptsGen = float(row[1])
                        continue

                    if row.__len__() == 0:
                        continue
                    d = row[0]
                    #if self.count == 1:
                        #self.intercepts.append(float(row[1]))

                    if d.split("*").__len__() == 1:
                        split = ""
                        try:
                            split = d.split('-')[0][2:4]
                            if split != "x0" and split != "x1":
                                split = d.split('-')[1]
                                num = float(d.split('-')[0].split('h(')[1])
                                # if id == id:
                                # if float(row[ 1 ]) < 10000:
                                try:
                                    # piecewiseFunc.append(
                                    # tf.math.multiply(tf.cast(tf.math.less(x, num), tf.float32),
                                    # (num - inputs)))
                                    if split.__contains__("x0"):
                                        piecewiseFunc.append((num - x0))  # * float(row[ 1 ]))
                                    if split.__contains__("x1"):
                                        piecewiseFunc.append((num - x1))  # * float(row[ 1 ]))
                                # if id ==  self.modelId:
                                # inputs = tf.where(x >= num, float(row[ 1 ]) * (inputs - num), inputs)
                                except:
                                    dc = 0
                            else:
                                ##x0 or x1
                                split = d.split('-')[0]
                                num = float(d.split('-')[1].split(')')[0])
                                # if id == id:
                                # if float(row[ 1 ]) < 10000:
                                try:
                                    if split.__contains__("x0"):
                                        piecewiseFunc.append((x0 - num))  # * float(row[ 1 ]))
                                    if split.__contains__("x1"):
                                        piecewiseFunc.append((x1 - num))  # * float(row[ 1 ]))

                                        # piecewiseFunc.append(
                                        # tf.math.multiply(tf.cast(tf.math.greater(x, num), tf.float32),
                                        # (inputs - num)))
                                # if id == self.modelId:
                                # inputs = tf.where(x <= num, float(row[ 1 ]) * (num - inputs), inputs)
                                except:
                                    dc = 0
                        except:
                            # if id == id:
                            # if float(row[ 1 ]) < 10000:
                            try:
                                piecewiseFunc.append(x0)

                                # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                # (inputs)))

                                # inputs = tf.where(x >= 0, float(row[ 1 ]) * inputs, inputs)
                            # continue
                            except:
                                dc = 0

                    else:
                        funcs = d.split("*")
                        nums = []
                        flgFirstx = False
                        flgs = []
                        for r in funcs:
                            try:
                                if r.split('-')[0][2] != "x":
                                    flgFirstx = True
                                    nums.append(float(r.split('-')[0].split('h(')[1]))

                                else:
                                    nums.append(float(r.split('-')[1].split(')')[0]))

                                flgs.append(flgFirstx)
                            except:
                                flgFirstx = False
                                flgs = []
                                split = d.split('-')[0][2]
                                try:
                                    if d.split('-')[0][2] == "x":
                                        # if id == id:
                                        # if float(row[ 1 ]) < 10000:
                                        try:

                                            if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                                                    "x1"):
                                                split = "x1"
                                            if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                                                    "x0"):
                                                split = "x0"
                                            if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                                                    "x1"):
                                                split = "x01"
                                            if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                                                    "x0"):
                                                split = "x10"

                                            if split == "x0":
                                                piecewiseFunc.append(x0 * (x0 - nums[0]) * float(row[1]))
                                            elif split == "x1":
                                                piecewiseFunc.append(x1 * (x1 - nums[0]) * float(row[1]))
                                            elif split == "x01":
                                                piecewiseFunc.append(x0 * (x1 - nums[0]) * float(row[1]))
                                            elif split == "x10":
                                                piecewiseFunc.append(x1 * (x0 - nums[0]) * float(row[1]))
                                                # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                # (inputs) * (
                                                # inputs - nums[ 0 ])))

                                                # inputs = tf.where(x >= 0,
                                                # float(row[ 1 ]) * (inputs) * (inputs - nums[ 0 ]), inputs)
                                        except:
                                            dc = 0

                                    else:
                                        flgFirstx = True
                                        # if id == id:
                                        # if float(row[ 1 ]) < 10000:
                                        try:

                                            if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                1].__contains__("x1"):
                                                split = "x1"
                                            if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                1].__contains__("x0"):
                                                split = "x0"
                                            if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                1].__contains__("x1"):
                                                split = "x01"
                                            if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                1].__contains__("x0"):
                                                split = "x10"

                                            if split == "x0":
                                                piecewiseFunc.append(x0 * (nums[0] - x0) * float(row[1]))
                                            elif split == "x1":
                                                piecewiseFunc.append(x1 * (nums[0] - x1) * float(row[1]))
                                            elif split == "x01":
                                                piecewiseFunc.append(x0 * (nums[0] - x1) * float(row[1]))
                                            elif split == "x10":
                                                piecewiseFunc.append(x1 * (nums[0] - x0) * float(row[1]))

                                            # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                            # (inputs) * (
                                            # nums[ 0 ] - inputs)))

                                            # inputs = tf.where(x > 0 ,
                                            # float(row[ 1 ]) * (inputs) * (nums[ 0 ] - inputs),inputs)
                                            flgs.append(flgFirstx)
                                        except:
                                            dc = 0

                                except:
                                    # if id == id:
                                    # if float(row[ 1 ]) < 10000:
                                    try:
                                        piecewiseFunc.append(x0)

                                        # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                        # (inputs)))

                                        # inputs = tf.where(x >= 0, float(row[ 1 ]) * (inputs), inputs)
                                    except:
                                        dc = 0
                        try:
                            # if id == id:
                            if flgs.count(True) == 2:
                                # if float(row[ 1 ])<10000:
                                try:

                                    if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                                            "x1"):
                                        split = "x1"
                                    if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                                            "x0"):
                                        split = "x0"
                                    if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                                            "x1"):
                                        split = "x01"
                                    if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                                            "x0"):
                                        split = "x10"

                                    if split == "x0":
                                        piecewiseFunc.append((nums[0] - x0) * (nums[1] - x0) * float(row[1]))
                                    elif split == "x1":
                                        piecewiseFunc.append((nums[0] - x1) * (nums[1] - x1) * float(row[1]))
                                    elif split == "x01":
                                        piecewiseFunc.append((nums[0] - x0) * (nums[1] - x1) * float(row[1]))
                                    elif split == "x10":
                                        piecewiseFunc.append((nums[0] - x1) * (nums[1] - x0) * float(row[1]))


                                except:
                                    dc = 0

                            elif flgs.count(False) == 2:
                                # if float(row[ 1 ]) < 10000:
                                try:
                                    if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                                            "x1"):
                                        split = "x1"
                                    if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                                            "x0"):
                                        split = "x0"
                                    if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                                            "x1"):
                                        split = "x01"
                                    if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                                            "x0"):
                                        split = "x10"

                                    if split == "x0":
                                        piecewiseFunc.append((x0 - nums[0]) * (x0 - nums[1]) * float(row[1]))
                                    elif split == "x1":
                                        piecewiseFunc.append((x1 - nums[0]) * (x1 - nums[1]) * float(row[1]))
                                    elif split == "x01":
                                        piecewiseFunc.append((x0 - nums[0]) * (x1 - nums[1]) * float(row[1]))
                                    elif split == "x10":
                                        piecewiseFunc.append((x1 - nums[0]) * (x0 - nums[1]) * float(row[1]))
                                        # inputs = tf.where(x > nums[ 0 ] and x > nums[ 1 ],
                                        # float(row[ 1 ]) * (inputs - nums[ 0 ]) * (
                                        # inputs - nums[ 1 ]), inputs)
                                except:
                                    dc = 0
                            else:
                                try:
                                    if flgs[0] == False:
                                        if nums.__len__() > 1:
                                            # if float(row[ 1 ]) < 10000:
                                            try:
                                                if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                    2].__contains__("x1"):
                                                    split = "x1"
                                                if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                    2].__contains__("x0"):
                                                    split = "x0"
                                                if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                    2].__contains__("x1"):
                                                    split = "x01"
                                                if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                    2].__contains__("x0"):
                                                    split = "x10"

                                                if split == "x0":
                                                    piecewiseFunc.append(
                                                        (x0 - nums[0]) * (nums[1] - x0) * float(row[1]))
                                                elif split == "x1":
                                                    piecewiseFunc.append(
                                                        (x1 - nums[0]) * (nums[1] - x1) * float(row[1]))
                                                elif split == "x01":
                                                    piecewiseFunc.append(
                                                        (x0 - nums[0]) * (nums[1] - x1) * float(row[1]))
                                                elif split == "x10":
                                                    piecewiseFunc.append(
                                                        (x1 - nums[0]) * (nums[1] - x0) * float(row[1]))




                                            # inputs = tf.where(x > nums[ 0 ] and x < nums[ 1 ],
                                            # float(row[ 1 ]) * (inputs - nums[ 0 ]) * (
                                            # nums[ 1 ] - inputs), inputs)
                                            except:
                                                dc = 0
                                        else:
                                            # if float(row[ 1 ]) < 10000:
                                            try:
                                                if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                    1].__contains__("x1"):
                                                    split = "x1"
                                                if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                    1].__contains__("x0"):
                                                    split = "x0"
                                                if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                    1].__contains__("x1"):
                                                    split = "x01"
                                                if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                    1].__contains__("x0"):
                                                    split = "x10"

                                                piecewiseFunc.append((x0 - nums[0]) * float(row[1]))



                                                # inputs = tf.where(x > nums[0],
                                                # float(row[ 1 ]) * (inputs - nums[ 0 ]), inputs)
                                            except:
                                                dc = 0
                                    else:
                                        if nums.__len__() > 1:
                                            # if float(row[ 1 ]) < 10000:
                                            try:
                                                if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                    1].__contains__("x1"):
                                                    split = "x1"
                                                if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                    1].__contains__("x0"):
                                                    split = "x0"
                                                if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                    1].__contains__("x1"):
                                                    split = "x01"
                                                if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                    1].__contains__("x0"):
                                                    split = "x10"

                                                if split == "x0":
                                                    piecewiseFunc.append(
                                                        (nums[0] - x0) * (x0 - nums[1]) * float(row[1]))
                                                elif split == "x1":
                                                    piecewiseFunc.append(
                                                        (nums[0] - x1) * (x1 - nums[1]) * float(row[1]))
                                                elif split == "x01":
                                                    piecewiseFunc.append(
                                                        (nums[0] - x0) * (x1 - nums[1]) * float(row[1]))
                                                elif split == "x10":
                                                    piecewiseFunc.append(
                                                        (nums[0] - x1) * (x0 - nums[1]) * float(row[1]))
                                                    # inputs = tf.where(x < nums[ 0 ] and x > nums[1],
                                                    # float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                                    # inputs - nums[ 1 ]), inputs)
                                            except:
                                                dc = 0
                                        else:
                                            # if float(row[ 1 ]) < 10000:
                                            try:

                                                if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                    1].__contains__("x1"):
                                                    split = "x1"
                                                if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                    1].__contains__("x0"):
                                                    split = "x0"
                                                if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                    1].__contains__("x1"):
                                                    split = "x01"
                                                if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                    1].__contains__("x0"):
                                                    split = "x10"

                                                if split == "x0":
                                                    piecewiseFunc.append((x0 - nums[0]) * float(row[1]))
                                                elif split == "x1":
                                                    piecewiseFunc.append((x1 - nums[0]) * float(row[1]))
                                                    # piecewiseFunc.append(tf.math.multiply(tf.cast(
                                                    # tf.math.less(x, nums[ 0 ]), tf.float32),
                                                    # (
                                                    # inputs - nums[ 0 ])))

                                                    # inputs = tf.where(x < nums[ 0 ],
                                                    # float(row[ 1 ]) * (
                                                    # inputs - nums[ 0 ]), inputs)
                                            except:
                                                dc = 0
                                except:
                                    dc = 0
                        except:
                            dc = 0

        return piecewiseFunc

    def evaluateKerasNN1(self, unseenX, unseenY, modeler,output,xs,genericModel,partitionsX , scores):
        lErrors = []

        from tensorflow import keras
        path = '.\\'
        clusters = '30'

        count = 0
        for iCnt in range(np.shape(unseenX)[0]):
            pPoint =unseenX[iCnt]
            pPoint= pPoint.reshape(-1,unseenX.shape[1])

            trueVal = unseenY[iCnt]

            ind, fit = modeler.getBestPartitionForPoint(pPoint, partitionsX)


            prediction = (abs(modeler._models[ind].predict(pPoint)) + modeler._models[len(modeler._models) - 1].predict(pPoint) )/ 2

            lErrors.append(abs(prediction - trueVal))

        errors = np.asarray(lErrors)

        return errors, np.mean(errors), np.std(lErrors)

    def evaluateKerasNN(self, unseenX, unseenY, modeler,output,xs,genericModel,partitionsX , scores):
        lErrors = []
        vectorWeights = scores
        #unseenX = unseenX.reshape((unseenX.shape[ 0 ], 1, unseenX.shape[ 1 ]))
        #from sklearn.decomposition import PCA
        #pca = PCA()
        #pca.fit(unseenX)
        #unseenX = pca.transform(unseenX)
        from tensorflow import keras
        path = '.\\'
        clusters = '30'
        #partitionsX = [ ]
        #partitionsY = [ ]
        #for cl in range(0,30):
            # models.append(load_model(path+'\estimatorCl_'+str(cl)+'.h5'))
            #data = pd.read_csv('cluster_' + str(cl) + '_.csv')
            #partitionsX.append(self.readClusteredLarosDataFromCsvNew(data))

        #for cl in range(0,30):
            # models.append(load_model(path+'\estimatorCl_'+str(cl)+'.h5'))
            #data = pd.read_csv('cluster_foc' + str(cl) + '_.csv')
            #partitionsY.append(self.readClusteredLarosDataFromCsvNew(data))

        #from sklearn import preprocessing
        #scalerX = preprocessing.StandardScaler()
        #scalerY =preprocessing.StandardScaler()
        ##scalerX = scalerX.fit(np.concatenate(partitionsX))
        #scalerY = scalerY.fit(np.concatenate(partitionsY).reshape(-1, 1))
        count = 0
        for iCnt in range(np.shape(unseenX)[0]):
            pPoint =unseenX[iCnt]
            pPoint= pPoint.reshape(-1,unseenX.shape[1])


            preds = [ ]
            #listOfPoints = np.array(listOfPoints.split('[')[ 1 ].split(']')[ 0 ].split(',')).astype(np.float)
            #listOfPoints = listOfPoints.reshape(-1, 4)
            #pPoint = pPoint.reshape(-1,1,2)
            #pPoint = pPoint.reshape((pPoint.shape[ 0 ], 1, pPoint.shape[ 1 ]))
             # Convert to matrix
            #for vector in pPoint:
                #ind, fit = modeler.getBestPartitionForPoint(vector, partitionsX)
                #currModeler = keras.models.load_model('estimatorCl_rpm' + str(ind) + '.h5')
                #vector = vector.reshape(-1, 4)
                #rpm = currModeler.predict(vector)
                #preds.append(prediction[ 0 ][ 0 ])
                #pPoint = np.append(pPoint, [ rpm ])

            #try:
            trueVal = unseenY[iCnt]
            #except:
            #z=0
            #pPoint=[ pPoint for _ in range(len(modeler._models[0].input)) ]
            #trueVal = unseenY[ iCnt ]
            #prediction = modeler._models[0].model.predict([unseenX[:,0].reshape(1,unseenX.shape[0],1) , unseenX[ :, 1 ].reshape(1, unseenX.shape[ 0 ], 1)])
            #q = model1.predict(pPoint, verbose=0)
            #cl = q.argmax(1)
            #weight= model2.layers[1].get_weights()[0][0,:][cl]
            #prediction = np.sum(model2.predict(pPoint, verbose=0)) / 5

            #ind , fit = modeler.getBestPartitionForPoint(pPoint,partitionsX)
            ind, fit = modeler.getBestPartitionForPoint(pPoint, partitionsX)
            #fits = modeler.getFitsOfPoint(partitionsX,pPoint)
            #fit  = modeler.getFitofPointIncluster(pPoint,centroids[ind])
            #scaledPoint = scalerX.fit_transform(pPoint)


            #scaledPoint = scalerX.fit_transform(pPoint)


            #fits = modeler.getFitForEachPartitionForPoint(pPoint, partitionsX)
            #weightedPreds=[]
            #for n,model in enumerate(modeler._models):
                #weightedPreds.append(modeler._models[n].predict(pPoint)*fits[n])
            #weightedPreds.append(modeler._models[len(modeler._models)-1].predict(pPoint))
            #predX = np.mean(weightedPreds)

            #prediction = modeler._models[0].predict(pPoint)
            #prediction = modeler._models[len(modeler._models)-1].predict(pPoint)
            #from scipy.special import softmax
            #fits = softmax(fits)
            #pred=modeler._models[len(modeler._models)-1].predict(pPoint)
            #for i in range(0,len(fits)):
                #pred+=fits[i]*modeler._models[i].predict(pPoint)

            vector = self.extractFunctionsFromSplines(pPoint[0][0],pPoint[0][1],ind)
            XSplineVector=np.append(pPoint, vector)
            XSplineVector = XSplineVector.reshape(-1, XSplineVector.shape[ 0 ])

            vector = self.extractFunctionsFromSplines(pPoint[ 0 ][ 0 ], pPoint[ 0 ][ 1 ], 'Gen')
            XSplineGenVector = np.append(pPoint, vector)
            XSplineGenVector = XSplineGenVector.reshape(-1, XSplineGenVector.shape[ 0 ])
            #prediction = abs(modeler._models[ 0 ].predict(XSplineVector))
            #XSplineVector = XSplineGenVector if modeler._models[ind][1]=='GEN' else XSplineVector
            #prediction = (abs(modeler._models[ind].predict(XSplineVector)) + modeler._models[len(modeler._models)-1].predict(XSplineGenVector))/2
            prediction = (abs(modeler._models[ind].predict(XSplineVector)) + modeler._models[len(modeler._models) - 1].predict(XSplineGenVector) )/ 2
            #try:
                #if modeler._models[ ind ][ 1 ] == 'GEN':
            #prediction = modeler._models[ len(modeler._models) - 1 ][ 0 ].predict(XSplineGenVector)
                #else:
                    #prediction = (abs(modeler._models[ind][0].predict(XSplineVector)) + modeler._models[len(modeler._models)-1][0].predict(XSplineGenVector))/2
                #lErrors.append(abs(prediction - trueVal))
            #except:
                #prediction = modeler._models[len(modeler._models) - 1][0].predict(XSplineGenVector)
                #count=count+1
                #abs(modeler._models[ind].predict(XSplineVector))
                #(abs(modeler._models[ind].predict(XSplineVector)) + modeler._models[len(modeler._models)-1].predict(XSplineGenVector))/2
                #abs(modeler._models[ind].predict(XSplineVector)) if fit > 0.5 else  modeler._models[len(modeler._models)-1].predict(XSplineGenVector)

            #prediction = (scalerY.inverse_transform(prediction))  # + scalerY.inverse_transform(currModeler1.predict(scaled))) / 2
            #states_value = modeler._models[0].model.predict([unseenX[:,0].reshape(1,unseenX.shape[200],1) , unseenX[ :, 1 ].reshape(1, unseenX.shape[ 200 ], 1)])

            ##########

            #if abs(prediction - trueVal)>10:
                #w=0
        #x=unseenX[:,0].reshape(-1,unseenX.shape[0])
            lErrors.append(abs(prediction - trueVal))
        #prediction =modeler._models[ 0 ].predict(unseenX.reshape(2,2860))
        #print np.mean(abs(prediction - unseenY))
        #print("EXCEPTIONS :  "+str(count))
        errors = np.asarray(lErrors)

        return errors, np.mean(errors), np.std(lErrors)



    def ANOVAtest(self,clusters,var,error,models,partitioners):

        df = pd.DataFrame({
                            'clusters': clusters,
                            'var': var,
                            'error':error,
                            'partitioners':partitioners,
                            'models':models
                            #'meanBearing':trFeatures[1]
                            })
        groups=[error,clusters,var]


        #data=[error,np.var(unseenX),clusters]
        print(stats.kruskal(error,clusters,var,models,partitioners))
        #print(self.kw_dunn(groups,[(0,1),(0,2)]))

        #dataf = pd.DataFrame.from_dict(df, orient='index')
        df.to_csv('./NEWres_1.csv', index=False)
        #df.melt(var_name='groups', value_name='values')
        df=pd.melt(df,var_name='groups', value_name='values')
        print(df)
        #df2=sp.posthoc_dunn(df, val_col='values', group_col='groups')
        #df2.to_csv('./kruskal.csv', index=False)
        #print(df2)


        #formula = 'error ~ C(clusters) + C(var) + C(error)'
        ##model =  ols(formula, df).fit()
        #aov_table = statsmodels.stats.anova.anova_lm(model, typ=2)
        #print(aov_table)
