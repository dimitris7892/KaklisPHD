import numpy as np
import tensorflow as tf
from sklearn.cluster import DBSCAN
import pandas as pd
from scipy import stats
import math
from scipy.spatial import Delaunay, ConvexHull
import csv
from numpy import array
import matplotlib.pyplot as plt
import pyearth as sp
from matplotlib import rc
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
#rc('text', usetex=True)
font = {'family': 'normal',
        'weight': 'bold',
        'size': 14}

plt.rc('font', **font)


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
    def evaluate(self, unseenX, unseenY, modeler,genericModel,partitionsX):
        lErrors = []
        for iCnt in range(np.shape(unseenX)[0]):
            pPoint = unseenX[iCnt].reshape(1, -1)#[0] # Convert to matrix
            trueVal = unseenY[iCnt]
            preds=[]
            # fits = modeler.getFitForEachPartitionForPoint(pPoint, partitionsX)

            if modeler.__class__.__name__ == 'TensorFlowCA':
                if len(partitionsX)> 1:
                    preds = []
                    for n in range(0, len(partitionsX)):
                            preds.append(modeler._models[n].predict(pPoint)[0][0])
                #trainedWeights = genericModel
                prediction = modeler._models[len(modeler._models) - 1].predict(pPoint)
                preds.append(prediction)
                if len(partitionsX) > 1:
                    prediction = genericModel.predict(np.array(preds).reshape(-1,len(partitionsX)+1))
                    #np.average(np.array(preds), weights=trainedWeights.reshape(len(preds)))
                else:
                    prediction =  prediction
            else:
                prediction = (modeler.getBestModelForPoint(pPoint).predict(pPoint) + modeler._models[len(modeler._models) - 1].predict(pPoint)) / 2


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
        '''EVALUATION AND IMPLEMENTATION OF TRIANGULATION BASED INTERPOLATION
        '''

        lErrors = [ ]

        dataXnew = trainX
        dataYnew = trainY

        #########

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
        '''DELETE ALL TRIANGLES WITH CORRESPONDING VARIANCE OF RPM of their points belonging in their adjacency list BIGGER  THAN A PREDEFINED TRESHOLD = > MORE stable triangulation'''
        '''????'''
        '''for i in range(0, len(dataXnew)):
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
                            delIndexes.append(i)'''
        ##################################3
        #dataXnew = np.delete(dataXnew,delIndexes,axis=0)
        #dataYnew = np.delete(dataYnew, delIndexes)
        '''END OF PRE - PROCESSING STEP'''


        #vertices = self.plotly_trisurf(dataXnew[ :, 0 ], zx, dataXnew[ :, 1 ], triNew.simplices)
        #self.plotly_trisurf(dataX[ tri.vertices[ 0 ] ][ :, 0 ], zx[ 0:3 ], dataX[ tri.vertices[ 0 ] ][ :, 1 ],
         #tri.simplices)
        #plt.plot(unseenX[ 0:, 0 ], unseenX[ 0:, 1 ], 'o', markersize=8,color='blue')
        #for v in vertices:
            #k = np.array([ [ v[ 0 ][ 0 ], v[ 0 ][ 2 ] ], [ v[ 1 ][ 0 ], v[ 1 ][ 2 ] ], [ v[ 2 ][ 0 ], v[ 2 ][ 2 ] ] ])
            #t = plt.Polygon(k, fill=False,color='red',linewidth=3)
            #plt.gca().add_patch(t)
        #plt.show()

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

            #if varPerTri > 10:
                # delSimplices.append(int(i))
                #flagedVertices.append(int(i))

        ###build triangulations on clusters
        #triangulationsClusters = [ ]
        #for i in range(0, len(trainX)):
            #try:
                #triangulationsClusters.append(Delaunay(trainX[ i ]))
            #except Exception,e:
                #print(str(e))
                #x=0




        errorsTue=[]
        errorsFalse=[]
        countTrue = 0
        countFalse = 0
        predictions=[]
        yTrue=[]
        for iu in range(0,len(unseenX)):

            flg = False
            trueVal = unseenY[ iu ]
            candidatePoint = unseenX[ iu ]

            preds=[]


            for k in range(0 ,len(triNew.vertices)):

                V1 = dataXnew[triNew.vertices[k]][0]
                V2 = dataXnew[triNew.vertices[k]][1]
                V3 = dataXnew[triNew.vertices[k]][2]


                b = triNew.transform[ k, :2 ].dot(candidatePoint - triNew.transform[ k, 2 ])
                W1 = b[ 0 ]
                W2 = b[ 1 ]
                W3 = 1 - np.sum(b)
                #################3##########
                '''if point inside triangle'''
                if W1>0 and W2>0 and W3>0:

                    flg  = True######


                    rpm1 = dataYnew[ triNew.vertices[ k ] ][ 0 ]
                    rpm2 = dataYnew[ triNew.vertices[ k ] ][ 1 ]
                    rpm3 = dataYnew[ triNew.vertices[ k ] ][ 2 ]

                    flgR=False
                    flgV=False

                    ##efaptomena trigwna panw sto trigwno sto opoio anikei to Point estimate
                    tri1 =np.array(triNew.vertices[
                    [ i for i, x in enumerate(list(triNew.vertices[ :, 0 ] == triNew.vertices[ k, 0 ] )) if x ] ])



                    tri11=[]
                    k_set = set(triNew.vertices[k])

                    for tr in tri1:
                        a_set = set(tr)
                        if len(k_set.intersection(a_set)) == 2 and\
                                 (tr !=triNew.vertices[k]).any():
                            tri11.append(tr)

                    tri11 = np.array(tri11)
                    neighboringTri1 = dataXnew[tri11] if tri11.__len__() > 0 else []
                    neighboringTri1rpm = dataYnew[tri11] if tri11.__len__() > 0 else []

                    tri2 =np.array( triNew.vertices[
                        [ i for i, x in enumerate(list(triNew.vertices[ :, 1 ] == triNew.vertices[ k, 1 ])) if x ] ])



                    #tri2 = np.delete(tri2, triNew.vertices[ k ])
                    tri22 = [ ]
                    for tr in tri2:
                        b_set = set(tr)
                        if len(k_set.intersection(b_set)) == 2 \
                                and (tr !=triNew.vertices[k]).any():
                            tri22.append(tr)
                    tri22 = np.array(tri22)
                    neighboringTri2 = dataXnew[tri22] if tri22.__len__() > 0 else []
                    neighboringTri2rpm = dataYnew[tri22] if tri22.__len__() > 0 else []

                    tri3 =np.array( triNew.vertices[
                        [ i for i, x in enumerate(list(triNew.vertices[ :, 2 ] == triNew.vertices[ k, 2 ])) if x ] ])


                    #tri3 = np.delete(tri3, triNew.vertices[ k ])
                    tri33 = [ ]


                    for tr in tri3:
                        c_set = set(tr)
                        if len(k_set.intersection(c_set)) == 2 \
                                and (tr !=triNew.vertices[k]).any():
                            tri33.append(tr)

                    tri33 = np.array(tri33)
                    neighboringTri3 = dataXnew[tri33] if tri33.__len__() > 0 else []
                    neighboringTri3rpm = dataYnew[tri33] if tri33.__len__() > 0 else []

                    ######
                    #attched_vertex = np.concatenate([tri11,tri22,tri33])

                    rpms=[]
                    vs=[]
                    if tri11!=[]:
                        for tr in tri11:
                            vi=[ x for x in tr if x not in triNew.vertices[k] ]
                            rpms.append( dataYnew[ vi ])
                            vs.append( dataXnew[ vi ][0])

                    if tri22!=[]:
                        for tr in tri22:
                            vi=[ x for x in tr if x not in triNew.vertices[k] ]
                            rpms.append( dataYnew[ vi ])
                            vs.append(dataXnew[ vi ][0])

                    if tri33!=[]:
                        for tr in tri33:
                            vi=[ x for x in tr if x not in triNew.vertices[k] ]
                            rpms.append( dataYnew[ vi ])
                            vs.append(dataXnew[ vi ][0])

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

                    ##barycenters (W12,W23,W13) of neighboring triangles and solutions of linear 3x3 sustem in order to find gammas.
                    try:

                        ###################################################
                        eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                                         [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

                        eq2 = np.array([ v12 - V3[ 0 ], v12 - V3[ 1 ] ])
                        solutions = np.linalg.solve(eq1, eq2)

                        W12_1 = solutions[ 0 ]
                        W12_2 = solutions[ 1 ]
                        W12_3 = 1 - solutions[ 0 ] - solutions[ 1 ]

                        cond_numer1 = np.linalg.cond(eq1)
                        ###########################

                        eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                                         [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

                        eq2 = np.array([ v23 - V3[ 0 ], v23 - V3[ 1 ] ])
                        solutions = np.linalg.solve(eq1, eq2)

                        W23_1 = solutions[ 0 ]
                        W23_2 = solutions[ 1 ]
                        W23_3 = 1 - solutions[ 0 ] - solutions[ 1 ]
                        cond_numer2 = np.linalg.cond(eq1)
                        ####################################################

                        eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                                         [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

                        eq2 = np.array([ v31- V3[ 0 ], v31 - V3[ 1 ] ])
                        solutions = np.linalg.solve(eq1, eq2)

                        W31_1 = solutions[ 0 ]
                        W31_2 = solutions[ 1 ]
                        W31_3 = 1 - solutions[ 0 ] - solutions[ 1 ]
                        cond_numer3 = np.linalg.cond(eq1)
                    ####################################
                    ####################################
                        b = triNew.transform[k, :2].dot(vs[0] - triNew.transform[k, 2])
                        W12_1 = b[0]
                        W12_2 = b[1]
                        W12_3 = 1 - np.sum(b)
                    ####################################
                        b = triNew.transform[k, :2].dot(vs[1] - triNew.transform[k, 2])
                        W23_1 = b[0]
                        W23_2 = b[1]
                        W23_3 = 1 - np.sum(b)
                    ##############################
                        b = triNew.transform[k, :2].dot(vs[2] - triNew.transform[k, 2])
                        W31_1 = b[0]
                        W31_2 = b[1]
                        W31_3 = 1 - np.sum(b)
                    #####################################
                        eq1 = np.array([[2 * W12_1 * W12_2, 2 * W12_2 * W12_3, 2 * W12_1 * W12_3],
                                        [2 * W23_1 * W23_2, 2 * W23_2 * W23_3, 2 * W23_1 * W23_3],
                                        [2 * W31_1 * W31_2, 2 * W31_2 * W31_3, 2 * W31_1 * W31_3]])
                        eq2 = np.array([rpm12 -
                                        math.pow(W12_2, 2) * rpm1 - math.pow(W12_1, 2) * rpm2 - math.pow(W12_3, 2) * rpm3,

                                        rpm23 - math.pow(W23_2, 2) * rpm1 - math.pow(W23_1, 2) * rpm2 - math.pow(
                                            W23_3, 2) * rpm3,

                                        rpm31 - math.pow(W31_2, 2) * rpm1 - math.pow(W31_1, 2) * rpm2 - math.pow(
                                            W31_3, 2) * rpm3])
                        cond_numerGen = np.linalg.cond(eq1)

                        solutions = np.linalg.solve(eq1, eq2)
                        ypredS = (math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) + \
                                    2 * W1 * W2 * solutions[0] + 2 * W2 * W3 * solutions[1] + 2 * W1 * W3 * solutions[2]

                        predictions.append(ypredS)
                        yTrue.append(trueVal)

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
                        edgesInitial=[]



                        if  (len(tri11)==0 or len(tri22)==0 or len(tri33)==0) and flgV==False:
                            rpm31 = rpm31 if rpm31 !=0 else (rpm1+rpm2+rpm3+rpm23+rpm12)/5
                            Xpred = (math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) + \
                                2 * W1 * W2 *rpm12 + 2 * W2 * W3 * rpm23 + 2 * W1 * W3 * rpm31


                            if iu in np.arange(0,40):
                                plt.clf()
                                plt.plot(candidatePoint[0], candidatePoint[1], 'o', markersize=10)  ####

                                if neighboringTri1.__len__()>0:
                                    for v in neighboringTri1:
                                        r = np.array(v)
                                        t = plt.Polygon(r, fill=False, color='green', linewidth=3)
                                        plt.gca().add_patch(t)


                                if neighboringTri2.__len__() > 0:
                                    for v in neighboringTri2:
                                        r = np.array(v)
                                        t = plt.Polygon(r, fill=False, color='green', linewidth=3)
                                        plt.gca().add_patch(t)


                                if neighboringTri3.__len__() > 0:
                                    for v in neighboringTri3:
                                        r = np.array(v)
                                        t = plt.Polygon(r, fill=False, color='green', linewidth=3)
                                        plt.gca().add_patch(t)



                                r = np.array([[V1[0], V1[1]], [V2[0], V2[1]], [V3[0], V3[1]]])
                                t = plt.Polygon(r, fill=False, color='red', linewidth=3)
                                plt.gca().add_patch(t)

                                #plt.xlabel(r'V', fontsize=16)
                                #plt.ylabel(r'\bar{V}_N', fontsize=16)

                                rpm= [rpm1,  rpm2,  rpm3,  rpm12,    rpm23,    rpm31   ]
                                v=   [V1[0], V2[0], V3[0], vs[0][0], vs[1][0], vs[2][0]]
                                vn = [V1[1], V2[1], V3[1], vs[0][1], vs[1][1], vs[2][1]]

                                #=(math.pow(W2, 2) * x + math.pow(W1, 2) * x + math.pow(W3, 2) * x) + \
                                        #2 * W1 * W2 * solutions[0] + 2 * W2 * W3 * solutions[1] + 2 * W1 * W3 * solutions[2]

                                plt.grid()

                                fig = plt.figure()
                                #ax = Axes3D(fig)
                                ax = fig.add_subplot(111, projection='3d')
                                x = [v[0], v[1], v[2]]
                                y = [vn[0], vn[1], vn[2]]
                                z = [0,0,0]
                                    #[rpm1, rpm2, rpm3]
                                verts = [list(zip(x, y, z))]
                                if neighboringTri1.__len__()==2 and neighboringTri2.__len__()==1:
                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='red'),)

                                    x = [neighboringTri1[0][0][0], neighboringTri1[0][1][0], neighboringTri1[0][2][0]]
                                    y = [neighboringTri1[0][0][1], neighboringTri1[0][1][1], neighboringTri1[0][2][1]]
                                    z = [0,0,0]
                                        #[neighboringTri1rpm[0][0], neighboringTri1rpm[0][1], neighboringTri1rpm[0][2]]
                                    verts = [list(zip(x, y, z))]

                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='green'), )

                                    x = [neighboringTri2[0][0][0], neighboringTri2[0][1][0], neighboringTri2[0][2][0]]
                                    y = [neighboringTri2[0][0][1], neighboringTri2[0][1][1], neighboringTri2[0][2][1]]
                                    z = [0,0,0]
                                        #[neighboringTri2rpm[0][0], neighboringTri2rpm[0][1], neighboringTri2rpm[0][2]]
                                    verts = [list(zip(x, y, z))]

                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='green', ))

                                    x = [neighboringTri1[1][0][0], neighboringTri1[1][1][0], neighboringTri1[1][2][0]]
                                    y = [neighboringTri1[1][0][1], neighboringTri1[1][1][1], neighboringTri1[1][2][1]]
                                    z = [0,0,0]
                                        #[neighboringTri1rpm[1][0], neighboringTri1rpm[1][1], neighboringTri1rpm[1][2]]
                                    verts = [list(zip(x, y, z))]
                                    #ax.plot_trisurf(x, y, z,
                                                    #linewidth=0.2, antialiased=True)
                                    ax.add_collection3d(Poly3DCollection(verts ,facecolors='green',) )

                                elif neighboringTri1.__len__() == 2 and neighboringTri3.__len__() == 1:
                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='red'), )

                                    x = [neighboringTri1[0][0][0], neighboringTri1[0][1][0], neighboringTri1[0][2][0]]
                                    y = [neighboringTri1[0][0][1], neighboringTri1[0][1][1], neighboringTri1[0][2][1]]
                                    z = [0, 0, 0]
                                    # [neighboringTri1rpm[0][0], neighboringTri1rpm[0][1], neighboringTri1rpm[0][2]]
                                    verts = [list(zip(x, y, z))]

                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='green'), )

                                    x = [neighboringTri3[0][0][0], neighboringTri3[0][1][0], neighboringTri3[0][2][0]]
                                    y = [neighboringTri3[0][0][1], neighboringTri3[0][1][1], neighboringTri3[0][2][1]]
                                    z = [0, 0, 0]
                                    # [neighboringTri2rpm[0][0], neighboringTri2rpm[0][1], neighboringTri2rpm[0][2]]
                                    verts = [list(zip(x, y, z))]

                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='green', ))

                                    x = [neighboringTri1[1][0][0], neighboringTri1[1][1][0], neighboringTri1[1][2][0]]
                                    y = [neighboringTri1[1][0][1], neighboringTri1[1][1][1], neighboringTri1[1][2][1]]
                                    z = [0, 0, 0]
                                    # [neighboringTri1rpm[1][0], neighboringTri1rpm[1][1], neighboringTri1rpm[1][2]]
                                    verts = [list(zip(x, y, z))]
                                    # ax.plot_trisurf(x, y, z,
                                    # linewidth=0.2, antialiased=True)
                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='green', ))

                                elif neighboringTri2.__len__() == 2 and neighboringTri1.__len__()==1:
                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='red'), )

                                    x = [neighboringTri1[0][0][0], neighboringTri1[0][1][0], neighboringTri1[0][2][0]]
                                    y = [neighboringTri1[0][0][1], neighboringTri1[0][1][1], neighboringTri1[0][2][1]]
                                    z = [0, 0, 0]
                                    # [neighboringTri1rpm[0][0], neighboringTri1rpm[0][1], neighboringTri1rpm[0][2]]
                                    verts = [list(zip(x, y, z))]

                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='green'), )

                                    x = [neighboringTri2[0][0][0], neighboringTri2[0][1][0], neighboringTri2[0][2][0]]
                                    y = [neighboringTri2[0][0][1], neighboringTri2[0][1][1], neighboringTri2[0][2][1]]
                                    z = [0, 0, 0]
                                    # [neighboringTri2rpm[0][0], neighboringTri2rpm[0][1], neighboringTri2rpm[0][2]]
                                    verts = [list(zip(x, y, z))]

                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='green', ))

                                    x = [neighboringTri2[1][0][0], neighboringTri2[1][1][0], neighboringTri2[1][2][0]]
                                    y = [neighboringTri2[1][0][1], neighboringTri2[1][1][1], neighboringTri2[1][2][1]]
                                    z = [0, 0, 0]
                                    # [neighboringTri1rpm[1][0], neighboringTri1rpm[1][1], neighboringTri1rpm[1][2]]
                                    verts = [list(zip(x, y, z))]
                                    # ax.plot_trisurf(x, y, z,
                                    # linewidth=0.2, antialiased=True)
                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='green', ))

                                elif neighboringTri3.__len__() == 2 and neighboringTri1.__len__()==1:
                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='red'), )

                                    x = [neighboringTri1[0][0][0], neighboringTri1[0][1][0], neighboringTri1[0][2][0]]
                                    y = [neighboringTri1[0][0][1], neighboringTri1[0][1][1], neighboringTri1[0][2][1]]
                                    z = [0, 0, 0]
                                    # [neighboringTri1rpm[0][0], neighboringTri1rpm[0][1], neighboringTri1rpm[0][2]]
                                    verts = [list(zip(x, y, z))]

                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='green'), )

                                    x = [neighboringTri3[0][0][0], neighboringTri3[0][1][0], neighboringTri3[0][2][0]]
                                    y = [neighboringTri3[0][0][1], neighboringTri3[0][1][1], neighboringTri3[0][2][1]]
                                    z = [0, 0, 0]
                                    # [neighboringTri2rpm[0][0], neighboringTri2rpm[0][1], neighboringTri2rpm[0][2]]
                                    verts = [list(zip(x, y, z))]

                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='green', ))

                                    x = [neighboringTri3[1][0][0], neighboringTri3[1][1][0], neighboringTri3[1][2][0]]
                                    y = [neighboringTri3[1][0][1], neighboringTri3[1][1][1], neighboringTri3[1][2][1]]
                                    z = [0, 0, 0]
                                    # [neighboringTri1rpm[1][0], neighboringTri1rpm[1][1], neighboringTri1rpm[1][2]]
                                    verts = [list(zip(x, y, z))]
                                    # ax.plot_trisurf(x, y, z,
                                    # linewidth=0.2, antialiased=True)
                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='green', ))

                                elif neighboringTri3.__len__() == 2 and neighboringTri2.__len__()==1:
                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='red'), )

                                    x = [neighboringTri2[0][0][0], neighboringTri2[0][1][0], neighboringTri2[0][2][0]]
                                    y = [neighboringTri2[0][0][1], neighboringTri2[0][1][1], neighboringTri2[0][2][1]]
                                    z = [0, 0, 0]
                                    # [neighboringTri1rpm[0][0], neighboringTri1rpm[0][1], neighboringTri1rpm[0][2]]
                                    verts = [list(zip(x, y, z))]

                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='green'), )

                                    x = [neighboringTri3[0][0][0], neighboringTri3[0][1][0], neighboringTri3[0][2][0]]
                                    y = [neighboringTri3[0][0][1], neighboringTri3[0][1][1], neighboringTri3[0][2][1]]
                                    z = [0, 0, 0]
                                    # [neighboringTri2rpm[0][0], neighboringTri2rpm[0][1], neighboringTri2rpm[0][2]]
                                    verts = [list(zip(x, y, z))]

                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='green', ))

                                    x = [neighboringTri3[1][0][0], neighboringTri3[1][1][0], neighboringTri3[1][2][0]]
                                    y = [neighboringTri3[1][0][1], neighboringTri3[1][1][1], neighboringTri3[1][2][1]]
                                    z = [0, 0, 0]
                                    # [neighboringTri1rpm[1][0], neighboringTri1rpm[1][1], neighboringTri1rpm[1][2]]
                                    verts = [list(zip(x, y, z))]
                                    # ax.plot_trisurf(x, y, z,
                                    # linewidth=0.2, antialiased=True)
                                    ax.add_collection3d(Poly3DCollection(verts, facecolors='green', ))


                                ax.scatter(candidatePoint[0], candidatePoint[1],trueVal,marker='o',s=60,)

                                x = []
                                y = []
                                z = []
                                for i in range(0, int(trueVal)-10, 10):
                                    x.append(candidatePoint[0])
                                    y.append(candidatePoint[1])
                                    z.append(i)
                                x.append(candidatePoint[0])
                                y.append(candidatePoint[1])
                                z.append(trueVal)
                                ax.plot(x, y, z, '--', alpha=0.8, linewidth=1, )

                                #ax.scatter(candidatePoint[0], candidatePoint[1], 0, marker='o', s=200, )

                                def point_on_triangle(pt1, pt2, pt3):
                                    """
                                    Random point on the triangle with vertices pt1, pt2 and pt3.
                                    """
                                    s, t = sorted([np.random.random(), np.random.random()])
                                    return (s * pt1[0] + (t - s) * pt2[0] + (1 - t) * pt3[0],
                                            s * pt1[1] + (t - s) * pt2[1] + (1 - t) * pt3[1])

                                rpmApprxs = []
                                from matplotlib import cm
                                pointsInside = [point_on_triangle(V1, V2, V3) for _ in range(1500)]
                                pointsInside.append(candidatePoint)
                                for  i in range(0,len(pointsInside)):
                                    b = triNew.transform[k, :2].dot(pointsInside[i] - triNew.transform[k, 2])
                                    W1i = b[0]
                                    W2i = b[1]
                                    W3i = 1 - np.sum(b)

                                    rpmApprx =   (math.pow(W2i, 2) * rpm2 + math.pow(W1i, 2) * rpm1 + math.pow(W3i, 2) * rpm3) + \
                                    2 * W1i * W2i * solutions[0] + 2 * W2i * W3i * solutions[1] + 2 * W1i * W3i * solutions[2]
                                    rpmApprxs.append(rpmApprx)
                                    #rpmApprxs.append(rpmApprx)


                                #pointsInside.append((V1[0],V1[1]))
                                #pointsInside.append((V2[0],V2[1]))
                                #pointsInside.append((V3[0],V3[1]))
                                #rpmApprxs.append([rpm1])
                                #rpmApprxs.append([rpm2])
                                #rpmApprxs.append([rpm3])
                                rpmApprxs = np.array(rpmApprxs).reshape(len(pointsInside))

                                #verts = [list(zip(np.array(pointsInside)[:,0], np.array(pointsInside)[:,1], rpmApprxs))]
                                #ax.add_collection3d(Poly3DCollection(verts, facecolors='lightblue',antialiased=True ))

                                #ax.plot_wireframe(np.array(pointsInside)[:,0], np.array(pointsInside)[:,1], rpmApprxs)
                                ax.plot_trisurf(np.array(pointsInside)[:,0], np.array(pointsInside)[:,1], rpmApprxs, linewidth=1, antialiased=True,color='lightblue',shade=True)
                                #ax.contour3D(np.array(pointsInside)[:,0], np.array(pointsInside)[:,1], rpmApprxs, 50, cmap='binary')
                                #ax.plot_surface(np.array(pointsInside)[:,0], np.array(pointsInside)[:,1], rpmApprxs, rstride=1, cstride=1, alpha=0.2,antialiased=True)
                                #X, Y = np.meshgrid(np.array(pointsInside)[:, 0],  np.array(pointsInside)[:, 1])
                                #ax.plot_surface(X, Y,
                                                #np.array(rpmApprxs),
                                                #linewidth=0, antialiased=True)
                                #ax.scatter(pointsInside[i][0], pointsInside[i][1], rpmApprx, marker='o', s=10,
                                               #c='black')





                                ax.scatter(V1[0], V1[1], rpm1, marker='o', s=60,c='red')
                                ax.scatter(V2[0], V2[1], rpm2, marker='o', s=60, c='red')
                                ax.scatter(V3[0], V3[1], rpm3, marker='o', s=60, c='red')

                                #ax.scatter(candidatePoint[0], candidatePoint[1], ypredS, marker='o', s=50,c='blue')

                                x=[]
                                y=[]
                                z=[]
                                for i in range(0,int(rpm1)-10,10):
                                    x.append(V1[0])
                                    y.append(V1[1])
                                    z.append(i)

                                x.append(V1[0])
                                y.append(V1[1])
                                z.append(rpm1)
                                ax.plot(x, y, z, '--', alpha=0.8, linewidth=0.7, c='red')
                                x = []
                                y = []
                                z = []
                                for i in range(0, int(rpm2)-10, 10):
                                    x.append(V2[0])
                                    y.append(V2[1])
                                    z.append(i)
                                x.append(V2[0])
                                y.append(V2[1])
                                z.append(rpm2)

                                ax.plot(x, y, z, '--', alpha=0.8, linewidth=0.7, c='red')
                                x = []
                                y = []
                                z = []
                                for i in range(0, int(rpm2)-10, 10):
                                    x.append(V3[0])
                                    y.append(V3[1])
                                    z.append(i)

                                x.append(V3[0])
                                y.append(V3[1])
                                z.append(rpm3)
                                ax.plot(x, y, z, '--', alpha=0.8, linewidth=0.7,c='red')

                                v12=vs[0]
                                v23 = vs[1]
                                v31 = vs[2]
                                #ax.scatter(v12[0], v12[1], rpm12, marker='o', s=60, c='green')
                                #ax.scatter(v23[0], v23[1], rpm23, marker='o', s=60, c='green')
                                #ax.scatter(v31[0], v31[1], rpm31, marker='o', s=60, c='green')
                                x = []
                                y = []
                                z = []
                                for i in range(0, int(rpm12)-10, 10):
                                    x.append(v12[0])
                                    y.append(v12[1])
                                    z.append(i)
                                x.append(v12[0])
                                y.append(v12[1])
                                z.append(rpm12)
                                #ax.plot(x, y, z, '--', alpha=0.8, linewidth=1,c='green' )

                                x = []
                                y = []
                                z = []
                                for i in range(0, int(rpm23)-10, 10):
                                    x.append(v23[0])
                                    y.append(v23[1])
                                    z.append(i)
                                x.append(v23[0])
                                y.append(v23[1])
                                z.append(rpm23)
                                #ax.plot(x, y, z, '--', alpha=0.8, linewidth=1, c='green')

                                x = []
                                y = []
                                z = []
                                for i in range(0, int(rpm31)-10, 10):
                                    x.append(v31[0])
                                    y.append(v31[1])
                                    z.append(i)
                                x.append(v31[0])
                                y.append(v31[1])
                                z.append(rpm31)
                                #ax.plot(x, y, z, '--', alpha=0.8, linewidth=1, c='green')

                                xlim =np.array(vs)[:,0]
                                ylim = np.array(vs)[:,1]
                                xlim = [V1[0],V2[0],V3[0],xlim[0],xlim[1],xlim[2]]
                                ylim = [V1[1], V2[1], V3[1], ylim[0], ylim[1], ylim[2]]

                                ax.set_xlim(np.min(xlim)-0.3, np.max(xlim)+0.3)
                                ax.set_ylim(np.min(ylim)-0.3, np.max(ylim)+0.3)
                                #ax.set_xlim(11, 19)
                                #ax.set_ylim(11,19)
                                ax.set_zlim(0, 90)

                                #ax.set_xlabel(r'V',fontsize=16)
                                #ax.set_ylabel(r'\bar{V}_N', fontsize=16)
                                #ax.set_zlabel(r'RPM', fontsize=16)

                            plt.show()
                            #list(neighboringTri1rpm[1]).index([k for k in neighboringTri1rpm[1] if k not in  [rpm1,rpm2,rpm3]  ])
                            x=0

                            #Xpred=XpredN
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

                            #Xpred=XpredN


                            #Xpred = (modeler.getBestModelForPoint(candidatePoint.reshape(-1,2)).predict(candidatePoint.reshape(-1,2))\
                            #+(rpm1+rpm2+rpm3)/3)/2
                            #Xpred=(math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) + \
                                        #2 * W1 * W2 *rpm12 + 2 * W2 * W3 * rpm23 + 2 * W1 * W3 * rpm31
                            ################

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
                                #Xpred=XpredN
                                        #Xpred = [ W2 * rpm2 + W1 * rpm1 + W3 * rpm3 ]

                    except Exception as e:
                        print(str(e))

                    break
            '''if point doesn't belong to any triangle of the triangulated space  
                find the triangle that is closer to the point by taking 
                the mean Euclidean dist of all points in adjanceny
                list of each triangle from candidate point.
                When closest triangle to the candidate point is found work the same way as above.
            '''
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
                try:
                    meanPreds = Xpred
                except:
                    d=0
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

        plt.plot(np.linspace(0, len(predictions), len(predictions)), predictions, '-', c='blue',
                 label='RPM Predictions with triangulation, acc: %')

        plt.plot(np.linspace(0, len(yTrue), len(yTrue)), yTrue, '-', c='red', label='Actual RPM')
        # plt.fill_between(np.linspace(0, len(rpmMean), len(rpmMean)), abs(np.array(rpmMean) - 1.434),abs(np.array(rpmMean) + 1.434),color='gray', alpha=0.2)

        plt.ylabel('RPM', fontsize=19)
        plt.xlabel('Observations', fontsize=19)
        # plt.title('Performance comparison of top 3 methods')
        plt.legend()
        plt.grid()
        plt.show()

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
        self.count = self.count + 1
        csvModels=['./trainedModels/model_'+str(modelId)+'_.csv']
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
                        self.interceptsGen = float(row[1])
                        continue

                    if row.__len__() == 0:
                        continue
                    d = row[0]
                    #if self.count == 1:
                    self.intercepts.append(float(row[1]))

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

        count = 0
        for iCnt in range(np.shape(unseenX)[0]):
            pPoint =unseenX[iCnt]
            pPoint= pPoint.reshape(-1,unseenX.shape[1])
            XSplineVector=[]
            trueVal = unseenY[iCnt]
            preds = []
            ind, fit = modeler.getBestPartitionForPoint(pPoint, partitionsX)

            if len(partitionsX)>1:
                #fits = modeler.getFitForEachPartitionForPoint(pPoint, partitionsX)
                for n in range(0, len(partitionsX)):
                    preds.append(modeler._models[n].predict(pPoint)[0][0])

            prediction = modeler._models[len(partitionsX)-1].predict(pPoint)
            preds.append(prediction)
            #trainedWeights =  scores.reshape(len(partitionsX)) if str(scores) !='None' else scores
                #np.where(scores<0, 0, scores).reshape(len(partitionsX))
            #trainedWeights = np.sum(preds*trainedWeights)/trainedWeights
            if len(modeler._models) > 1:
                prediction = scores.predict(np.array(preds).reshape(-1,len(partitionsX)+1))
                    #(np.average(preds, weights=trainedWeights) + prediction) / 2
            else:
                prediction = prediction


            lErrors.append(abs(prediction - trueVal))

        errors = np.asarray(lErrors)

        return errors, np.mean(errors), np.std(lErrors)

    def evaluateKerasNN(self, unseenX, unseenY, modeler,output,xs,genericModel,partitionsX , scores):
        lErrors = []
        self.intercepts = []
        preds=[]
        self.count = 0
        errorStwArr = []
        rpm = []
        errorRpm = []

        for iCnt in range(np.shape(unseenX)[0]):
            pPoint =unseenX[iCnt]
            pPoint= pPoint.reshape(-1,unseenX.shape[1])


            trueVal = unseenY[iCnt]


            #ind, fit = modeler.getBestPartitionForPoint(pPoint, partitionsX)

            #fits = modeler.getFitForEachPartitionForPoint(pPoint, partitionsX)


            if len(modeler._models) > 1:
                preds = []
                for n in range(0,len(partitionsX)):
                    self.intercepts = []
                    vector = self.extractFunctionsFromSplines(pPoint[0][0], pPoint[0][1], n)
                    #XSplineVector = np.append(pPoint, vector)
                    # XSplineVector = XSplineVector.reshape(-1, XSplineVector.shape[ 0 ])

                    #XSplinevectorNew = np.array(self.intercepts) * vector
                    #XSplinevectorNew = np.array([i + self.interceptsGen for i in XSplinevectorNew])

                    XSplinevectorNew = np.append(pPoint, vector)
                    XSplinevectorNew = XSplinevectorNew.reshape(-1, XSplinevectorNew.shape[0])
                    #try:
                    preds.append(modeler._models[n].predict(XSplinevectorNew)[0][0] )
                    #except:
                        #d=0
                # weightedPreds.append(modeler._models[len(modeler._models)-1].predict(pPoint))
                #############################

                #prediction = np.average(preds ,weights=fits )
            #else:

            self.intercepts = []
            vector = self.extractFunctionsFromSplines(pPoint[0][0], pPoint[0][1], 'Gen')

            #XSplineGenvectorNew = np.array(self.intercepts) * vector
            #XSplineGenvectorNew = np.array([i + self.interceptsGen for i in vector])

            #XSplineGenvectorNew= np.sum(np.array(self.intercepts) * vector) + self.interceptsGen
            XSplineGenvectorNew = np.append(pPoint, vector)
            XSplineGenvectorNew = XSplineGenvectorNew.reshape(-1, XSplineGenvectorNew.shape[0])

            prediction = abs(modeler._models[len(modeler._models)-1].predict(XSplineGenvectorNew))
            preds.append(prediction)

            #trainedWeights = genericModel
            if len(modeler._models) > 1:
                prediction = scores.predict(np.array(preds).reshape(-1,len(partitionsX)+1))
                #prediction = (np.average(preds, weights=trainedWeights) )#+ prediction) / 2
            else:
                prediction = prediction

            lErrors.append(abs(prediction - trueVal))

            #lErrors.append(abs(prediction - trueVal))
            percError = abs((prediction - trueVal) / trueVal) * 100
            #errorStwArr.append(np.array(
                #np.append(np.asmatrix(pPoint[0][0]).reshape(-1, 1), np.asmatrix([percError[0]]).T, axis=1)))
            errorRpm.append(percError)
            rpm.append(trueVal)
            preds.append(prediction[0][0])

            error = abs(prediction - trueVal)
            lErrors.append(error)

        errorStwArr = np.array(errorStwArr)
        errorStwArr = errorStwArr.reshape(-1, 2)

        if type == 'train':
            with open('./TRAINerrorPercFOC' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv',
                      mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['FOC', 'PERC'])
                for i in range(0, len(errorFoc)):
                    data_writer.writerow(
                        [foc[i], errorFoc[i][0][0]])

            with open('./TRAINerrorSTW' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv', mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['STW', 'MAE'])
                for i in range(0, len(errorStwArr)):
                    data_writer.writerow(
                        [errorStwArr[i][0], errorStwArr[i][1]])
        else:
            with open('./TESTerrorPercRPMxiNN' + str(len(partitionsX)) + '_' + str(0) + '.csv',
                      mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['FOC', 'PRED','PERC'])
                for i in range(0, len(errorRpm)):
                    data_writer.writerow([])
                        #[rpm[i],preds[i] ,errorRpm[i][0][0]])


        #prediction =modeler._models[ 0 ].predict(unseenX.reshape(2,2860))
        #print np.mean(abs(prediction - unseenY))
        #print("EXCEPTIONS :  "+str(count))
        print('Accuracy: ' + str(np.round(100 - np.mean(errorRpm), 2)))
        errors = np.asarray(lErrors)

        return errors, np.mean(errors), np.std(lErrors)

    def evaluateKerasNNLSTM(self, unseenX, unseenY, modeler,output,xs,genericModel,partitionsX , scores):
        lErrors = []
        self.intercepts = []
        self.count = 0
        errorStwArr = []
        rpm = []
        errorRpm = []
        preds=[]
        subsetInd=0
        type='test'
        XSplineGenvectorNews=[]
        for iCnt in range(np.shape(unseenX)[0]):
            pPoint =unseenX[iCnt]
            pPoint= pPoint.reshape(-1,unseenX.shape[1])

            #try:
            trueVal = unseenY[iCnt]


            self.intercepts = []
            vector = self.extractFunctionsFromSplines(pPoint[ 0 ][ 0 ], pPoint[ 0 ][ 1 ], 'Gen')


            #XSplineGenvectorNew = np.array(self.intercepts) * vector
            #XSplineGenvectorNew = np.array([i + self.interceptsGen for i in XSplineGenvectorNew])

            #XSplineGenvectorNew = np.append(pPoint, vector)
            #XSplineGenvectorNews.append(XSplineGenvectorNew)

            splineVector = np.append(pPoint, vector)
            XSplineGenvectorNews.append(np.append(splineVector, unseenY[iCnt]))
            #XSplineGenvectorNew = XSplineGenvectorNew.reshape(-1, XSplineGenvectorNew.shape[0])

        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(sequence) - 1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix][:,0:sequence.shape[1]-1], sequence[i][sequence.shape[1]-1]
                X.append(seq_x)
                y.append(seq_y)
            return array(X), array(y)

            # define input sequence

        raw_seq = np.array(XSplineGenvectorNews)

        n_steps = 5
        # split into samples
        unseenXlstm, unseenYlstm = split_sequence(raw_seq, n_steps)

        for iCnt in range(np.shape(unseenXlstm)[0]):
            pPoint =unseenXlstm[iCnt]
            #pPoint= pPoint.reshape(-1,unseenX.shape[1])



            #try:
            trueVal = unseenYlstm[iCnt]

            #ind , fit = modeler.getBestPartitionForPoint(pPoint,partitionsX)

            '''ind, fit = modeler.getBestPartitionForPoint(pPoint, partitionsX)

            if len(modeler._models) > 1:
                self.intercepts = []
                vector = self.extractFunctionsFromSplines(pPoint[0][0],pPoint[0][1],ind)
                XSplineVector=np.append(pPoint, vector)
                XSplineVector = XSplineVector.reshape(-1, XSplineVector.shape[ 0 ])

                XSplinevectorNew = np.array(self.intercepts) * vector
                XSplinevectorNew = np.array([i + self.interceptsGen for i in XSplinevectorNew])

                XSplinevectorNew = np.append(pPoint, XSplinevectorNew)
                XSplinevectorNew = XSplinevectorNew.reshape(-1, XSplinevectorNew.shape[0])
                XSplinevectorNew = np.reshape(XSplinevectorNew, (XSplinevectorNew.shape[0], XSplinevectorNew.shape[1], 1))
                #######################################################
                prediction = abs(modeler._models[ind].predict(XSplinevectorNew))
            else:
                self.intercepts = []
                vector = self.extractFunctionsFromSplines(pPoint[ 0 ][ 0 ], pPoint[ 0 ][ 1 ], 'Gen')


                XSplineGenvectorNew = np.array(self.intercepts) * vector
                XSplineGenvectorNew = np.array([i + self.interceptsGen for i in XSplineGenvectorNew])

                XSplineGenvectorNew = np.append(pPoint, XSplineGenvectorNew)
                XSplineGenvectorNew = XSplineGenvectorNew.reshape(-1, XSplineGenvectorNew.shape[0])
                XSplineGenvectorNew = np.reshape(XSplineGenvectorNew, (XSplineGenvectorNew.shape[0], XSplineGenvectorNew.shape[1], 1))'''

            pPoint = np.reshape(pPoint, (1, pPoint.shape[0], pPoint.shape[1]))
            prediction = modeler._models[len(modeler._models) - 1].predict(pPoint)

            '''if len(modeler._models)>1:
                prediction = (abs(modeler._models[ind].predict(XSplinevectorNew)) + modeler._models[len(modeler._models) - 1].predict(XSplineGenvectorNew) )/ 2
            else:
                prediction =  modeler._models[len(modeler._models) - 1].predict(XSplinevectorNew)'''

            percError = abs((prediction - trueVal) / trueVal) * 100
            # errorStwArr.append(np.array(
            # np.append(np.asmatrix(pPoint[0][0]).reshape(-1, 1), np.asmatrix([percError[0]]).T, axis=1)))
            errorRpm.append(percError)
            rpm.append(trueVal)
            preds.append(prediction[0][0])

            error = abs(prediction - trueVal)
            lErrors.append(error)

        errorStwArr = np.array(errorStwArr)
        errorStwArr = errorStwArr.reshape(-1, 2)

        if type == 'train':
            with open('./TRAINerrorPercFOC' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv',
                      mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['FOC', 'PERC'])
                for i in range(0, len(errorRpm)):
                    data_writer.writerow(
                        [rpm[i], errorRpm[i][0][0]])

            with open('./TRAINerrorSTW' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv', mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['STW', 'MAE'])
                for i in range(0, len(errorStwArr)):
                    data_writer.writerow(
                        [errorStwArr[i][0], errorStwArr[i][1]])
        else:
            with open('./TESTerrorPercRPMxiLSTM' + str(len(partitionsX)) + '_' + str(0) + '.csv',
                      mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['FOC', 'PRED', 'PERC'])
                for i in range(0, len(errorRpm)):
                    data_writer.writerow(
                        [rpm[i], preds[i], errorRpm[i][0][0]])



        # prediction =modeler._models[ 0 ].predict(unseenX.reshape(2,2860))
        # print np.mean(abs(prediction - unseenY))
        # print("EXCEPTIONS :  "+str(count))
        print('Accuracy: ' + str(np.round(100 - np.mean(errorRpm), 2)))
        errors = np.asarray(lErrors)
        return errors, np.mean(errors), np.std(lErrors)

    def evaluateKerasNNNweights(self, unseenX, unseenY, modeler,output,xs,genericModel,partitionsX , scores):
        lErrors = []
        #self.intercepts = []
        self.count = 0

        count = 0
        for iCnt in range(np.shape(unseenX)[0]):

            self.intercepts=[]
            pPoint =unseenX[iCnt]
            pPoint= pPoint.reshape(-1,unseenX.shape[1])



            #try:
            trueVal = unseenY[iCnt]

            ind, fit = modeler.getBestPartitionForPoint(pPoint, partitionsX)

            if len(modeler._models) > 1:
                vector = self.extractFunctionsFromSplines(pPoint[0][0], pPoint[0][1], ind)
                weightsCl = abs(modeler._models[ind].predict(pPoint)[0])
                XSplinevectorNew = np.array(self.intercepts) * vector

                XSplinevectorNew = np.array([i + self.interceptsGen for i in XSplinevectorNew])
                XSplinevectorNew = XSplinevectorNew if len(XSplinevectorNew) > 0 else [self.interceptsGen]


            else:

                vector = self.extractFunctionsFromSplines(pPoint[0][0], pPoint[0][1], 'Gen')
                weightsCl = abs(modeler._models[len(modeler._models) - 1].predict(pPoint)[0])
                #XSplinevectorNew = np.array(self.intercepts) * vector
                #XSplinevectorNew = np.array([i + self.interceptsGen for i in XSplinevectorNew])

                XSplinevectorNew = np.array(self.intercepts) *  vector
                XSplinevectorNew = np.array([i + self.interceptsGen  for i in XSplinevectorNew])
                #weightsCl = [weightsCl[w] if XSplinevectorNew[w]>0 else 0 for w in range(0,len(weightsCl))]

            #

            prediction = np.average(XSplinevectorNew, weights=weightsCl)



                    #np.average((np.array(self.intercepts) * [i for i in vector]),weights=(modeler._models[ind].predict(pPoint)[0]))+self.interceptsGen

            #prediction = [XSplinevectorNew[k] * weightsCl[k] for k in range(0,len(XSplinevectorNew))]
            #prediction = np.sum(prediction)
            #prediction = XSplinevectorNew[np.argmax(weightsCl)]
            '''if len(modeler._models)>1:
                prediction = (abs(modeler._models[ind].predict(pPoint)) + modeler._models[len(modeler._models) - 1].predict(pPoint) )/ 2
            else:
                prediction =  modeler._models[len(modeler._models) - 1].predict(pPoint)'''

            ##########

            if abs(prediction - trueVal)>10:
                w=0
        #x=unseenX[:,0].reshape(-1,unseenX.shape[0])
            lErrors.append(abs(prediction - trueVal))

        errors = np.asarray(lErrors)

        return errors, np.mean(errors), np.std(lErrors)

    def ANOVAtest(self,clusters,var,trError,error,models,partitioners):

        # 'Trerror': trError,
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
        #print(stats.kruskal(error,clusters,var,models,partitioners))
        #print(self.kw_dunn(groups,[(0,1),(0,2)]))

        #dataf = pd.DataFrame.from_dict(df, orient='index')
        df.to_csv('./experimentalResults/NEWres_1.csv', index=False)
        #df.melt(var_name='groups', value_name='values')
        df=pd.melt(df,var_name='groups', value_name='values')
        print(df)
        #df2=sp.posthoc_dunn(df, val_col='values', group_col='groups')
        #df2.to_csv('./kruskal.csv', index=False)
        #print(df2)


        #formula = 'error ~ C(clusters) + C(var) + C(error)'
        ##model =  ols(formula, df).fit()
        #aov_tab

    def evaluateKerasNNLSTM2(self, unseenX, unseenY, modeler, output, xs, genericModel, partitionsX, scores):
        lErrors = []
        self.intercepts = []
        self.count = 0
        rpm=[]
        errorRpm=[]
        preds=[]
        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(sequence) - 1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(unseenY[i])
            return array(X), array(y)

            # define input sequence

        raw_seq = np.array(unseenX)

        n_steps = 5
        # split into samples
        unseenXlstm, unseenYlstm = split_sequence(raw_seq, n_steps)

        for iCnt in range(np.shape(unseenXlstm)[0]):
            pPoint =unseenXlstm[iCnt]
            pPoint = np.reshape(pPoint, (1, pPoint.shape[0], pPoint.shape[1]))
            #pPoint = pPoint.reshape(-1, unseenX.shape[1])

            # try:
            trueVal = unseenY[iCnt]

            ind, fit = modeler.getBestPartitionForPoint(pPoint, partitionsX)

            pPoint = np.reshape(pPoint, (pPoint.shape[0], pPoint.shape[1], 1))
            if len(modeler._models) > 1:
                prediction = (abs(modeler._models[ind].predict(pPoint)) + modeler._models[
                    len(modeler._models) - 1].predict(pPoint)) / 2
            else:
                prediction = modeler._models[len(modeler._models) - 1].predict(pPoint)


            lErrors.append(abs(prediction - trueVal))

            percError = abs((prediction - trueVal) / trueVal) * 100
            # errorStwArr.append(np.array(
            # np.append(np.asmatrix(pPoint[0][0]).reshape(-1, 1), np.asmatrix([percError[0]]).T, axis=1)))
            errorRpm.append(percError)
            rpm.append(trueVal)
            preds.append(prediction[0][0])

            error = abs(prediction - trueVal)
            lErrors.append(error)

        errorStwArr = np.array([])
        errorStwArr = errorStwArr.reshape(-1, 2)

        if type == 'train':
            with open('./TRAINerrorPercFOC' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv',
                      mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['FOC', 'PERC'])
                for i in range(0, len(errorFoc)):
                    data_writer.writerow(
                        [foc[i], errorFoc[i][0][0]])

            with open('./TRAINerrorSTW' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv', mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['STW', 'MAE'])
                for i in range(0, len(errorStwArr)):
                    data_writer.writerow(
                        [errorStwArr[i][0], errorStwArr[i][1]])
        else:
            with open('./TESTerrorPercRPMLSTM' + str(len(partitionsX)) + '_' + str(0) + '.csv',
                      mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['FOC', 'PRED', 'PERC'])
                for i in range(0, len(errorRpm)):
                    data_writer.writerow(
                        [rpm[i], preds[i], errorRpm[i][0][0]])



        # prediction =modeler._models[ 0 ].predict(unseenX.reshape(2,2860))
        # print np.mean(abs(prediction - unseenY))
        # print("EXCEPTIONS :  "+str(count))
        print('Accuracy: ' + str(np.round(100 - np.mean(errorRpm), 2)))


        errors = np.asarray(lErrors)

        return errors, np.mean(errors), np.std(lErrors)