import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
#######################
for s in neighboringTri:
    V1n = dataXnew[ s ][ 0 ]
    V2n = dataXnew[ s ][ 1 ]
    V3n = dataXnew[ s ][ 2 ]

    rpm1n = dataYnew[ s ][ 0 ]
    rpm2n = dataYnew[ s ][ 1 ]
    rpm3n = dataYnew[ s ][ 2 ]

    nRpms.append(rpm1n - B1)
    nRpms.append(rpm2n - B1)
    nRpms.append(rpm3n - B1)
    ###barycentric coords of neighboring points in relation to initial triangle
    eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                     [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

    eq2 = np.array([ V1n[ 0 ] - V3[ 0 ], V1n[ 1 ] - V3[ 1 ] ])
    solutions = np.linalg.solve(eq1, eq2)

    W1n = solutions[ 0 ]
    W2n = solutions[ 1 ]
    W3n = 1 - solutions[ 0 ] - solutions[ 1 ]

    nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))

    eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                     [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

    eq2 = np.array([ V2n[ 0 ] - V3[ 0 ], V2n[ 1 ] - V3[ 1 ] ])
    solutions = np.linalg.solve(eq1, eq2)

    W1n = solutions[ 0 ]
    W2n = solutions[ 1 ]
    W3n = 1 - solutions[ 0 ] - solutions[ 1 ]

    nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))

    eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                     [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

    eq2 = np.array([ V3n[ 0 ] - V3[ 0 ], V3n[ 1 ] - V3[ 1 ] ])
    solutions = np.linalg.solve(eq1, eq2)

    W1n = solutions[ 0 ]
    W2n = solutions[ 1 ]
    W3n = 1 - solutions[ 0 ] - solutions[ 1 ]

    nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))
    # ki = np.array([ V1n, V2n,
    # V3n ])
    # t5 = plt.Polygon(ki, fill=False, color='yellow', linewidth=3)
    # plt.gca().add_patch(t5)

##############################
# plt.plot(candidatePoint[0],candidatePoint[1], 'o',color='green' ,markersize=10)
# plt.plot(V2[ 0 ], V2[ 1 ], 'o', color='green', markersize=10)
# plt.plot(V3[ 0 ], V3[ 1 ], 'o', color='green', markersize=10)
# plt.show()
##############BArycentric coords.
# eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
# [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

# eq2 = np.array([ candidatePoint[ 0 ] - V3[ 0 ], candidatePoint[ 1 ] - V3[ 1 ] ])
# solutions = np.linalg.solve(eq1, eq2)

# W1 = solutions[ 0 ]
# W2 = solutions[ 1 ]
# W3 = 1 - solutions[ 0 ] - solutions[ 1 ]

# W1 = (V2[ 1 ] - V3[ 1 ]) * (candidatePoint[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (
# candidatePoint[ 1 ] - V3[ 1 ]) \
# / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

# W2 = (V3[ 1 ] - V1[ 1 ]) * (candidatePoint[ 0 ] - V3[ 0 ]) + (V1[ 0 ] - V3[ 0 ]) * (
# candidatePoint[ 1 ] - V3[ 1 ]) \
# / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

# W3 = 1 - W1 - W2
##################################
#########   false flag


# index,bestFit=modeler.getBestPartitionForPoint(candidatePoint)


#############################################################
##############################################################
# simplicesIndexes = triNew.find_simplex(trainX[index])
# dataXnew=dataX[triNew.vertices[simplicesIndexes]]
# dataYnew=dataY[triNew.vertices[simplicesIndexes]]

# dataXnew = np.concatenate(dataX[ tri.vertices[ simplicesIndexes ] ])
# dataYnew = np.concatenate(dataY[ tri.vertices[ simplicesIndexes ] ])
# if len(dataX)>=3:
# try:
# triNew = Delaunay(dataXnew)
# except:
# print ("")
# else:
# lErrors.append(np.mean(dataYnew) - trueVal)

# plt.plot(candidatePoint[0],candidatePoint[1], 'o',color='green' ,markersize=8)
# plt.plot(dataXnew[ 0:, 0 ], dataXnew[ 0:, 1 ], 'o', markersize=8)

# for v in vertices:
# k = np.array([ [ v[ 0 ][ 0 ], v[ 0 ][ 2 ] ], [ v[ 1 ][ 0 ], v[ 1 ][ 2 ] ], [ v[ 2 ][ 0 ], v[ 2 ][ 2 ] ] ])
# t = plt.Polygon(k, fill=False,color='red',linewidth=3)
# plt.gca().add_patch(t)
# plt.show()
###########################
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
    if varPerTri > 30:
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
    if varPerTri > 30:
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
                    delIndexes.append(k)
                # dataXnew = np.delete(dataXnew,i,axis=0) if (dataXnew[ i ] == V1).all()==True else dataXnew
                # dataYnew = np.delete(dataYnew, i) if dataYnew[ i ] == rpm1 and (dataXnew[ i ] == V1).all()==True else dataYnew

                delListX.append(V1)
                delListY.append(rpm1)
                if V2 not in np.array(dataXnewFiltered):
                    dataXnewFiltered.append(V2)
                    dataYnewFiltered.append(rpm2)
                if V3 not in np.array(dataXnewFiltered):
                    dataXnewFiltered.append(V3)
                    dataYnewFiltered.append(rpm3)
            elif outlier_ind == 1:
                if (dataXnew[ i ] == V2).all() == True:
                    delIndexes.append(k)
                # dataXnew = np.delete(dataXnew, i, axis=0) if (dataXnew[ i ] == V2).all()==True else dataXnew
                # dataYnew = np.delete(dataYnew, i) if dataYnew[ i ] == rpm2 and (dataXnew[ i ] == V2).all()==True else dataYnew

                delListX.append(V2)
                delListY.append(rpm2)
                if V1 not in np.array(dataXnewFiltered):
                    dataXnewFiltered.append(V1)
                    dataYnewFiltered.append(rpm1)
                if V3 not in np.array(dataXnewFiltered):
                    dataXnewFiltered.append(V3)
                    dataYnewFiltered.append(rpm3)
            else:
                if (dataXnew[ i ] == V3).all() == True:
                    delIndexes.append(k)
                # dataXnew = np.delete(dataXnew, i, axis=0) if (dataXnew[ i ] == V3).all()==True else dataXnew
                # dataYnew = np.delete(dataYnew, i) if dataYnew[ i ] == rpm3 and  (dataXnew[ i ] == V3).all()==True else dataYnew

                delListX.append(V3)
                delListY.append(rpm3)
                if V1 not in np.array(dataXnewFiltered):
                    dataXnewFiltered.append(V1)
                    dataYnewFiltered.append(rpm1)
                if V2 not in np.array(dataXnewFiltered):
                    dataXnewFiltered.append(V2)
                    dataYnewFiltered.append(rpm2)
    else:
        if V1 not in np.array(dataXnewFiltered):
            dataXnewFiltered.append(V1)
            dataYnewFiltered.append(rpm1)
        if V2 not in np.array(dataXnewFiltered):
            dataXnewFiltered.append(V2)
            dataYnewFiltered.append(rpm2)
        if V3 not in np.array(dataXnewFiltered):
            dataXnewFiltered.append(V3)
            dataYnewFiltered.append(rpm3)
        largeVarVert.append(triNew.vertices[ k ])

############################
neighboringVertices1 = [ ]
for u in range(0, 2):
    try:
        neighboringVertices1.append(triNew.vertex_neighbor_vertices[ 1 ][
                                    triNew.vertex_neighbor_vertices[ 0 ][ triNew.vertices[ k ][ u ] ]:
                                    triNew.vertex_neighbor_vertices[ 0 ][ triNew.vertices[ k ][ u ] + 1 ] ])
    except:
        break
neighboringTri = triNew.vertices[ triNew.find_simplex(dataXnew[ np.concatenate(np.array(neighboringVertices1)) ]) ]

##find distance of pointEstimate from the three vertexes
V1 = dataXnew[ triNew.vertices[ k ] ][ 0 ]
V2 = dataXnew[ triNew.vertices[ k ] ][ 1 ]
V3 = dataXnew[ triNew.vertices[ k ] ][ 2 ]

distV1init = math.sqrt(
    math.pow(candidatePoint[ 0 ] - V1[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V1[ 1 ], 2))
distV2init = math.sqrt(
    math.pow(candidatePoint[ 0 ] - V2[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V2[ 1 ], 2))
distV3init = math.sqrt(
    math.pow(candidatePoint[ 0 ] - V3[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V3[ 1 ], 2))
###find 3 neighboring triangles of candidate point P
ind = list((distV1init, distV2init, distV3init)).index(np.min(np.array([ distV1init, distV2init, distV3init ])))

distList = {"data": [ ]}
count = 0
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

    # distList.append(np.mean([ distV1, distV2, distV3 ]))

    distance = {}
    distance[ "dist" ] = np.mean([ distV1, distV2, distV3 ])
    distance[ "index" ] = count
    distList[ "data" ].append(distance)
    count += 1

distList.values()[ 0 ].sort()
for v in distList[ "data" ]:
    neighboringIndexes = np.array([ distList[ "data" ][ 0 ][ "index" ],
                                    distList[ "data" ][ 1 ][ "index" ],
                                    distList[ "data" ][ 2 ][ "index" ] ])
# k = minIndexDist
neighboringVertices1 = [ ]
# simplex = triNew.vertices[ triNew.find_simplex(
# [ dataXnew[ triNew.vertices[ k ] ][ 0 ], dataXnew[ triNew.vertices[ k ] ][ 1 ],
# dataXnew[ triNew.vertices[ k ] ][ 2 ] ]) ]
for s in neighboringTri:
    V1 = dataXnew[ s ][ 0 ]
    V2 = dataXnew[ s ][ 1 ]
    V3 = dataXnew[ s ][ 2 ]

    rpm1s = dataYnew[ s ][ 0 ]
    rpm2s = dataYnew[ s ][ 1 ]
    rpm3s = dataYnew[ s ][ 2 ]

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

    prediction = (W1 * rpm1 + W2 * rpm2 + W3 * rpm3) / (W1 + W2 + W3)
    pred1S = (W1x * rpm1 + W2x * rpm2 + W3x * rpm3) / (W1x + W2x + W3x)
    preds.append(prediction)
#################################################################################
###########################################################
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

W1 = (V2[ 1 ] - V3[ 1 ]) * (candidatePoint[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (candidatePoint[ 1 ] - V3[ 1 ]) \
     / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

W2 = (V3[ 1 ] - V1[ 1 ]) * (candidatePoint[ 0 ] - V3[ 0 ]) + (V1[ 0 ] - V3[ 0 ]) * (
        candidatePoint[ 1 ] - V3[ 1 ]) \
     / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

W3 = 1 - W1 - W2

prediction = (W1 * rpm1 + W2 * rpm2 + W3 * rpm3) / (W1 + W2 + W3)
pred1 = (W1x * rpm1 + W2x * rpm2 + W3x * rpm3) / (W1x + W2x + W3x)
# preds.append(pred1)
preds.append(prediction)
spline1 = sp.Earth()
spline2 = sp.Earth()
spline3 = sp.Earth()

sp1 = spline1.fit(np.array([ V1 ]).reshape(1, -1), [ rpm1 ])
sp2 = spline2.fit(np.array([ V2 ]).reshape(1, -1), [ rpm2 ])
sp3 = spline3.fit(np.array([ V3 ]).reshape(1, -1), [ rpm3 ])

coef1 = sp1.coef_[ 0 ][ 0 ]
coef2 = sp2.coef_[ 0 ][ 0 ]
coef3 = sp3.coef_[ 0 ][ 0 ]

# (coef2*coef3*2*W2*W3 +coef3*coef1 *2* W1*W3 + coef1*coef2*2*W1*W2 +
rpmsd = [ ]
for n in neighboringTri[ neighboringIndexes ]:
    V1d = dataXnew[ n ][ 0 ]
    V1d = dataXnew[ n ][ 0 ]
    V2d = dataXnew[ n ][ 1 ]
    V3d = dataXnew[ n ][ 2 ]

    rpm1d = dataYnew[ n ][ 0 ]
    rpm2d = dataYnew[ n ][ 1 ]
    rpm3d = dataYnew[ n ][ 2 ]

    gamma1 = rpm1
    gamma2 = rpm2
    gamma3 = rpm3

    rpmsd.append([ rpm1d, rpm2d, rpm3d ])

################################################
for x in dataXnew:
    if listX == [ ]:
        listX.append(x)
    else:
        flag = False
        for l in listX:
            if spatial.distance.euclidean(l, x) < 0.4:
                flag = True
                break
        if flag == False:
            listX.append(x)

#######################################
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

    prediction = (W1 * rpm1 + W2 * rpm2 + W3 * rpm3) / (W1 + W2 + W3)
    pred1S = (W1x * rpm1 + W2x * rpm2 + W3x * rpm3) / (W1x + W2x + W3x)
    preds.append(pred1S)

#####################################
## 3 efaptomena trignwa found


###find outlier between neighboring pojnts of initial triangle
# if abs(np.mean([rpm1,rpm2,rpm3])-rpm1) > 10:
# rpm1 = rpm2
# Xpred = [ (rpm1 + rpm2 + rpm3) / 3 ]
# elif abs(np.mean([rpm1,rpm2,rpm3])-rpm2) > 10:
# rpm2=rpm1
# Xpred = [ (rpm1 + rpm2 + rpm3) / 3 ]
# elif abs(np.mean([ rpm1, rpm2, rpm3 ]) - rpm3) > 10:
# rpm3 =rpm1
# Xpred = [ (rpm1 + rpm2 + rpm3) / 3 ]

# if rpm1 -(rpm3+rpm2)/2<-20 or rpm2 - (rpm1+rpm3)/2<-20 or rpm3 - (rpm1+rpm2)/2<-20:
# if abs(rpm1-rpm2) > abs(rpm1-rpm3):
# rpm2=rpm3
# elif abs(rpm2-rpm1) > abs(rpm2-rpm3):
# rpm1=rpm3
# elif abs(rpm1-rpm3) > abs(rpm2-rpm1):
# rpm3=rpm1
# Xpred = [ (rpm1 + rpm2 + rpm3) / 3 ]

# recalculate barycentric coords.
# else:
##################################
#########################################
for simplex in hull.simplices:
    plt.plot(dataXnew[ simplex, 0 ], dataXnew[ simplex, 1 ], 'k-', linewidth=2)

plt.plot(candidatePoint[ 0 ], candidatePoint[ 1 ], 'o', markersize=8)
plt.plot(v12[ 0 ], v12[ 1 ], 'o', markersize=8)
plt.plot(v23[ 0 ], v23[ 1 ], 'o', markersize=8)
plt.plot(v31[ 0 ], v31[ 1 ], 'o', markersize=8)
ki = np.array([ dataXnew[ tri11[ 0 ] ][ 0 ], dataXnew[ tri11[ 0 ] ][ 1 ],
                dataXnew[ tri11[ 0 ] ][ 2 ] ])
t1 = plt.Polygon(ki, fill=False, color='red', linewidth=3)
plt.gca().add_patch(t1)

ki = np.array([ dataXnew[ tri22[ 0 ] ][ 0 ], dataXnew[ tri22[ 0 ] ][ 1 ],
                dataXnew[ tri22[ 0 ] ][ 2 ] ])
t2 = plt.Polygon(ki, fill=False, color='green', linewidth=3)
plt.gca().add_patch(t2)

ki = np.array([ dataXnew[ tri33[ 0 ] ][ 0 ], dataXnew[ tri33[ 0 ] ][ 1 ],
                dataXnew[ tri33[ 0 ] ][ 2 ] ])
t3 = plt.Polygon(ki, fill=False, color='black', linewidth=3)
plt.gca().add_patch(t3)

ki = np.array([ V1, V2,
                V3 ])
t4 = plt.Polygon(ki, fill=False, color='blue', linewidth=3)
plt.gca().add_patch(t4)
# plt.show()
print("W1 for red triangle:" + str(W12_1))
print("W2 for red triangle:" + str(W12_2))
print("W3 for red triangle:" + str(W12_3))

###################################################
from sklearn.cluster import DBSCAN

_dataInit = np.array([ rpm1, rpm2, rpm3 ]).reshape(-1, 1)

outlier_detection = DBSCAN(min_samples=1, eps=3)
clustersInit = outlier_detection.fit_predict(_dataInit)
list(clustersInit).count(-1)
partitionsXInit = [ ]
for curLbl in np.unique(clustersInit):
    # Create a partition for X using records with corresponding label equal to the current
    partitionsXInit.append(np.asarray(_dataInit[ clustersInit == curLbl ]))

_dataNeigh = np.array([ rpm12, rpm23, rpm31 ]).reshape(-1, 1)

outlier_detection = DBSCAN(min_samples=1, eps=3)
clustersNeigh = outlier_detection.fit_predict(_dataNeigh)
list(clustersNeigh).count(-1)
partitionsXNeigh = [ ]
for curLbl in np.unique(clustersNeigh):
    # Create a partition for X using records with corresponding label equal to the current
    partitionsXNeigh.append(np.asarray(_dataNeigh[ clustersNeigh == curLbl ]))

############################
distV1n = math.sqrt(
    math.pow(candidatePoint[ 0 ] - v12[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - v12[ 1 ], 2))
distV2n = math.sqrt(
    math.pow(candidatePoint[ 0 ] - v23[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - v23[ 1 ], 2))
distV3n = math.sqrt(
    math.pow(candidatePoint[ 0 ] - v31[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - v31[ 1 ], 2))

W1n = 1 / distV1 if distV1 != 0 else 0
W2n = 1 / distV2 if distV2 != 0 else 0
W3n = 1 / distV3 if distV3 != 0 else 0
##################3

XpredInit = [ (W1x * initrpm1 + W2x * initrpm2 + W3x * initrpm3) / (W1x + W2x + W3x) ]
XpredNeigh = [ (W1n * rpm12 + W2n * rpm23 + W3n * rpm31) / (W1n + W2n + W3n) ]
if flgV == False:
    if len(partitionsXInit) > 1 and len(partitionsXNeigh) == 1:
        Xpred = XpredNeigh
    elif len(partitionsXInit) == 1 and len(partitionsXNeigh) > 1:
        Xpred = XpredInit
    elif len(partitionsXInit) > 1 and len(partitionsXNeigh) > 1:
        Xpred = (XpredInit[ 0 ] + XpredNeigh[ 0 ]) / 2
    else:
        Xpred = [ (rpm1 + rpm2 + rpm3) / 3 ]
else:
    Xpred = [ (rpm1 + rpm2 + rpm3) / 3 ]

###################################
if len(partitionsXInit) > 1:
    x = 0
rpm31 = rpm3 if rpm31[ 0 ] == 0 else rpm31
if abs(np.mean([ rpm1, rpm2, rpm3 ]) - rpm1) > 10:
    rpm1 = rpm2
    Xpred = [ (rpm1 + rpm2 + rpm3) / 3 ]
elif abs(np.mean([ rpm1, rpm2, rpm3 ]) - rpm2):
    rpm2 = rpm1
    Xpred = [ (rpm1 + rpm2 + rpm3) / 3 ]
elif abs(np.mean([ rpm1, rpm2, rpm3 ]) - rpm3) > 10:
    rpm3 = rpm1
    Xpred = [ (rpm1 + rpm2 + rpm3) / 3 ]
#########################################################
elif abs(np.mean([ rpm12, rpm23, rpm31 ]) - rpm12) > 10:
    rpm12 = rpm23
    Xpred = [ (rpm12 + rpm23 + rpm31) / 3 ]

elif abs(np.mean([ rpm12, rpm23, rpm31 ]) - rpm23) > 10:
    rpm23 = rpm12
    Xpred = [ (rpm12 + rpm2 + rpm3) / 3 ]
elif abs(np.mean([ rpm12, rpm23, rpm31 ]) - rpm31) > 10:
    Xpred = [ (rpm12 + rpm23 + rpm31) / 3 ]
else:
    Xpred = [ (rpm1 + rpm2 + rpm3) / 3 ]


##########################
W1 = (V2[ 1 ] - V3[ 1 ]) * (candidatePoint[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (
        candidatePoint[ 1 ] - V3[ 1 ]) \
     / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

W2 = (V3[ 1 ] - V1[ 1 ]) * (candidatePoint[ 0 ] - V3[ 0 ]) + (V1[ 0 ] - V3[ 0 ]) * (
        candidatePoint[ 1 ] - V3[ 1 ]) \
     / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

W3 = 1 - W1 - W2
###############################################

##1st
                        W12_1 =  (V2[1]- V3[1])*(v12[0]-V3[0]) + (V3[0]-V2[0])*(v12[1] - V3[1])\
                        /(V2[1]-V3[1])*(V1[0] - V3[0]) + (V3[0]-V2[0])*(V1[1]-V3[1])

                        W12_2 = (V3[ 1 ] - V1[ 1 ]) * (v12[ 0 ] - V3[ 0 ]) + (V1[ 0 ] - V3[ 0 ]) * (
                                v12[ 1 ] - V3[ 1 ]) \
                             / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

                        W12_3 = 1 - W12_1 - W12_2

                        ##2nd
                        W23_1 = (V2[ 1 ] - V3[ 1 ]) * (v23[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (
                                v23[ 1 ] - V3[ 1 ]) \
                             / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

                        W23_2 = (V3[ 1 ] - V1[ 1 ]) * (v23[ 0 ] - V3[ 0 ]) + (V1[ 0 ] - V3[ 0 ]) * (
                                v23[ 1 ] - V3[ 1 ]) \
                             / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

                        W23_3 = 1 - W23_1 - W23_2

                        ##3rd
                        W31_1 =(V2[ 1 ] - V3[ 1 ]) * (v31[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (
                                v31[ 1 ] - V3[ 1 ]) \
                             / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

                        W31_2 = (V3[ 1 ] - V1[ 1 ]) * (v31[ 0 ] - V3[ 0 ]) + (V1[ 0 ] - V3[ 0 ]) * (
                                v31[ 1 ] - V3[ 1 ]) \
                             / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

                        W31_3 = 1 - W31_1 - W31_2
################################################
W1b = solutions[ 0 ]
W2b = solutions[ 1 ]
W3b = 1 - solutions[ 0 ] - solutions[ 1 ]

eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                 [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

eq2 = np.array([ V2[ 0 ] - V3[ 0 ], V2[ 1 ] - V3[ 1 ] ])
solutions = np.linalg.solve(eq1, eq2)

W1b2 = solutions[ 0 ]
W2b2 = solutions[ 1 ]
W3b2 = 1 - solutions[ 0 ] - solutions[ 1 ]

eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                 [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

eq2 = np.array([ V3[ 0 ] - V3[ 0 ], V3[ 1 ] - V3[ 1 ] ])
solutions = np.linalg.solve(eq1, eq2)

W1b3 = solutions[ 0 ]
W2b3 = solutions[ 1 ]
W3b3 = 1 - solutions[ 0 ] - solutions[ 1 ]
############################################
plt.plot(candidatePoint[0], candidatePoint[1], 'o', markersize=8)
                    ki = np.array([  dataXnew[tri22[0]][0] , dataXnew[tri22[0]][1],
                                     dataXnew[ tri22[ 0 ] ][ 2 ]])
                    t1 = plt.Polygon(ki, fill=False, color='red', linewidth=3)
                    plt.gca().add_patch(t1)

                    ki = np.array([ dataXnew[tri33[0]][0], dataXnew[tri33[0]][1],
                                    dataXnew[ tri33[ 0 ] ][ 2 ] ])
                    t2 = plt.Polygon(ki, fill=False, color='green', linewidth=3)
                    plt.gca().add_patch(t2)

                    ki = np.array([ dataXnew[tri33[1]][0], dataXnew[tri33[1]][1],
                                    dataXnew[ tri33[ 1 ] ][ 2 ] ])
                    t3 = plt.Polygon(ki, fill=False, color='black', linewidth=3)
                    plt.gca().add_patch(t3)

                    ki = np.array([ V1, V2,
                                   V3 ])
                    t4 = plt.Polygon(ki, fill=False, color='blue', linewidth=3)
                    plt.gca().add_patch(t4)
                    plt.show()

##################################################################
self.intersect(tpm1,tpm2,V1,V2) ==False and\
                                self.intersect(tpm2, tpm3, V1, V2) == False and\
                                self.intersect(tpm3, tpm1, V1, V2) == False and\
                                self.intersect(tpm1, tpm2, V2, V3) == False and\
                                self.intersect(tpm2, tpm3, V2, V3) == False and\
                                self.intersect(tpm3, tpm1, V2, V3) == False and\
                                self.intersect(tpm1, tpm2, V1, V3) == False and\
                                self.intersect(tpm2, tpm3, V1, V3) == False and\
                                self.intersect(tpm3, tpm1, V1, V3) == False and\
##################################################################
##################################################################
for k in range(0, len(tri.vertices)):
    if self.PointInTriangle(candidatePoint, dataX[ tri.vertices[ k ] ][ 0 ],
                            dataX[ tri.vertices[ k ] ][ 1 ],
                            dataX[ tri.vertices[ k ] ][ 2 ]):

        V1 = dataX[ tri.vertices[ k ] ][ 0 ]
        V2 = dataX[ tri.vertices[ k ] ][ 1 ]
        V3 = dataX[ tri.vertices[ k ] ][ 2 ]

        rpm1 = dataY[ tri.vertices[ k ] ][ 0 ]
        rpm2 = dataY[ tri.vertices[ k ] ][ 1 ]
        rpm3 = dataY[ tri.vertices[ k ] ][ 2 ]

        distV1 = math.sqrt(
            math.pow(candidatePoint[ 0 ] - V1[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V1[ 1 ], 2))
        distV2 = math.sqrt(
            math.pow(candidatePoint[ 0 ] - V2[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V2[ 1 ], 2))
        distV3 = math.sqrt(
            math.pow(candidatePoint[ 0 ] - V3[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V3[ 1 ], 2))

        W1 = 1 / distV1 if distV1 != 0 else 0
        W2 = 1 / distV2 if distV2 != 0 else 0
        W3 = 1 / distV3 if distV3 != 0 else 0
        # prediction = (W1 * rpm1 + W2 * rpm2 + W3 * rpm3) / (W1 + W2 + W3)
        if np.min([ distV1, distV2, distV3 ]) < 1.0:
            prediction = rpm1 if distV1 < distV2 and distV1 < distV3 else rpm2 if \
                distV2 < distV1 and distV2 < distV3 else rpm3 if \
                distV3 < distV2 and distV3 < distV1 else 0
        else:
            prediction = modeler.getBestModelForPoint(candidatePoint.reshape(1, -1)).predict(
                candidatePoint.reshape(1, -1))
        break

        # else:

or (
                        (candidatePoint == dataX[ tri.vertices[ k ] ][ 0 ]).all() \
                        or ((candidatePoint == dataX[ tri.vertices[ k ] ][ 1 ]).all()) or (
                        (candidatePoint == dataX[ tri.vertices[ k ] ][ 2 ]).all())):


if modeler.__class__.__name__ != 'TensorFlow':
    _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluate(eval.MeanAbsoluteErrorEvaluation(), X, Y,
                                                                      modeler, genericModel)
else:
    _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN(eval.MeanAbsoluteErrorEvaluation(), X,
                                                                             Y,
                                                                             modeler, output, xs)
print ("Mean absolute error on training data: %4.2f (+/- %4.2f standard error)" % (
meanError, sdError / sqrt(unseenFeaturesY.shape[ 0 ])))
print("Evaluating on seen data... Done.")
##################################################################
###################################
##################################################################
##################################################################
ax = Axes3D(fig)
def map_z2color(zval, colormap, vmin, vmax):
    #map the normalized value zval to a corresponding color in the colormap

    if vmin>vmax:
        raise ValueError('incorrect relation between vmin and vmax')
    t=(zval-vmin)/float((vmax-vmin))#normalize val
    R, G, B, alpha=colormap(t)
    return 'rgb('+'{:d}'.format(int(R*255+0.5))+','+'{:d}'.format(int(G*255+0.5))+\
           ','+'{:d}'.format(int(B*255+0.5))+')'

def tri_indices(simplices):
    #simplices is a numpy array defining the simplices of the triangularization
    #returns the lists of indices i, j, k

    return ([triplet[c] for triplet in simplices] for c in range(3))

def plotly_trisurf(x, y, z, simplices, colormap=cm.RdBu, plot_edges=None):
    #x, y, z are lists of coordinates of the triangle vertices
    #simplices are the simplices that define the triangularization;
    #simplices  is a numpy array of shape (no_triangles, 3)
    #insert here the  type check for input data

    points3D=np.vstack((x,y,z)).T
    tri_vertices=map(lambda index: points3D[index], simplices)# vertices of the surface triangles
    zmean=[np.mean(tri[:,2]) for tri in tri_vertices ]# mean values of z-coordinates of

    entropies=[]
    for tri in tri_vertices:
        entropy = (1/2+ math.log((tri[0][2]  - tri[0][1])/2)+ 1/2 + math.log((tri[1][2]  - tri[1][1])/2)+ 1/2+ math.log(tri[2][2]  - tri[2][1])/2)/3
        entropies.append(entropy)

                                                      #triangle vertices
    min_zmean=np.min(zmean)
    max_zmean=np.max(zmean)
    facecolor=[map_z2color(zz,  colormap, min_zmean, max_zmean) for zz in zmean]
    I,J,K=tri_indices(simplices)

    triangles=go.Mesh3d(x=x,
                     y=y,
                     z=z,
                     facecolor=facecolor,
                     i=I,
                     j=J,
                     k=K,
                     name=''
                    )

    if plot_edges is None:# the triangle sides are not plotted
        return [triangles]
    else:
        #define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
        #None separates data corresponding to two consecutive triangles
        lists_coord=[[[T[k%3][c] for k in range(4)]+[ None]   for T in tri_vertices]  for c in range(3)]
        Xe, Ye, Ze=[reduce(lambda x,y: x+y, lists_coord[k]) for k in range(3)]

        #define the lines to be plotted
        lines=go.Scatter3d(x=Xe,
                        y=Ye,
                        z=Ze,
                        mode='lines',
                        line=dict(color= 'rgb(50,50,50)', width=1.5)
               )
        return [triangles, lines]
##############
data = pd.read_csv(
'/home/dimitris/Desktop/kaklis.csv',nrows=50000)

#X = data.values[0:,3:23]
#Y = data.values[0:, 2]
#Z=X[3000:6000,4]
#Y=X[3000:6000,1]
Y=[]
Y1=list()
Z1=list()
for i in range(0,len(Y)):
    if i % 60 ==0:
        Y1.append(Y[i])
        Z1.append(X[i])

#X=list(range(0,len(Y1)))

axis = dict(
showbackground=True,
backgroundcolor="rgb(230, 230,230)",
gridcolor="rgb(255, 255, 255)",
zerolinecolor="rgb(255, 255, 255)",
    )

layout = go.Layout(
         title='Moebius band triangulation',
         width=900,
         height=900,
         scene=dict(
         xaxis=dict(axis),
         yaxis=dict(axis),
         zaxis=dict(axis),
        aspectratio=dict(
            x=1,
            y=1,
            z=0.5
        ),
        )
        )

#X=np.array(X).flatten()
#Y=np.array(Y1).flatten()


points2D=np.vstack([X,Y]).T
points2D=np.array(points2D)
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def numberToRGBAColor(x, vmin = -100, vmax = 100):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.hot

    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba(x)


def _colorPlotDemo():
    """
    Demo of scatter plot with varying marker colors and sizes.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cbook as cbook

    # Load a numpy record array from yahoo csv data with fields date,
    # open, close, volume, adj_close from the mpl-data/example directory.
    # The record array stores python datetime.date as an object array in
    # the date column
    datafile = cbook.get_sample_data('/usr/share/matplotlib/sample_data/goog.npy')
    try:
        # Python3 cannot load python2 .npy files with datetime(object) arrays
        # unless the encoding is set to bytes. However this option was
        # not added until numpy 1.10 so this example will only work with
        # python 2 or with numpy 1.10 and later
        price_data = np.load(datafile, encoding='bytes').view(np.recarray)
    except TypeError:
        price_data = np.load(datafile).view(np.recarray)
    price_data = price_data[-250:]  # get the most recent 250 trading days

    delta1 = np.diff(price_data.adj_close)/price_data.adj_close[:-1]

    # Marker size in units of points^2
    volume = (15 * price_data.volume[:-2] / price_data.volume[0])**2
    close = 0.003 * price_data.close[:-2] / 0.003 * price_data.open[:-2]

    fig, ax = plt.subplots()
    ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)

    ax.set_xlabel(r'$\Delta_i$', fontsize=15)
    ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=15)
    ax.set_title('Volume and percent change')

    ax.grid(True)
    fig.tight_layout()

    plt.show()