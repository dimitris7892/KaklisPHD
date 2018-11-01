from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import math
import numpy as np
from scipy.spatial import Delaunay,ConvexHull
from scipy import spatial,random
import colorsys

from utils import *

class DefaultPartitioner:
    '''
    Performs a partitioning of the data, returning:
    partitionsX: An array of instance sets from X, where each instance set belongs to the same cluster.
    partitionsY: An array of instance sets from Y, where each instance set belongs to the same cluster.
    partitionLabels: The array of labels mapped to each partition.
    centroids: A representative instance from each partition.
    clusteringModel: The sklearn clustering model (or None if not applicable)

    The Default action is one partition containing everything.
    '''

    def clustering(self, dataX, dataY=None, nClusters=None, showPlot=False, random_state=1000):
        if nClusters is not None:
            print("WARNING: DefaultPartitioner ignores number of clusters.")
        return ([dataX], [dataY], [0], [dataX[0]], None)

    def showClusterPlot(self, dataX, labels, nClusters):
        xData = np.asarray(dataX[:, 0]).T[0]
        yData = np.asarray(dataX[:, 1]).T[0]
        colors=tuple([numberToRGBAColor(l) for l in labels])

        plt.scatter(xData, yData, c=colors, cmap=plt.cm.gnuplot)
        plt.title("K-means with " + str(nClusters) + " clusters")
        plt.show()


class KMeansPartitioner(DefaultPartitioner):
    # Clusters data, for a given number of clusters
    def clustering(self, dataX, dataY = None, nClusters = None, showPlot=False, random_state=1000):

        # Check if we need to use dataY
        dataUpdatedX = dataX
        if dataY is not None:
            dataUpdatedX = np.append(dataX, np.asmatrix([dataY]).T, axis=1)

            # Init clustering model
        self._nClusters = nClusters
        self.random_state = random_state
        clusteringModel = self.getClusterer()

        # Fit the input data
        dataModel = clusteringModel.fit(dataUpdatedX)
        self._dataModel = dataModel
        # Get the cluster labels
        labels = dataModel.predict(dataUpdatedX)
        # Extract centroid values
        centroids = self.getCentroids()

        # Create partitions, based on common label
        partitionsX = []
        partitionsY = []
        partitionLabels = []
        # For each label
        for curLbl in np.unique(labels):
            # Create a partition for X using records with corresponding label equal to the current
            partitionsX.append(np.asarray(dataX[labels == curLbl]))
            # Create a partition for Y using records with corresponding label equal to the current
            partitionsY.append(np.asarray(dataY[labels == curLbl]))
            # Keep partition label to ascertain same order of results
            partitionLabels.append(curLbl)

        # Show plot, if requested
        if (showPlot):
            self.showClusterPlot(dataUpdatedX, labels, nClusters)

        return partitionsX, partitionsY, partitionLabels , centroids , clusteringModel


    def getClusterer(self, dataX=None, dataY = None):
        # Assign a default number of clusters if not provided
        if self._nClusters == None:
            if dataX != None:
                self._nClusters = int(math.log2(len(dataX) + 1))
            else:
                self._nClusters = 2

        print("Number of clusters: %d"%(self._nClusters))
                # Return clusterer
        return KMeans(n_clusters=self._nClusters , random_state=self.random_state)

    def getCentroids(self):
        return self._dataModel.cluster_centers_


class BoundedProximityPartitioner (DefaultPartitioner):
    def clustering(self, dataX, dataY = None, nClusters = None, showPlot=False, random_state=1000):
        return self._clustering(dataX, dataY, nClusters, showPlot, random_state)

    # Clusters data, for a given number of clusters
    def _clustering(self, dataX, dataY = None, nClusters = None, showPlot=False, random_state=1000, xBound = None, yBound = None):
        # Init return arrays
        partitionsX = []
        partitionsY = []
        partitionLabels = np.zeros((dataX.shape[0], 1)) # Init to input data size
        representatives = []

        # V is the set of all V values in the training set.
        #V, inverseIndex = np.unique(dataX, axis=0, return_inverse=True)
        V, inverseIndex = np.unique(dataX, return_inverse=True)

        # R is the set of corresponding RPM values in the training set.
        R = dataY[inverseIndex]

        # yBound is the bound of error in y axis
        if yBound is None:
            yBound = np.mean(np.max(dataY) - np.min(dataY)) / math.log(2.0 + dataY.shape[0], 2.0)

        # Initialize the set U of used V values to the empty set.
        U = np.empty((1, dataX.shape[1]))

        canContinue = True
        iCurClusterLbl = -1 # Init label counter
        while canContinue:
            # Get next cluster label
            iCurClusterLbl += 1

            # (Randomly) Select a value v1 in the V space not already in U.
            notInU = self.setDiff(U, V)
            v1 = notInU[np.random.randint(0, notInU.shape[0], 1)]
            # Add v1 to the U set
            U  = np.append(U, v1, axis=0)
            # Select for the given v1 a value r which is the average/median of R values for points with coordinate V=v1
            r = np.mean(self.getAllRowsYsWhereRowsXEqV(dataX, dataY, v1))
            # Collapse all V to one dimension, based on PCA
            pcaMapping = PCA(1)
            pcaMapping.fit(dataX)
            Vmapped = pcaMapping.transform(dataX)
            # xBound is the bound of error in x PCA axis
            if xBound is None:
                xBound = (np.max(Vmapped) - np.min(Vmapped)) / math.log(2.0 + dataY.shape[0], 2.0)

            # Select another point (v2, r) with r_v2 - r_v1 < bound in the V space not in U.
            notInU = self.setDiff(U, V)

            # Init selected v2 indices
            selectionIndices = []
            ## For each point in notInU
            for iIdx in range(notInU.shape[0]):
                # Examine whether the R of its corresponding group
                # is within xBound distance from the v1 R
                # and its distance in the PCA space is also equally bounded by yBound
                v2 = np.asarray([notInU[iIdx]])
                pcaDist = spatial.distance.euclidean(pcaMapping.transform(v2), pcaMapping.transform(v1))
                # Find all points with v2 as X value
                allCandidateV2Indices = self.getAllRowsYsWhereRowsXEqV(dataX, np.arange(dataY.shape[0]), v2[0])
                rDist = np.abs(np.mean(R[allCandidateV2Indices]) - r)
                if (pcaDist <= xBound) and (rDist <= yBound):
                    # then add to selection indices
                    selectionIndices.extend(allCandidateV2Indices)
                    # Update U with used v2
                    U  = np.append(U, v2, axis=0)

            # If nothing found
            if len(selectionIndices) == 0:
                # Go to next v1
                canContinue = notInU.size > 2
                continue
            # Add all points in T to new Cluster
            clusterX = np.append(v1, dataX[selectionIndices], axis=0)
            clusterY = np.append(np.asarray([r]), dataY[selectionIndices], axis=0)
            # Add to returned partitions
            partitionsX.append(clusterX)
            partitionsY.append(clusterY)
            # Create representative (centroid)
            representatives.append(np.append(np.mean(np.asarray(dataX), axis=0), np.mean(clusterY))) # TODO: Check if we need Y column

            # Store cluster
            partitionLabels[selectionIndices] = iCurClusterLbl

            # Add all points in T to U
            U = np.append(U, dataX[selectionIndices], axis=0)
            notInU = self.setDiff(U, V)

            # DEBUG LINES
            print("Created a total of %d clusters. Remaining %d instances unclustered."%(len(representatives),
                                                                                         notInU.shape[0]))
            #############

            # Repeat until no more points can be added.
            canContinue = notInU.size > 2

        # Show plot, if requested
        if (showPlot):
            self.showClusterPlot(dataX, partitionLabels, nClusters)

        # Return
        return partitionsX, partitionsY, partitionLabels, representatives, None

    def getAllRowsYsWhereRowsXEqV(self, dataX, dataY, v):
        return dataY[(dataX == v).all(axis=1)]

    def setDiff(self, U, V):
        notInU = set(map(tuple, V)).difference(map(tuple, U))
        notInU = np.asarray(list(map(list, notInU)))
        return notInU

class BoundedProximityPartionerwithTriangles:

    def showTriangle(self,x, y, z, error):
        # vertices = plotly_trisurf(x, y, z, tri.simplices)
        # polytops = list()
        k = np.vstack(np.array([ [ x ], [ y ], [ z ] ]))
        t = plt.Polygon(k, fill=False)
        plt.gca().add_patch(t)
        plt.xlim(np.min(k[ 0:, 0 ] - 10), np.max(k[ 0:, 0 ]) + 10)
        plt.ylim(np.min(k[ 0:, 1 ] - 10), np.max(k[ 0:, 1 ]) + 10)
        plt.title("Triangle with error bound " + str(round(error, 2)))
        plt.xlabel("Velocity")
        plt.ylabel("RPM +- " + str(round(error, 2)))
        plt.show()

    def sign(self,p1, p2, p3):
        return (p1[ 0 ] - p3[ 0 ]) * (p2[ 1 ] - p3[ 1 ]) - (p2[ 0 ] - p3[ 0 ]) * (p1[ 1 ] - p3[ 1 ])

    def distance(self,a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def is_between(self,a, c, b):
        return self.distance(a, c) + self.distance(c, b) == self.distance(a, b)

    def PointInTriangle(self,pt, v1, v2, v3):

        b1 = self.sign(pt, v1, v2) <= 0.0 or self.is_between(v1,pt,v2)
        b2 = self.sign(pt, v2, v3) <= 0.0 or self.is_between(v2,pt,v3)
        b3 = self.sign(pt, v3, v1) <= 0.0 or self.is_between(v3,pt,v1)

        return ((b1 == b2) and (b2 == b3))

    def getTriangle(self,X, Y, usedV, errorBound):
        X = X[ :, 0 ]
        indinXnotinU = filter(lambda x: x not in usedV, range(0, len(X)))
        if usedV == [ ]:
            pointV1 = random.sample(X, 1)
            index = random.sample(list([ num for num in range(0, len(X)) if X[ num ] > pointV1 ]), 1)
            pointV2 = X[ index ]
            pointRpm2 = Y[ index ]

        else:
            pointV1 = random.sample(X[ indinXnotinU ], 1)
            # p = [filter(lambda x: x not in usedV,range(0,len(X)))]
            x = X[ indinXnotinU ]
            y = Y[ indinXnotinU ]
            index = random.sample(list([ num for num in x if num > pointV1 ]), 1)
            pointV2 = index
            pointRpm2 = index

        pointRpm1 = np.mean([ Y[ n ] for n in range(0, len(Y)) if X[ n ] == pointV1 ])

        # points =np.vstack([[pointV1[0],pointRpm1],[pointV2[0],(pointRpm2+errorBound)[0]],[pointV2[0],(pointRpm2-errorBound)[0]]])
        # tri = Delaunay(points)
        self.showTriangle([ pointV1[ 0 ], pointRpm1 ], [ pointV2[ 0 ], (pointRpm1 + errorBound) ],
                     [ pointV2[ 0 ], (pointRpm1 - errorBound) ], errorBound)
        d = [
            self.PointInTriangle([ X[ i ], Y[ i ] ], [ pointV1[ 0 ], pointRpm1 ], [ pointV2[ 0 ], (pointRpm1 + errorBound) ],
                            [ pointV2[ 0 ], (pointRpm1 - errorBound) ])
            for i in indinXnotinU ]

        indexInTr = [ i for i in range(0, len(indinXnotinU)) if d[ i ] == True ]
        return indexInTr

    def tri_indices(self,simplices):
        # simplices is a numpy array defining the simplices of the triangularization
        # returns the lists of indices i, j, k

        return ([ triplet[ c ] for triplet in simplices ] for c in range(3))

    def plotly_trisurf(self,x, y, z, simplices):
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



    def HSVToRGB(self,h, s, v):

        (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
        return (int(255 * r), int(255 * g), int(255 * b))

    def getDistinctColors(self,n):
        huePartition = 1.0 / (n + 1)
        return np.array(self.HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n))

    def clustering(self, dataX, dataY = None, nClusters = None, showPlot=False):
          i = 0
          cutoffPoint = 0.12
          errorBound= 5
          UsedV=[]
          ##Convex HUll
          #XY=np.vstack([dataX[:,0],dataY]).T
          hull = ConvexHull(dataX)
          history=20
          pointsy=[]
          pointsx0x1=[]
          #plt.plot(dataX[ :, 0 ], dataX[0:,1], 'o')
          for simplex in hull.simplices:
              plt.plot(dataX[ simplex, 0 ], dataX[simplex,1], 'k-')

              pointsx0x1.append(dataX[ simplex, 0 ])
              pointsx0x1.append(dataX[ simplex, 1 ])
              pointsy.append(dataY[ simplex[0]])
              pointsy.append(np.mean(dataY[simplex[1]-history: simplex[1] ]))
          ######Delaunay triangulation of the training instances
          #plt.show()
          #points2D=list(dataX)[0:len(dataX):2000]
          #pointsY2D = list(dataY)[ 0:len(dataY):2000]
          points2D=[]
          pointsY2D = []

          for p in pointsy: pointsY2D.append(p)
          for p in pointsx0x1: points2D.append(p)

          points2D = np.array(dataX)

          zx = range(0, len(dataX[:,0]))
          tri = Delaunay(points2D)
          vertices = self.plotly_trisurf(points2D[:,0], zx, points2D[:,1], tri.simplices)

          for v in vertices:
              k = np.array([ [ v[ 0 ][ 0 ], v[ 0 ][ 2 ] ], [ v[ 1 ][ 0 ], v[ 1 ][ 2 ] ], [ v[ 2 ][ 0 ], v[ 2 ][ 2 ] ] ])
              #t = plt.Polygon(k, fill=False,color='red')
              #plt.gca().add_patch(t)

          #plt.show()
          cl=0
          graph={}

          neighboringV=[]
          ##################################
          for i in range(0, len(dataX)):
              pointX = dataX[ i, 0 ]
              pointY = dataX[ i, 1 ]
              neighbors = [ ]
              sets = set()
              setsy=set()
              # for v in vertices:
              cluster = {}
              flag1 = True
              c = tri.find_simplex(dataX[ i ])

              neighboringVertices=tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][i]:tri.vertex_neighbor_vertices[0][i+1]]

              for n in list(neighboringVertices):
                  if spatial.distance.euclidean(dataX[ i ], dataX[n]) <= cutoffPoint:
                      #v=[dataX[n][0],dataX[n][1]]
                      v=n
                      vy=[dataY[n]]
                      #if neighbors.__contains__(v) == False:
                      neighbors.append(v)
                      sets.add(v)
                      setsy.add(tuple(vy))
                  else:
                      flag1 = False

              if flag1 :
                  cluster[ cl ] = neighbors

              else:
                  cl += 1
              #if list(sets) !=[]:
              graph[i]=[sets,False,dataY[i]]
              #graph[""].append(n)
              neighboringV.append(neighbors)

          #####################################
          ##########################################3
          def dfs(graph, start):
              visited, stack = set(), [ start ]
              while stack:
                 vertex = stack.pop()
                 if vertex not in visited:
                    try:
                     visited.add(vertex)
                    except:
                        print visited
                        return visited
                        #vertex = stack[0].pop()
                        #visited.add(vertex)
                    stack.extend(graph[ str(np.array(vertex)) ] - visited)
              return visited
          ###############################################NEW

          ###################################
              # A function used by DFS
          def DFSUtil( v,i ,graph,cl,cly):

                  # Mark the current node as visited and print it
                  graph[ i ][ 1 ] = True
                  print v,
                  cl.append(dataX[i])
                  cly.append(graph[ i ][2])
                  # Recur for all the vertices adjacent to
                  # this vertex
                  for k in list(graph[ i ][0]):
                      if graph[k][1] == False:
                          DFSUtil(dataX[k],k, graph,cl,cly)

                          # The function to do DFS traversal. It uses
              # recursive DFSUtil()
                  #return cl
          def DFS():
                  V = len(graph)  # total vertices

                  # Mark all the vertices as not visited
                  #visited = [ False ] * (V)

                  # Call the recursive helper function to print
                  # DFS traversal starting from all vertices one
                  # by one
                  clusters=[]
                  clustersy=[]
                  uncl=[]
                  uncly=[]
                  for i in range(0,len(graph)):
                      cl=[]
                      cly=[]
                      if graph[i][1] == False:
                          #try:
                          #  splitStr=i.replace('[', '').replace(']', '').split(' ')
                            tuplex=[]
                          #  for n in splitStr:
                          #      if n!='':
                            tuplex.append(dataX[i])
                            DFSUtil(tuple(tuplex),i ,graph,cl,cly)
                            if len(cl)>1:
                                clusters.append(cl)
                                clustersy.append(cly)
                            else:
                                uncl.append(cl)
                                uncly.append(cly)

                          #except:
                          #    print i
                  return clusters,clustersy,uncl,uncly
                          ######################################
          clusters,clustersy,uncl,uncly=DFS()
          train_x_list = [ ]
          train_y_list = [ ]
          trains_x_list = [ ]
          trains_y_list = [ ]
          trlabels=np.empty(sum(map(len,clusters)))
          trData=[]
          counter=0
          for i in range(0,len(clusters)):

              trains_x_list.append(np.array(clusters[i]))
              trains_y_list.append(np.array(clustersy[i]))
              for v in clusters[i]:
                  trData.append(v)
                  trlabels[counter]=i
                  counter+=1
         # return trains_x_list, trains_y_list,list(range(0,len(clusters))), None, None
              #train_y_list.append(np.array(cluster)[:,1])
              #if len(train_x_list) > 1:
              #trains_x_list.append(np.array(train_x_list).reshape(-1,1))
              #trains_y_list.append(np.array(train_y_list).reshape(-1,1))
              #train_x_list = [ ]
              #train_y_list = [ ]
          from sklearn import  svm
          trData=np.array(trData)
          clf=svm.SVC()
          clf.fit(trData,trlabels)
          uncl=np.array(uncl).reshape(-1,2)
          uncly = np.array(uncly).reshape(-1, 1)
          labels=clf.predict(np.array(uncl))

          for curLbl in np.unique(labels):
              # Create a partition for X using records with corresponding label equal to the current
              listx=list(trains_x_list[int(curLbl)])
              listx.extend(np.asarray(uncl[ labels == curLbl ]))
              trains_x_list[int(curLbl)]=listx
              # Create a partition for Y using records with corresponding label equal to the current
              listy = list(trains_y_list[ int(curLbl) ])
              listy.extend(np.asarray(uncly[ labels == curLbl ]))
              trains_y_list[ int(curLbl) ] = listy

          return trains_x_list, trains_y_list, list(range(0, len(clusters))),None,None
          #import sys
          #labels=np.empty(len(dataX))
          #for i in range(0,len(dataX)):
          #    point=dataX[i]
          #    minDist=sys.maxsize
          #    for k in range(0,len(trains_x_list)):
          #      dist=spatial.distance.correlation(point,(np.mean(trains_x_list[k],axis=0)))
          #      if dist< minDist:
          #          labels[i]=k
          #          minDist= dist

          trains_x_list=[]
          trains_y_list=[]
          partitionLabels = [ ]
          for curLbl in np.unique(labels):
              # Create a partition for X using records with corresponding label equal to the current
              trains_x_list.append(np.asarray(dataX[ labels == curLbl ]))
              # Create a partition for Y using records with corresponding label equal to the current
              trains_y_list.append(np.asarray(dataY[ labels == curLbl ]))
              # Keep partition label to ascertain same order of results
              partitionLabels.append(curLbl)

          return trains_x_list, trains_y_list, partitionLabels, None, None
          clusters=[]
          tuples = zip(dataX[:,0],dataX[:,1])
          tuples=sorted(tuples, key=lambda k: k[0])
          st1=np.array(tuples)
          dataX=sorted(dataX, key=lambda row: row[0])
          # X=stack[0:,0]
          # Y=stack[0:,1]
          stack=[]
          c=0
          sdataX=set()
          for n in dataX:
              sdataX.add(tuple(n))
          counter=0
          while True!=[]:
               counter+=1
               if counter==13: break
               cluster = [ ]
               print len(sdataX)
               if c==0:
                stack=list(sdataX)[0]
                sdataX.remove(list(sdataX)[0])
               else:
                   try:
                       ind=random.choice(range(0,len(sdataX)))
                       stack=list(sdataX)[ind]
                   except: break
               flag=False
               stacks=[]
               flagG = False
               while stack!=[]:
                    flagG1 = True
                    if len(stack)==2:
                        stack = dfs(graph, stack)
                        if cluster.__contains__(stack):
                            break
                        cluster.append(stack)
                        stacks.append(stack)

                    else:
                        #c=0
                        while True:
                            flagG=False
                            stacksx=stacks
                            stacks=[]
                            #stacks.append(stackx)
                            for stack in stacksx:
                                 for v in np.array(list(stack)):
                                        stacki=dfs(graph, tuple(v))

                                        if cluster.__contains__(stacki)==False:
                                         cluster.append(stacki)
                                         stacks.append(stacki)
                                         c += 1
                                         flagG = True
                                        #else:

                        #if stacks!=[]:
                        #    stack=stacks.pop()
                        #else:
                        #    flagG=True
                        #    break
                            if flagG==False :
                                flagG1=False
                                break

                    if flagG1 == False: break
               x=set()
               for n in cluster:
                   sdataX = sdataX - n
                   #for k in n:
                    #del graph[str(np.array(k))]
               if clusters.__contains__(cluster)==False:
                clusters.append(cluster)
                c+=1

          train_x_list=[]
          train_y_list=[]
          trains_x_list = [ ]
          trains_y_list = [ ]
          for cluster in np.array(list(clusters)):
             for x in np.array(list(cluster)):
                 for n in x:
                     train_x_list.append(n[0])
                     train_y_list.append(n[1])
             if len(train_x_list)>1:
              trains_x_list.append(np.array(train_x_list).reshape(-1,1))
              trains_y_list.append(np.array(train_y_list).reshape(-1,1))
             train_x_list = [ ]
             train_y_list = [ ]

          return  trains_x_list,trains_y_list,partitionLabels,None,None
          #dfs(graph,tuple(dataX[0]))

          labels=itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w','g'])
          for i in range(0,len(neighboringV)):
            plt.scatter(np.array(neighboringV[i])[:,0],np.array(neighboringV[i])[:,1],c=next(labels))
          #for i in range(1, len(neighboringV)):
           #     t = plt.Polygon(neighboringV[i], fill=False,color=numberToRGBAColor(i))
            #    plt.gca().add_patch(t)

          plt.show()


          train_xy=dataX
          #train_y=dataX[:,1]
          trains_x_list=[]
          trains_y_list=[]
          labels= np.empty((len(dataX)))
          cluster=0
          neighboringV.remove(neighboringV[0])
          for v in neighboringV:
              xy = np.array(v)
              trains_x_list.append(np.array(xy[ :, 0 ]).reshape(-1,1))
              trains_y_list.append(np.array(xy[ :, 1 ]).reshape(-1,1))

          return trains_x_list, trains_y_list, range(0,len(neighboringV)), None, None

          for p in vertices:
              train_x_list = []
              train_y_list = []
              for i in range(0, len(dataX)):
                  if (self.PointInTriangle([ train_xy[ i,0 ], train_xy[ i,1 ] ], [ p[ 0 ][ 0 ], p[ 0 ][ 1 ] ],
                                      [ p[ 1 ][ 0 ], p[ 1 ][ 1 ] ], [ p[ 2 ][ 0 ], p[ 2 ][ 1 ] ])):
                      #k = [ [ p[ 0 ][ 0 ], p[ 0 ][ 1 ] ], [ p[ 1 ][ 0 ], p[ 1 ][ 1 ] ], [ p[ 2 ][ 0 ], p[ 2 ][ 1 ] ] ]
                      train_x_list.append(train_xy[ i ])
                      train_y_list.append(dataY[ i ])
                      if (len(train_x_list) > 0): labels[i]=cluster
                      else: np.delete(labels,i)
              if(len(train_x_list)>0):
                trains_x_list.append(train_x_list)
                trains_y_list.append(train_y_list)
                cluster=cluster+1


          return trains_x_list,trains_y_list,np.delete(np.unique(labels),1) , None, None
          #while i < len(dataX[ 0, 0: ]):

          #while i < len(dataX[0,0:]):
          #  for k in range(0,len(dataX)):
          #        if np.log( spatial.distance.euclidean(dataX[k,1] , dataX[i,1]))<=cutoffPoint:
          #          pass




