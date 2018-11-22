import sklearn.linear_model as sk
from sklearn.linear_model import  LinearRegression
import pyearth as sp
import numpy as np
import numpy.linalg
from scipy.spatial import Delaunay
import random
import parameters
import itertools
from scipy import spatial
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from basis_expansions1 import NaturalCubicSpline
from sklearn.preprocessing import StandardScaler
from gekko import GEKKO
import  matplotlib.pyplot as plt

class BasePartitionModeler:
    def createModelsFor(self,partitionsX, partitionsY, partition_labels):
        pass

    def getBestModelForPoint(self, point):
        #mBest = None
        mBest=None
        dBestFit = 0
        # For each model
        for m in self._models:
            # If it is a better for the point
            dCurFit = self.getFitnessOfModelForPoint(m, point)
            if dCurFit > dBestFit:
                # Update the selected best model and corresponding fit
                dBestFit = dCurFit
                mBest = m

            if mBest==None:  return self._models[0]
            else: return mBest

    def  getFitnessOfModelForPoint(self, model, point):
        return 0.0

class LinearRegressionModeler(BasePartitionModeler):
    def createModelsFor(self, partitionsX, partitionsY, partition_labels):
        # Init result model list
        models = []
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            curModel = sk.LinearRegression()
            # Fit to data
            curModel.fit(partitionsX[idx], partitionsY[idx])
            # Add to returned list
            models.append(curModel)
            self._partitionsPerModel[curModel] = partitionsX[idx]

        # Update private models
        self._models = models

        # Return list of models
        return models

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))

class SplineRegressionModeler(BasePartitionModeler):
    def createModelsFor(self, partitionsX, partitionsY, partition_labels):
        # Init result model list
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            curModel = sp.Earth(use_fast=True)
            ##HP tuning
            random_search3 = RandomizedSearchCV(curModel, param_distributions=parameters.param_mars_dist, n_iter=4)
            #try:
            #    random_search3.fit(partitionsX[ idx ], partitionsY[ idx ])
            #    curModel.set_params(**self.report(random_search3.cv_results_))
            #except:
            #    print "Error on HP tuning"
            # Fit to data
            try:
             curModel.fit(partitionsX[ idx ], partitionsY[ idx ])
            except:
                print idx
            # Add to returned list
            models.append(curModel)
            self._partitionsPerModel[ curModel ] = partitionsX[ idx ]

        # Update private models
        self._models = models

        # Return list of models
        return models

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))

class CSplineRegressionModeler(BasePartitionModeler):
    def createModelsFor(self, partitionsX, partitionsY, partition_labels):
        # Init result model list
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            curModel = GEKKO()  # create GEKKO model
            curModel.options.IMODE = 2  # solution mode
            x = curModel.Param(value=partitionsX[idx])  # prediction points
            y = curModel.Var()  # prediction results

            # Fit to data
            curM=GEKKO()
            curM.options.IMODE=2
            x1 = curM.Param(value=np.array(partitionsX[ idx ][:,0]).reshape(1,-1)[0])  # prediction points
            y1 = curM.Var()  # prediction results
            curM.cspline(x1, y1, np.array(partitionsX[ idx ][:,0]).reshape(1,-1)[0], np.array(partitionsX[ idx ][:,0]).reshape(1,-1)[0])  # cubic spline
            curM.solve(disp=False)

            curModel.cspline(x, y,y1, partitionsY[ idx ])  # cubic spline
            curModel.solve(disp=False)
            # Add to returned list
            plt.plot(partitionsX[ idx ], partitionsY[ idx ], 'bo')
            plt.plot(x, y.value, 'r--', label='cubic spline')
            plt.legend(loc='best')
            plt.show()

            models.append(curModel)
            self._partitionsPerModel[ curModel ] = partitionsX[ idx ]

        # Update private models
        self._models = models

        # Return list of models
        return models

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))



class NCSRegressionModeler(BasePartitionModeler):
    def createModelsFor(self, partitionsX, partitionsY, partition_labels):
        knots=12

        def make_natural_cubic_regression(n_knots):
                return Pipeline([
                    ('standardizer', StandardScaler()),
                    ('nat_cubic', NaturalCubicSpline(-2, 2, n_knots=n_knots)),
                    ('regression', LinearRegression(fit_intercept=True))
                ])

        # Init result model list
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model

            curModel = Pipeline([
                    ('standardizer', StandardScaler()),
                    ('nat_cubic', sp.Earth(max_degree=3, penalty=11.0,smooth=True,allow_linear=False)),
                    ('regression', LinearRegression(fit_intercept=True))
                ])
            ##HP tuning
            #random_search3 = RandomizedSearchCV(curModel, param_distributions=parameters.param_mars_dist, n_iter=4)
            # try:
            #    random_search3.fit(partitionsX[ idx ], partitionsY[ idx ])
            #    curModel.set_params(**self.report(random_search3.cv_results_))
            # except:
            #    print "Error on HP tuning"
            # Fit to data
            try:
                curModel.fit(partitionsX[ idx ], partitionsY[ idx ])
            except:
                print idx
            # Add to returned list
            models.append(curModel)
            self._partitionsPerModel[ curModel ] = partitionsX[ idx ]

        # Update private models
        self._models = models

        # Return list of models
        return models



    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[model])-point))

    def report(self,results, n_top=1):
        for i in range(1, n_top + 1):
            candidate = np.flatnonzero(results[ 'rank_test_score' ] == i)
            # print("Mean validation score: {0:.3f} )".format(
            # results[ 'mean_test_score' ][ candidate[0]]))
            # print("Parameters: {0}".format(results[ 'params' ][ candidate[0] ]))
            return results[ 'params' ][ candidate[ 0 ] ]

class MarkovChainModeling(SplineRegressionModeler,LinearRegressionModeler):
    def FindMostrewardingNextTrSet(self,partitionsX):
        pass

    def Initialize(self,partitionsX,partition_labels):
        # The statespace
        states=[]
        for idx, pCurLbl in enumerate(partition_labels):
            states.append( [partitionsX[idx],pCurLbl] )

        # Possible sequences of eventsfor L in range(0, len(stuff)+1):
        for l in range(0, len(states) + 1):
            for subset in itertools.combinations(states, l):
                pass
                #transinions.append()

        # Probabilities matrix (transition matrix)
        transitionMatrix = [ [ 0.2, 0.6, 0.2 ], [ 0.1, 0.6, 0.3 ], [ 0.2, 0.7, 0.1 ] ]

    def FindMostrewardingNextTrSet(self,transMatrx,currPartitionX,partitionsX):
            pass