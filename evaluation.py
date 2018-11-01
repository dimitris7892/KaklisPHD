import numpy as np
import dataModeling as dt

class Evaluation:
    def evaluate(self, train, unseen, model):
        return 0.0

class MeanAbsoluteErrorEvaluation (Evaluation):
    '''
    Performs evaluation of the datam returning:
    errors: the list of errors over all instances
    meanError: the mean of the prediction error
    sdError: standard deviation of the error
    '''
    def evaluate(self, unseenX, unseenY, modeler):
        lErrors = []
        for iCnt in range(np.shape(unseenX)[0]):
            pPoint = unseenX[iCnt].reshape(1, -1)#[0] # Convert to matrix
            trueVal = unseenY[iCnt]
            prediction = modeler.getBestModelForPoint(pPoint).predict(pPoint)
            lErrors.append(abs(prediction - trueVal))
        errors = np.asarray(lErrors)

        return errors, np.mean(errors), np.std(lErrors)