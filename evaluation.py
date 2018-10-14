import numpy as np
import dataModeling as dt

class Evaluation:
    def evaluate(self, train, unseen, model):
        return 0.0

class MeanAbsoluteErrorEvaluation (Evaluation):
    def evaluate(self, unseenX, unseenY, modeler):
        lErrors = []
        for iCnt in range(np.shape(unseenX)[0]):
            pPoint = unseenX[iCnt]
            trueVal = unseenY[iCnt]
            prediction = modeler.getBestModelForPoint(pPoint).predict(pPoint)
            lErrors.append(abs(prediction - trueVal))
        errors = np.asarray(lErrors)

        return np.mean(errors)