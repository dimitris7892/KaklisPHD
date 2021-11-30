import glob, os
import dataReading
import dataReading as dRead
#from Danaos_ML_Project import dataReading as DANdRead ##   NEWWWW
import dataReading as DANdRead
import dataPartitioning as dPart
import dataModeling as dModel
import Danaos_ML_Project.evaluation as eval
import numpy as np
import json
import seaborn as sns
import datetime
from sklearn.cluster import KMeans
from math import sqrt
import pandas as pd
from pylab import *
import datetime
import Danaos_ML_Project.plotResults as pltRes
import Danaos_ML_Project.generateProfile as genProf
import mappingData_functions as mpf
import preProcessLegs as preLegs
import extractLegs as extrctLegs
import runPipeline1 as runPipe

# vessels = ['LEO C', 'HYUNDAI SMART', 'GENOA']
#vessels = ['LEO C']
#vessels = ['GENOA']
vessels = [ 'NERVAL']


prelegs = preLegs.preProcessLegs(knotsFlag = False)
gener = genProf.BaseProfileGenerator()
mapping = mpf.Mapping()
dread = dataReading.BaseSeriesReader()


def main():

   #vessel = 'BIANCA'

   #1 Task extract legs
   for vessel in vessels:
        extLegs = extrctLegs.extractLegs(company='DANAOS',
                                        vessel= vessel,
                                        sid='OR12',
                                        server='10.2.5.80',
                                        password='shipping',
                                        usr='shipping')

        #extLegs.runExtractionProcess()
        #return
        #2 Task preprocess Legs
        #dataRaw = prelegs.concatLegs(vessel, './legs/'+vessel+'/')
        #prelegs.extractDataFromLegs(dataRaw, './legs/'+vessel+'/', vessel, 'legs')

        #3 Train Model in cleaned dataset
        algs = ['NNW1']
        cls = ['KM']
        fromFile = True
        runPipe.main(vessel, algs, cls, fromFile, None)
# # ENTRY POINT
if __name__ == "__main__":
    main()
