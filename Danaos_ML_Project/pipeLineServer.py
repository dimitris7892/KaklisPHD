#import airflow
#from airflow.models import DAG
#from airflow.operators.python_operator import PythonOperator
import sys
from datetime import date, datetime
import  dataReading as dRead
import scriptForLAROSExtraction as dataRetr
import parseLAROS as pLAROS
import preProcessLegs as preLegs
import extractLegs as extLegs
import dataModeling as dModel
import neuralDT as nDT
import insertAtDB as inDB
import time
import  runPipeline1 as runPipe
import numpy as np
import pandas as pd
import generateReportCMA as genRep
import configparser
import requests
from fastapi import APIRouter

router = APIRouter()

#config = configparser.ConfigParser()
#config.read('configPipeLine.ini')['DEFAULT']
config = {}

config["DEFAULT"] = {

"larosext" : False,
"parselaros": False,
"mappwithws" : False,
"extlegs" : False,
"cleandata" : False,
"insertatdb" : False,
"trainmodel" : False,
"evalneuralonlegs" : False,
"filldt" : False,
'updateProfileInDB': True,
"generaterep" : False }

configProp = config['DEFAULT']


sidShipping = 'OR12'
pwdShipping = 'shipping'
usrShipping = 'shipping'
company = "DANAOS"
vessel = 'NERVAL'
hostname = "10.2.5.80"
sid = "or19"
usr = "laros"
pwd = "laros"
siteID = 3



dretrieve = dataRetr.scriptForLAROSExtraction(company, vessel, hostname, sid, usr, pwd, siteID)
plaros = pLAROS.ParseLaros()
extLegs = extLegs.extractLegs(company, vessel, hostname, sidShipping, pwdShipping, usrShipping)
preLegs = preLegs.preProcessLegs(knotsFlag=False)
dRead = dRead.BaseSeriesReader()
dModel = dModel.BasePartitionModeler()
nDT = nDT.neuralDT(vessel)



CLIENT = 'kafka:9092'
TOPIC = 'TopicA'

PATH_NEW_DATA = './data/to_use_for_model_update/'
PATH_USED_DATA = './data/used_for_model_update/'
PATH_TEST_SET = './data/test_set.p'

PATH_INITIAL_MODEL = './models/initial_model/'
PATH_CURRENT_MODEL = './models/current_model/'

PATH_MODEL_ARCHIVE = './models/archive/'

N_STEPS = 15
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 100



def main():

    ##init connections
    connInst = inDB.DBconnections(usr='shipping', password='shipping', sid='or12')
    instanceDataDB = connInst.DBcredentials("sa", "sa1!", '10.2.4.54', company)
    cnxn = connInst.connectToDB(instanceDataDB)
    vesselId = connInst.findVesselCode(vessel)
    print("Vessel ID: " +str(vesselId))
    ##init connections

    ##DATA RETRIEVAL LAROS
    if configProp['larosext'] == True:
        print("Extracting LAROS data from DB . . .\n")
        dretrieve.extractDataForSiteID(vessel, siteID, '2019-01-01 11:00', '2020-05-01 12:00')
    ##DATA RETRIEVAL LAROS

    ###PARSE LAROS
    if configProp['parselaros'] == True:
        print("Parsing LAROS data  . . .\n")
        #plaros.extractRAWLarosData(siteID, './pipeLineData/' + vessel + '_' + str(siteID) + '_.csv', './pipeLineData/'+vessel + '_data.csv')
        plaros.extractFeaturesFromLarosData('./pipeLineData/' + vessel + '_data.csv', company, vessel )
    ###PARSE LAROS

    ##instance preProcess
    knotsFlag = plaros.deterMineWindSpeedMU(company, vessel)
    print("\nknotsFlag: " +str(knotsFlag))
    preLegs.knotsFLag = knotsFlag
    ##instance preProcess

    ###Map Data with Weather Service
    if configProp['mappwithws'] == True:
        print("Mapping data  with WS . . .\n")
        dRead.GenericParserForDataExtraction('LEMAG', company, vessel, driver='ORACLE',
                                                     server = '10.2.5.80',
                                                     sid='OR12', usr='shipping', password='shipping',
                                                     rawData=True, telegrams=True, companyTelegrams=False,
                                                     pathOfRawData='/home/dimitris/Desktop/SEEAMAG')
    ###Map Data with Weather Service

    ####EXTRACT LEGS
    if configProp['extlegs'] == True:
        extLegs.runExtractionProcess()
        dataRaw = preLegs.concatLegs(vessel, './legs/' + vessel + '/', False)
        preLegs.extractDataFromLegs(dataRaw, './legs/' + vessel + '/', vessel, 'legs')
    ####EXTRACT LEGS

    #################### PROCESS RAW DATA - DATA CLEANING
    if configProp['cleandata'] == True:
        print("Processing data  . . .\n")
        preLegs.correctData(vessel)
        minSpeed = 12  ## speed to cut off and below
        dataRaw = pd.read_csv('./data/DANAOS/' + vessel + '/mappedData.csv', delimiter=',').values
        preLegs.buildTableWithMeanValues(dataRaw, vessel, minSpeed)
        preLegs.cleanDataset(dataRaw, "raw", vessel, minSpeed)
        preLegs.extractLegsFromRawCleanedData(vessel, 'raw')
    #################### PROCESS RAW DATA - DATA CLEANING



    ##INSERT AT DB RAW, FILTERED, TLG #########################################
    if configProp['insertatdb'] == True:
        print("Inserting data  at DB. . .\n")

        dataCleaned = pd.read_csv('./consProfileJSON_Neural/cleaned_raw_' + vessel + '.csv', delimiter=',').values

        connInst.createTables(company, vessel, 'installations')
        connInst.createTables(company, vessel, 'tlg')
        connInst.createTables(company, vessel, 'raw')
        connInst.createTables(company, vessel, 'processed')
        values = inDB.Values()
        connInst.insertIntoInstallations(vessel)
        dataVesselClassListTlg = inDB.fillDataTlgVesselClass(company, vessel)
        connInst.insertIntoDataTableforVessel(company, vessel, vessel + "_TLG", connInst, "tlg", "", dataVesselClassListTlg)

        dataVesselClassListRaw, dataVesselClassListBulkRaw = inDB.fillDataVesselClass(company, vessel, connInst, [])

        dataVesselClassList, dataVesselClassListBulk = inDB.fillDataVesselClass(company, vessel, connInst, dataCleaned)



        connInst.insertIntoDataTableforVessel(company, vessel, vessel, connInst, "raw", "bulk", dataVesselClassListRaw,
                                              dataVesselClassListBulk)

        connInst.insertIntoDataTableforVessel(company,vessel, vessel+"_PROCESSED", connInst, "raw", "bulk", dataVesselClassList,
                                              dataVesselClassListBulk)
    ##INSERT AT DB RAW, FILTERED, TLG #########################################

    ### MODEL TRAINING ###########
    if configProp['trainmodel'] == True:

        fromFile = False
        processedData = None

        if fromFile == False:
            ##read trData from DB
            tableNameVslProcessed = vessel+"_PROCESSED"
            installationId = vesselId

            data = pd.read_sql_query(
                'SELECT  t_stamp,'
                         +'course,'
                         +'trim,'
                         +'trim,'
                         +'at_port,'
                         +'daysFromLastPropClean,'
                         +'daysFromLastDryDock,'
                         +'curr_speed,'
                         +'draft,'
                         +'curr_dir,'
                         +'sensor_wind_dir,'
                         +'sensor_wind_speed,'
                         +'stw,'
                         +'comb_waves_height,'
                         +'comb_waves_dir, '
                         + 'me_foc, '
                         + 'tlg_id, '
                         + 'trim, '
                         + 'tlg_id, '
                         + 'tlg_id, '
                         + 'swell_height, '
                         + 'swell_dir, '
                         + 'comb_waves_height,'
                         + 'comb_waves_dir, '
                         + 'wind_waves_height,'
                         + 'wind_waves_dir, '
                         + 'lat,'
                         + 'lon, '
                         + 'wind_speed,'
                         + 'wind_dir, '
                         + 'curr_speed, '
                         + 'curr_dir, '
                         + 'rpm,'
                         + 'power '

                        +'FROM  '  + tableNameVslProcessed + '    WHERE installationsId =  '"'" + str(installationId) + "'"'   ', connInst.connection)

            processedData = data.values

        algs = ['NNW1']
        cls = ['KM']

        runPipe.main(vessel, algs, cls, fromFile, processedData)
    ### MODEL TRAINING ###########


    ##MODEL EVAL ON LEGS
    if configProp['evalneuralonlegs'] == True:
        print("Evauating Neural on Legs . . .\n")
        #preLegs.extractLegsFromRawCleanedData(vessel, 'raw')
        nDT.evaluateNeuralDTOnFiltered(vessel, 15, False, 'ws', 'nn', 10)
        nDT.evaluateNeuralDTOnFiltered(vessel, 15, False, 'raw', 'nn', 10)
        nDT.evaluateNeuralDTOnFiltered(vessel, 15, False, 'raw', 'stats', 10)
    ##MODEL EVAL ON LEGS

    if configProp['filldt'] == True:

        fromFileDT = True

        if fromFileDT == False:
            tableNameVslProcessed = vessel + "_PROCESSED"
            installationId = vesselId

            data = pd.read_sql_query(
                'SELECT  t_stamp,'
                + 'course,'
                + 'trim,'
                + 'trim,'
                + 'at_port,'
                + 'daysFromLastPropClean,'
                + 'daysFromLastDryDock,'
                + 'curr_speed,'
                + 'draft,'
                + 'curr_dir,'
                + 'sensor_wind_dir,'
                + 'sensor_wind_speed,'
                + 'stw,'
                + 'comb_waves_height,'
                + 'comb_waves_dir, '
                + 'me_foc, '
                + 'tlg_id, '
                + 'trim, '
                + 'tlg_id, '
                + 'tlg_id, '
                + 'swell_height, '
                + 'swell_dir, '
                + 'comb_waves_height,'
                + 'comb_waves_dir, '
                + 'wind_waves_height,'
                + 'wind_waves_dir, '
                + 'lat,'
                + 'lon, '
                + 'wind_speed,'
                + 'wind_dir, '
                + 'curr_speed, '
                + 'curr_dir, '
                + 'rpm,'
                + 'power '

                + 'FROM  ' + tableNameVslProcessed + '    WHERE installationsId =  '"'" + str(installationId) + "'"'   ',
                connInst.connection)

            processedData = data.values


        else:
            processedData = pd.read_csv('./consProfileJSON_Neural/cleaned_raw_' + vessel + '.csv', delimiter=',').values

        print("Filling Neural DT . . .\n")
        nDT.fillDecisionTree(nDT.imo, vessel, processedData, 15, 'db', 11)


    if configProp['updateProfileInDB'] == True:

        updateServer = "WeatherServer"
        response = requests.get(
                "https://localhost:5005/UpdateVesselProfile/updateServer="+updateServer,
                verify=False)


    ### Report Generation (CMA)
    if configProp['generaterep'] == True:
        print("Generating Report . . .\n")
        genrep = genRep.GenerateReport(company, vessel, hostname, sidShipping, pwdShipping, usrShipping)
        genrep.generate()
    ### Report Generation (CMA)

if __name__ == "__main__":
    main()

