import airflow
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
import sys
from datetime import date, datetime
import scriptForLAROSExtraction as dataRetr
import parseLAROS as pLAROS
import preProcessLegs as preLegs
import extractLegs as extLegs
import dataReading as dRead
import dataModeling as dModel
import neuralDT as nDT
import generateReportCMA

#"DANAOS", "CMA CGM TANCREDI", "10.2.5.80", "or19", "laros", "laros"
'''company = sys.argv[0]
vessel = sys.argv[1]
hostname = sys.argv[2]
sid = sys.argv[3]
usr = sys.argv[4]
pwd = sys.argv[5]
siteID = sys.argv[6]'''


company = input("Company:")
vessel = input("Vessel:")
hostname = input("host:")
sid = input("sid:")
usr = input("usr:")
pwd = input("pwd:")
siteID = input("siteID:")




dretrieve = dataRetr.scriptForLAROSExtraction(company, vessel, hostname, sid, usr, pwd, siteID)
plaros = pLAROS.ParseLaros()
preLegs = preLegs.preProcessLegs(knotsFlag=False)
extLegs = extLegs.extractLegs(company, vessel, sid, hostname, pwd, usr)
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


args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(1),       # this in combination with catchup=False ensures the DAG being triggered from the current date onwards along the set interval
    'provide_context': True,                            # this is set to True as we want to pass variables on from one task to another
}

dag = DAG(
    dag_id='runPipeLine',
    default_args=args,
	schedule_interval='@daily',        # set interval
	catchup=False,                    # indicate whether or not Airflow should do any runs for intervals between the start_date and the current date that haven't been run thus far
)


task1 = PythonOperator(
    task_id='get_data_from_db',
    python_callable = dretrieve.extractDataForSiteID,            # function called to get data from the Kafka topic and store it
    op_kwargs={'vessel': vessel,
               'siteid': siteID,
               },
    dag=dag,
)

task2 = PythonOperator(
    task_id='parse_laros',
    python_callable=plaros.extractRAWLarosData(siteID, './'+vessel+'_'+siteID+'_'+str(date.today())+'_.csv', vessel+'_data.csv'),                      # function called to load data for further processing
    op_kwargs={'path_new_data': PATH_NEW_DATA,
               'path_test_set': PATH_TEST_SET},
    dag=dag,
)



task1 >> task2        # set task priority
