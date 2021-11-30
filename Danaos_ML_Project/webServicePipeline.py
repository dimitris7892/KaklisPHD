import json
import numpy as np
from flask import Flask, send_file, make_response, render_template
import preprocessing as pre
import insertAtDB as dbIns
import seaborn as sns
from flask import Response
import pandas as pd
import webServicePlotting as plott
import os

app = Flask(__name__)

env = os.getenv("run_env", "dev")

if env == "dev":
    BASE_URL = "http://localhost:4444/"
elif env == "prod":
    BASE_URL = "https://python-sci-plotting.herokuapp.com/"



connInst = dbIns.DBconnections()
instanceInfoInstallationsDB = connInst.DBcredentials("sa", "Man1f0ld", 'localhost, 1433', 'Installations')
connInst.connectToDB(instanceInfoInstallationsDB)

connData = dbIns.DBconnections()
instanceDataDB = connData.DBcredentials("sa", "Man1f0ld", 'localhost, 1433', 'DANAOS')
cnxn = connData.connectToDB(instanceDataDB)

connData.createTables("DANAOS", "EXPRESS_ATHENS", 'tlg')
connData.createTables("DANAOS", "EXPRESS_ATHENS", 'raw')
connData.createTables("DANAOS", "EXPRESS_ATHENS", 'processed')
#prepro = pre.preprocessing("DANAOS")
#df = pd.read_sql("SELECT  * FROM  EXPRESS_ATHENS_PROCESSED ", prepro.cnxn)
x=0

def setup_app(app):
    pass

setup_app(app)

@app.route("/")
def index():

    return render_template('index.html', base_url=BASE_URL)

    #json.dumps(np.array(dataCleaned).astype(str).tolist())

@app.route('/runPipeline/<params>', methods=['GET'])
def runPipeline(params):

    params = params.split(" ")
    company = params[0]
    vessel = params[1]
    stringBuilder = ""
    try:
        prepro = pre.preprocessing(company)
        data = prepro.extractData(prepro.cnxn, vessel)
        print("RAW data shape: "+ str(data.shape))
        stringBuilder += "RAW data shape: "+ str(data.shape)
        dataCleaned = prepro.processData(data)

        print("Cleaned data shape: " + str(np.array(dataCleaned).shape))
        stringBuilder += "\nCleaned data shape: " + str(np.array(dataCleaned).shape)

        connInst = dbIns.DBconnections()
        instanceInfoInstallationsDB = connInst.DBcredentials("sa", "Man1f0ld", 'localhost, 1433', 'Installations')
        connInst.connectToDB(instanceInfoInstallationsDB)

        prepro.connData.createTables(company, vessel, 'processed')

        prepro.connData.insertIntoDataTableforVessel(company, vessel+"_PROCESSED", connInst, "raw", "bulk", None,
                                                     dataCleaned)

        stringBuilder += "\nFinished Preprocessing!"
        return make_response(stringBuilder)
    except  Exception as e:

        return make_response(e, 400)


@app.route('/plots/focplots/features/<features>', methods=['GET'])
def focplot(features):

    features = features.split(" ")
    company = features[0]
    vessel = features[1]
    #feature = features[2]


    keys = "<br/>".join([k for k in df.keys().values])
    try:
        # parse columns
        #parsed_features = [feature.strip() for feature in features.split(',')]
        bytes_obj = plott.get_foc_plot_as_bytes(df, )

        return  Response(bytes_obj.getvalue(), mimetype='image/png')
            #send_file(bytes_obj,
                         #attachment_filename='plot.png',
                         #mimetype='image/png')
    except ValueError:
        # something went wrong to return bad request
        return make_response('Unsupported request, probably feature names are wrong. \nList of available features: <br/>'+keys, 400)



@app.route('/plots/distplots/features/<features>', methods=['GET'])
def distplot(features):

    features = features.split(" ")
    company = features[0]
    vessel = features[1]
    feature = features[2]

    #prepro = pre.preprocessing(company)
    #df = pd.read_sql("SELECT  * FROM  "+vessel+"_PROCESSED ", prepro.cnxn)
    keys = "<br/>".join([k for k in df.keys().values])
    try:
        # parse columns
        #parsed_features = [feature.strip() for feature in features.split(',')]
        bytes_obj = plott.get_seaborn_plot_as_bytes(df, feature)

        return send_file(bytes_obj,
                         attachment_filename='plot.png',
                         mimetype='image/png')
    except ValueError:
        # something went wrong to return bad request
        return make_response('Unsupported request, probably feature names are wrong. \nList of available features: <br/>'+keys, 400)


if __name__ == "__main__":

    app.run(host="localhost", port=4444, debug=True)