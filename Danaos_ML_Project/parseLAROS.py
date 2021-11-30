import pandas as pd
from datetime import datetime
import numpy as np
import csv
from global_land_mask import globe
import re
import os
import seaborn as sns

is_in_ocean = globe.is_ocean(23,54)

vessel = 'SAMSON'
def main():
    pl = ParseLaros()
    pl.extractRAWLarosData(6, vessel+'.csv',vessel+'_data.csv')

    pl.extractFeaturesFromLarosData(vessel+'_data.csv', vessel)


class ParseLaros:

    def deterMineWindSpeedMU(self,company, vessel ):

        data = pd.read_csv('./data/' + company + '/' + vessel + '/' + vessel +'.csv').values
        windSpeed = data[:, 5].astype(float)

        return True if np.max(windSpeed) >= 40 else False

    def findFeatures(self, data, company, vessel):

        emptyColumn = np.array(['nan'] * len(data['ttime'])).reshape(-1)
        ##init keys
        vslHeading = ""
        wsMS = ""
        waDeg = ""
        rpm = ""
        power = ""
        sovg = ""
        stw = ""
        drft = ""
        lat = ""
        lon = ""
        mefoflow = ""
        aftForeFlag = False
        ####

        wsList = np.array([k for k in data.keys() if "windspeed" in str(k).lower()])
        waList = np.array([k for k in data.keys() if "windangle" in str(k).lower()])
        draftList = np.array([k for k in data.keys() if "draft" in str(k).lower()])
        latList = np.array([k for k in data.keys() if "latitude" in str(k).lower()])
        lonList = np.array([k for k in data.keys() if "longitude" in str(k).lower() or "longtitude" in str(k).lower()])
        stwList = np.array([k for k in data.keys() if "stw" in str(k).lower()])
        sovgList = np.array([k for k in data.keys() if "speedover" in str(k).lower()])
        rpmList = np.array([k for k in data.keys() if "rpm" in str(k).lower()])
        powerList = np.array([k for k in data.keys() if "power" in str(k).lower()])
        vslHeadingList = np.array([k for k in data.keys() if "heading" in str(k).lower()])
        mefoflowList = np.array([k for k in data.keys() if ("m/efoflow" or  "mefoflow") in str(k).lower() ])

        if len(latList) > 0:

            lat = latList[0]

        if len(lonList) > 0:

            lon = lonList[0]

        if len(wsList) > 0:
            if len(wsList) > 1:

                wsMS = [k for k in wsList if "m/s" in  str(k).lower()]
                if len(wsMS) > 0:
                    wsMS = wsMS[0]
                else:
                    wsMS = wsList[0]
            else:
                wsMS = wsList[0]

        if len(waList) > 0:

            waDeg = waList[0]

        if len(draftList) > 0:
            if len(draftList) > 1:

                drftAFT = [k for k in draftList if " aft" in str(k).lower()]
                drftFORE = [k for k in draftList if " fore" in str(k).lower()]
                drftMid = [k for k in draftList if "mid" in str(k).lower()]

                if len(drftAFT) > 0 and len(drftFORE) > 0 :
                    aftForeFlag = True
                    drftaft = drftAFT[0]
                    drftfore = drftFORE[0]
                    drft = "aft+fore/2"
                else:
                    drft = draftList[0]
            else:

                drft = draftList[0]

        if len(rpmList) > 0:

           rpm = [k for k in rpmList if "m/e" or 'me'  in str(k).lower() and " " not in str(k)]
           rpm = rpm[0] if len(rpm) > 0 else ""


        if len(powerList) > 0:

            power = [k for k in powerList if "m/e" or 'me' in str(k).lower() and " " not in str(k)]
            power = power[0] if len(power) > 0  else ""

        if len(stwList) > 0:

           stw = stwList[0]

        if len(sovgList) > 0:

           sovg = sovgList[0]

        if len(vslHeadingList) > 0:

           vslHeading = vslHeadingList[0]

        if len(mefoflowList) > 1:

           mefoflow = [k for k in mefoflowList if "m/e" or 'me' in str(k).lower() and " " not in str(k)][0]
        else:
           mefoflow = mefoflowList[0]

        keys = "Keys: " +"\n" + vslHeading + "\n" + lat +"\n" + lon +"\n" + waDeg+"\n" + wsMS+"\n" + stw+"\n" + drft+"\n" + mefoflow+"\n" +sovg+"\n" + rpm +"\n" + power + '\n'
        print("Keys: " +keys)

        with open('./data/'+company+'/' + vessel + '/keysExtracted.txt', 'w') as f:
            f.write(keys)

        vslHeading = data[vslHeading].values if vslHeading != "" else emptyColumn
        lat = data[lat].values if lat != "" else emptyColumn
        lon = data[lon].values if lon != "" else emptyColumn
        waDeg = data[waDeg].values if waDeg != "" else emptyColumn
        wsMS = data[wsMS].values if wsMS != "" else emptyColumn
        stw = data[stw].values if stw != "" else emptyColumn
        if drft != "":
            if aftForeFlag:
                drft = np.round((data[drftaft].values + data[drftfore].values) / 2, 2)
            else:
                drft = data[drft].values
        else:
            drft = emptyColumn
        #drft = data[drft].values if drft != "" else emptyColumn

        sovg = data[sovg].values if sovg != "" else emptyColumn
        rpm = data[rpm].values if rpm != "" else emptyColumn
        power = data[power].values if power != "" else emptyColumn
        mefoflow = data[mefoflow].values if mefoflow != "" else emptyColumn



        return vslHeading, lat, lon, waDeg, wsMS, stw, drft, mefoflow, sovg, rpm, power,


    def extractFeaturesFromLarosData(self, fileName, company,  vessel):


        data = pd.read_csv( fileName, delimiter=',', skiprows=0)

        vslHeading, lat, lon, waDeg, wsMS, stw, drft, mefoflow, sovg, rpm, power = self.findFeatures(data, company, vessel)


        data = np.append(data['ttime'].values.reshape(-1, 1),
                         np.asmatrix([
                             vslHeading,
                             lat,
                             lon,
                             waDeg,
                             wsMS,
                             stw,
                             drft,
                             mefoflow,
                             sovg,
                             rpm,
                             power
                         ]).T, axis=1)

        data = np.array(data)  # .astype(float)

        ##################################################
        trData = data[data[:,0]>='2019-09-01']

        if os.path.isdir('./data/' + company ) == False:
            os.mkdir('./data/' + company + '/')

        if os.path.isdir('./data/' + company + '/' + vessel) == False:
            os.mkdir('./data/' + company + '/' + vessel)

        with open('./data/' + company + '/' + vessel + '/' + vessel +'.csv', mode='w') as data:

            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(0, len(trData)):
                data_writer.writerow(
                    [trData[i][0], trData[i][1], trData[i][2], trData[i][3], trData[i][4], trData[i][5], trData[i][6],
                     trData[i][7], trData[i][8], trData[i][9],trData[i][10], trData[i][11]])


    def extractRAWLarosData(self, siteId, fileName, destinationFile):

        #dictionary = pd.read_excel("/home/dimitris/Downloads/LarosMap.xlsx")
        dictionary = pd.read_excel(r"./LarosMap.xlsx")

        dictionary = dictionary[dictionary["SITE_ID"] == siteId][["STATUS_PARAMETER_ID", "NAME"]]
        l = pd.read_csv(  fileName,
                        names=["id", "sensor", "ship_id", "txt", "bin", "value", "time"])[["sensor", "time", "value"]]


        l["ttime"] = list(
            map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').replace(second=0, microsecond=0) if str(x).__contains__('.') else
                datetime.strptime(x, '%Y-%m-%d %H:%M:%S'), l["time"]))


        l = l.drop(columns=["time"])
        ll = l.groupby(["ttime", "sensor"])["value"].mean().reset_index()
        vector = ll.pivot(index='ttime', columns='sensor', values='value').reset_index()
        vector = vector.sort_values(by="ttime").reset_index().drop(columns=["index"])
        dates = vector["ttime"]
        del vector["ttime"]
        for i in vector.columns:
            vector[i] = vector[i].astype(float).fillna(method="ffill").fillna(method="bfill")
        helper = pd.DataFrame(vector.columns)
        helper["STATUS_PARAMETER_ID"] = helper["sensor"].apply(int)
        helper = helper.drop(columns=["sensor"])
        col_names = pd.merge(helper, dictionary, how='left', on=['STATUS_PARAMETER_ID'])
        # change of column names:
        vector.columns = list(col_names["NAME"])
        dvector = pd.concat([dates, vector], axis=1, sort=False)
        dvector.to_csv('./' + destinationFile, index=False)




if __name__ == "__main__":
    main()