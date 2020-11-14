import pandas as pd
from datetime import datetime
import numpy as np
import csv

data = pd.read_csv('./data/LAROS/leo_c.csv', delimiter=',', skiprows=0)

# data = data.drop(["blFlags"], axis=1)
# data = data.drop(["wind_speed", "wind_dir","trim"], axis=1)
# x_train = data.drop(["blFlags","focs","tlgsFocs"], axis=1)

# foc = np.array(np.mean(data.values[:,6:7],axis=1))
# y_train = pd.DataFrame({
# 'FOC': foc,
# })

vslHeading = data['TrueHeading'].values
for i in range(0,len(vslHeading)):
    if vslHeading[i]<0:
        vslHeading[i]=vslHeading[i]+180

emptyColumn = np.array([0] * len(vslHeading)).reshape(-1)
data = np.append(data['ttime'].values.reshape(-1, 1),
                 np.asmatrix([
                     vslHeading,
                     data['Latitude'].values,
                     data['Longitude'].values,
                     data['WindAngle'].values,
                     data['WindSpeed'].values,
                     data['SpeedOverGroundKnots'].values,
                     (emptyColumn) / 2,
                     ((data['ME FO flow (lt/min)'].values * 790 ) / 1000000 )* 1440,
                 ]).T, axis=1)
#data['DraftAfter'].values + data['DraftFore'].values) / 2
# data = pd.read_csv(sFile, delimiter=';')
# data = data.drop(["wind_speed", "wind_dir"], axis=1)
# data = data[data['stw']>7].values
data = np.array(data)  # .astype(float)

##################################################
trData = data
company = 'DANAOS'
vessel = 'LEO C'
with open('./data/' + company + '/' + vessel + '/LEO_Cdata.csv', mode='w') as data:
    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(0, len(trData)):
        data_writer.writerow(
            [trData[i][0], trData[i][1], trData[i][2], trData[i][3], trData[i][4], trData[i][5], trData[i][6],
             trData[i][7], trData[i][8]])



'''dictionary = pd.read_excel("/home/dimitris/Downloads/LarosMap.xlsx")
dictionary = dictionary[dictionary["SITE_ID"]==359][["STATUS_PARAMETER_ID","NAME"]]
l = pd.read_csv("/home/dimitris/Desktop/LEO_C.csv",names=["id","sensor","ship_id","txt","bin","value","time"])[["sensor","time","value"]]
l["ttime"] = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').replace(second=0, microsecond=0), l["time"]))
l = l.drop(columns=["time"])
ll = l.groupby(["ttime","sensor"])["value"].mean().reset_index()
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
#change of column names:
vector.columns = list(col_names["NAME"])
dvector = pd.concat([dates,vector],axis=1, sort=False)
dvector.to_csv('./data/LAROS/leo_c.csv', index=False)'''


'''company = 'DANAOS'
    vessel = 'EXPRESS ATHENS'
    with open('/home/dimitris/Downloads/mappedData3.csv', 'rU') as myfile:
        filtered = (line.replace('\n', '') for line in myfile)
        with open('/home/dimitris/Downloads/mappedData2.csv', mode='w') as data:
            for row in csv.reader(filtered):
                if row!=[]:
                    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow(row)'''