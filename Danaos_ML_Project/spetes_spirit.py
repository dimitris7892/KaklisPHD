import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime
from math import sqrt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import bokeh
import scipy
from fastapi import APIRouter
import os
import io
import seaborn as sns
from sklearn.linear_model import LinearRegression
import json
from pandas import json_normalize
import requests
from scipy import spatial
from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd, geoviews as gv, cartopy.crs as crs
from colorcet import fire
from IPython.display import HTML
import holoviews as hv
from datashader.utils import lnglat_to_meters
from holoviews.operation.datashader import datashade
from geoviews.tile_sources import EsriImagery
import holoviews.plotting.bokeh
import seaborn as sns
from bokeh.plotting import show, figure
from beaufort_scale import beaufort_scale_kmh
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import matplotlib
import param, panel as pn
from colorcet import palette
from sklearn.cluster import KMeans , DBSCAN
from sklearn.decomposition import PCA
from wand.image import Image as WImage
import matplotlib.cm as cm
from IPython.display import Image
import itertools
import plotly.graph_objects as go
import math
import plotly
from scipy import spatial,random
import dataframe_image as dfi
import colorsys
import csv
from sklearn.decomposition import PCA
hv.extension('bokeh', 'matplotlib')
import glob , path
from pyproj import Proj, transform
import pyproj
import geopy
from geopy import distance
import json
import leafmap


def getData(company, vessel):
    dataSet = []
    path = './data/' + company + '/' + vessel + '/'
    for infile in sorted(glob.glob(path + '*.xlsx')):
        data = pd.read_excel(infile, skiprows=1, index_col=0, header=None)
        dfTranposed = data.T
        dfTranposed = dfTranposed.drop(['REPORT'], axis=1)
        dataSet.append(dfTranposed)
        print(str(infile))

    dataSet = pd.concat(dataSet)
    print(dataSet.shape)
    dataSet.head(10)

    dfTranposed = dataSet
    return dfTranposed


def extractReproducedRoute():
    dfTranposed = getData('MODERNA','SPETSES SPIRIT')

    dates = dfTranposed['Date'].values[0]
    lat = dfTranposed['Latitude'].values[0]
    lon = dfTranposed['Longgitude'].values[0]

    row0 = dfTranposed.values[4]
    print(row0)

    row3 = dfTranposed.values[11]
    print(row3)

    dists = dfTranposed.values[4:12][:, 5]
    print(dists)

    totalDistanceLeg = np.sum(dfTranposed.values[4:12][:, 5])
    avgSpeedLeg = np.mean(dfTranposed.values[4:12][:, 8])
    avgBearing = np.mean(dfTranposed.values[4:12][:, 4])
    print(str(totalDistanceLeg) + " miles")
    print(str(avgSpeedLeg) + " knots")
    print(str(avgBearing) + " degrees")
    distKM = totalDistanceLeg * 1.6
    # geopy.distance.distance((0,105), (-3.75,115.64)).km
    print(str(distKM) + " km")

    geodesic = pyproj.Geod(ellps='WGS84')

    lats = row0[1]
    latsNumbers = row0[1].split(" ")[0]
    latsStart = float(latsNumbers.split("-")[0] + "." + latsNumbers.split("-")[1])
    latDirStart = lats.split(" ")[1]
    latsStart = -latsStart if latDirStart == 'S' else latsStart

    lats = row3[1]
    latsNumbers = row3[1].split(" ")[0]
    latsEnd = float(latsNumbers.split("-")[0] + "." + latsNumbers.split("-")[0])
    latDirEnd = lats.split(" ")[1]

    print("Lats Start:" + str(latsStart) + " " + latDirStart)
    print("Lats End:" + str(latsEnd) + " " + latDirEnd)
    # latsStart = latsStart if  latsStart < latsEnd else latsEnd

    lons = row0[2]
    lonsNumbers = row0[2].split(" ")[0]
    lonDirStart = lons.split(" ")[1]
    print(lonDirStart)
    lonsStart = float(lonsNumbers.split("-")[0] + "." + lonsNumbers.split("-")[1])

    lons = row3[2]
    lonsNumbers = row3[2].split(" ")[0]
    lonDirEnd = lons.split(" ")[1]
    lonsEnd = float(lonsNumbers.split("-")[0] + "." + lonsNumbers.split("-")[1])
    lonsEnd = -lonsEnd if lonDirEnd == 'W' else lonsEnd

    print("Lons Start:" + str(lonsStart) + " " + lonDirStart)
    print("Lons End:" + str(lonsEnd) + " " + lonDirEnd)

    # lonsStartNew = lonsStart if  lonsStart < lonsEnd else lonsEnd
    # lonsEndNew = lonsEnd if  lonsEnd > lonsStart else lonsStart

    # print(lats)

    avgSpeed = avgSpeedLeg

    distanceBetweenLatLon = 20
    R = 6371  # EarthRadius in Km
    coords = []
    lats = []
    lons = []
    totalDistance = 0
    bears = []
    # avgSpeedLeg * 2 #init distance 20km
    lats.append(latsStart)
    lons.append(lonsStart)
    bears.append(dfTranposed.values[4:12][0, 4])
    bearings = [float(k) for k in list(dfTranposed.values[4:12][:, 4])]
    print(bearings)
    distancesKM = [float(k) * 1.6 for k in list(dfTranposed.values[4:12][:, 5])]
    latsRanges = [0, -2, -4, -3]
    print(np.sum(distancesKM))
    i = 1
    # flag==False
    bearIndex = -1
    bearIndices = [0]
    lat2 = 0
    lat2 = -2
    bearingNew = dfTranposed.values[4:12][0, 4]
    print("Init bearing:" + str(bearingNew))
    print(latsStart)
    latsChange = [0, 1.13, 6.08, 11.02, 15.29, 20.22, 24.56, 26.26]
    lonsChange = [0, 119.13, 119.54, 120.27, 119.47, 119.48, 120.11, 120.13]

    while totalDistance <= distKM:

        changeBearingFlag = False

        if totalDistance >= 0 and totalDistance <= distancesKM[0]:
            bearingInit = bearings[0]
            bearIndex = 0
            flag = True
            nextLatTarget = latsChange[bearIndex + 1]
            nextLonTarget = lonsChange[bearIndex + 1]

        elif totalDistance > distancesKM[0] and totalDistance <= np.sum(distancesKM[0:2]):
            bearingInit = bearings[1]
            bearIndex = 1
            nextLatTarget = latsChange[bearIndex + 1]
            nextLonTarget = lonsChange[bearIndex + 1]

        elif totalDistance > np.sum(distancesKM[0:2]) and totalDistance <= np.sum(distancesKM[0:3]):
            bearingInit = bearings[2]
            bearIndex = 2
            nextLatTarget = latsChange[bearIndex + 1]
            nextLonTarget = lonsChange[bearIndex + 1]

        elif totalDistance > np.sum(distancesKM[0:3]) and totalDistance <= np.sum(distancesKM[0:4]):
            bearingInit = bearings[3]
            bearIndex = 3
            nextLatTarget = latsChange[bearIndex + 1]
            nextLonTarget = lonsChange[bearIndex + 1]

        elif totalDistance > np.sum(distancesKM[0:4]) and totalDistance <= np.sum(distancesKM[0:5]):
            bearingInit = bearings[4]
            bearIndex = 4
            nextLatTarget = latsChange[bearIndex + 1]
            nextLonTarget = lonsChange[bearIndex + 1]

        elif totalDistance > np.sum(distancesKM[0:5]) and totalDistance <= np.sum(distancesKM[0:6]):
            bearingInit = bearings[5]
            bearIndex = 5
            nextLatTarget = latsChange[bearIndex + 1]
            nextLonTarget = lonsChange[bearIndex + 1]

        elif totalDistance > np.sum(distancesKM[0:6]) and totalDistance <= np.sum(distancesKM[0:7]):
            bearingInit = bearings[6]
            bearIndex = 6
            nextLatTarget = latsChange[bearIndex + 1]
            nextLonTarget = lonsChange[bearIndex + 1]

        elif totalDistance > np.sum(distancesKM[0:7]) and totalDistance <= np.sum(distancesKM[0:8]):
            bearingInit = bearings[7]
            bearIndex = 7

        bearIndices.append(bearIndex)

        ##check to change append waypoint in lat lon lists

        # print(i)
        # print(len(bearIndices))
        if bearIndices[i] != bearIndices[i - 1] or i == 1:
            # print(i)
            bearingNew = bearingInit
            print(bearIndex)
            # bearingNew = bearings[bearIndex-1]
            changeBearingFlag = True

            # if latsChange[bearIndex] == latsChange[len(latsChange)-1]: break
            print(str(lats[i - 1]) + " " + "Init " + str(bearingNew))

        if latsChange[bearIndex] == 24.56:
            bearingNew = -179
        if bearIndex > 0:  ##if its past the first NOON
            if bearIndices[i] != bearIndices[i - 1]:
                # print(i)
                f = 0
                #if lats[i - 1] != latsChange[bearIndex-1]:
                lats.append(latsChange[bearIndex])
                lons.append(lonsChange[bearIndex])

                bearIndices.append(bearIndex)
                bears.append(bearingNew)
                i += 1
                if bearIndex == len(bearings)-2 or bearIndex == len(bearings)-1:
                    bearingNew = -179
                    #lats.append(latsEnd)
                    #lons.append(lonsEnd)
                    #bears.append(bearings[len(bearings)-1])
                    #distanceBetweenLatLon = 20
                    #break
                #if latsChange[bearIndex] == latsEnd: break



        latInit, lonInit, = lats[i - 1], lons[i - 1]

        if bearIndex < len(bearings) - 1:
            bearingNew = bearingNew + 1 if bearings[bearIndex] < bearings[bearIndex + 1] else bearingNew - 1
        # else:
        # lats.append(-3.9)
        # lons.append(115.6)
        # break

        destination = distance.distance(kilometers=distanceBetweenLatLon).destination((latInit, lonInit), bearingNew)
        lat2, lon2 = destination.latitude, destination.longitude

        if bearIndex == len(bearings) - 2 or bearIndex == len(bearings) - 1:
            fwd_azimuth, back_azimuth , dist = geodesic.inv(lonsEnd, latsEnd, lonInit, latInit)
        #if abs(lat2) < abs(latInit) and
        # distanceBetweenLatLon = geopy.distance.distance((latInit,lonInit),(lat2,lon2)).km
        # timeBetweenLatLon = distanceBetweenLatLon / (avgSpeed * 1.852)

        # print(np.round(timeBetweenLatLon))
        # print(totalDistance)
        if nextLatTarget < latsChange[bearIndex]:
            # print(latsChange[bearIndex])
            if lat2 > nextLatTarget and lon2 < nextLonTarget:
                lats.append(lat2)
                lons.append(lon2)
                totalDistance += distanceBetweenLatLon
            else:

                lat2, lon2 = nextLatTarget, nextLonTarget
                lats.append(lat2)
                lons.append(lon2)
                distanceNew = geopy.distance.distance((latInit, lonInit), (lat2, lon2)).km
                totalDistance = np.sum(distancesKM[0:bearIndex + 1]) + 1
                print(totalDistance)
        else:
            if lat2 < nextLatTarget and lon2 < nextLonTarget:
                lats.append(lat2)
                lons.append(lon2)
                totalDistance += distanceBetweenLatLon
            else:

                lat2, lon2 = nextLatTarget, nextLonTarget
                lats.append(lat2)
                lons.append(lon2)
                distanceNew = geopy.distance.distance((latInit, lonInit), (lat2, lon2)).km
                totalDistance = np.sum(distancesKM[0:bearIndex + 1]) + 1
                print(totalDistance)

        bears.append(bearingNew)
        i += 1

    with open('data/MODERNA/SPETSES SPIRIT/Coords/SPETSES_Coor_Bearings.csv', mode='w') as data:
        data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow(['Latitude', 'Longitude', 'Bearing'])
        for i in range(0, len(lats)):
            data_writer.writerow(
                [lats[i], lons[i], bears[i]])

    with open('data/MODERNA/SPETSES SPIRIT/Coords/SPETSES_Coor1.csv', mode='w') as data:
        data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow(['Latitude', 'Longitude'])
        for i in range(0, len(lats)):
            data_writer.writerow(
                [lats[i], lons[i]])
    drawRoute()

def drawRoute():
    geoviz = leafmap.Map()

    coords = pd.read_csv('data/MODERNA/SPETSES SPIRIT/Coords/SPETSES_Coor1.csv').values
    jsonCoords = {'type': "FeatureCollection",
                  "features": [{"type": "Feature",
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [],
                                    "properties": {
                                        "color": "red",
                                        "slug": "adria-1"
                                    }

                                }}]}
    states = {
        "type": "Feature",
        "properties": {"route": "",
                       },
        "geometry": {
            "type": "MultiLineString",
            "coordinates": [[

            ]],
            "properties": {
                "color": "ff0000",
                "slug": "adria-1"
            }
        }
    }

    # print(np.typename(coords[0][0]))
    for i in range(0, len(coords)):
        states['geometry']['coordinates'][0].append([float(coords[i][1]), float(coords[i][0])])

    with open('./data/MODERNA/SPETSES SPIRIT/SPETSES_SPIRITcoords.json', 'w') as json_file:
        json.dump(states, json_file)
    Map = leafmap.Map(center=[0, 0], zoom=2)
    Map.add_geojson('./data/MODERNA/SPETSES SPIRIT/SPETSES_SPIRITcoords.json', layer_name='Cable lines')
    #Map.s


def main():

    extractReproducedRoute()

# ENTRY POINT
if __name__ == "__main__":
    main()

