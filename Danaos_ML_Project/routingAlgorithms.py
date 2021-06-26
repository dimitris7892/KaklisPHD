import numpy.linalg
from scipy.spatial import Delaunay
from holoviews.operation.datashader import datashade, dynspread
from global_land_mask import globe
import math
import scipy as scipy
import pyproj
import numpy as np
import holoviews as hv
from datashader.utils import lnglat_to_meters
from holoviews.operation.datashader import datashade
from geoviews.tile_sources import EsriImagery
import matplotlib
import param, panel as pn
from colorcet import palette
import pyodbc
import csv
import dask.dataframe as dd, geoviews as gv, cartopy.crs as crs
from bokeh.io import export_png
import shapely.geometry
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely.geometry import Point
#from turfpy.measurement import boolean_point_in_polygon
#from geojson import Point, Polygon, Feature
import datetime as datetime
from fastapi import APIRouter
import requests
import json
import os
import geopy.distance
import io
from mpl_toolkits.basemap import Basemap
from pyproj import Proj, transform
from numpy import ones, vstack
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes

class Routing:

    bm = Basemap(resolution='c')
    hv.extension('bokeh', 'matplotlib')
    geodesic = pyproj.Geod(ellps='WGS84')

    totalSailingTimeHours = 0
    initialPoint = 0
    endingPoint = 0
    meandDraft = 0
    w_min = 0
    c1 = 0
    c2 = 0
    w_max = 1
    it_max = 0
    distanceAB = 0
    vmin = 0
    vmax = 0

    def PSO(self):

        pass

    def ConvertMSToBeaufort(self,ws):
        wsB = 0
        if ws >= 0 and ws < 0.2:
            wsB = 0
        elif ws >= 0.3 and ws < 1.5:
            wsB = 1
        elif ws >= 1.6 and ws < 3.3:
            wsB = 2
        elif ws >= 3.4 and ws < 5.4:
            wsB = 3
        elif ws >= 5.5 and ws < 7.9:
            wsB = 4

        elif ws >= 8 and ws < 10.7:
            wsB = 5
        elif ws >= 10.8 and ws < 13.8:
            wsB = 6
        elif ws >= 13.9 and ws < 17.1:
            wsB = 7
        elif ws >= 17.2 and ws < 20.7:
            wsB = 8
        elif ws >= 20.8 and ws < 24.4:
            wsB = 9
        elif ws >= 24.5 and ws < 28.4:
            wsB = 10
        elif ws >= 28.5 and ws < 32.6:
            wsB = 11
        elif ws >= 32.7:
            wsB = 12
        return wsB

    def extractJSON_TestData(self,imo,vessel,stw,ws,wd,swell,draft):

        laddenJSON = '{}'
        json_decoded = json.loads(laddenJSON)
        json_decoded['ConsumptionProfile_Dataset'] = {"vessel_code": str(imo), 'vessel_name': vessel,
                                                         "dateCreated": datetime.date.today().strftime("%d/%m/%Y"), "data": []}

        item = {"draft": np.round(draft, 2), 'stw': np.round(stw, 2),
                       "windBFT": float(np.round(ws, 2)),
                       "windDir": np.round(wd, 2), "swell": np.round(swell, 2),
                       "cons": -1}
        json_decoded['ConsumptionProfile_Dataset']["data"].append(item)
               # json_decoded['ConsumptionProfile']['consProfile'].append(outerItem)

        with open('/home/dimitris/Desktop/gitlabDanaos/danaos-deep-learning/ConsModelComparison/ConsModelComparison_webapi/bin/Debug/net5.0/'
                  '/TestData_' + vessel + '_.json', 'w') as json_file:

               json.dump(json_decoded, json_file)



    def pointInRect(self,point, rect):
        x1, y1, w, h = rect
        x2, y2 = x1 + w, y1 + h
        x, y = point
        if (x1 <= x and x <= x2):
            if (y1 <= y and y <= y2):
                return True
        return False

    def calculateConsPavlos(self, stw, ws, wd, swell, draft):

        self.extractJSON_TestData('9484948','EXPRESS ATHENS', stw, ws, wd, swell, draft)
        router = APIRouter()

        @router.get("/")
        def get_ModelComparison(interpolation):
            response = requests.get("https://localhost:5001/CompareConsumptionModels/?interpolation=" + interpolation,
                                    verify=False)
            return response.text

        modelComp =  get_ModelComparison("true")
        json_dict = json.load(io.StringIO(modelComp))
        df = pd.concat([pd.DataFrame(json_dict)])
        return df['stats_Avg']

    def getCoastLineEurope(self):

        shape = gpd.read_file("./coastlines/Europe_coastline_raw_rev2017.shp")
        print(shape.head())

        inProj = Proj(init='epsg:3857')
        outProj = Proj(init='epsg:4326')
        lats = []
        lons = []
        for x in shape['geometry']:
            coords = np.array(x)
            for i in range(0, len(coords)):
                x1, y1 = coords[i][0], coords[i][1]
                x2, y2 = transform(inProj, outProj, x1, y1)
                # print(x2, y2)
                lats.append(x2)
                lons.append(y2)


        with open('./EuropeCoastLine.csv', mode='w') as dataw:
            data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['Latitude', 'Longitude'])
            for i in range(0, len(lats)):
                data_writer.writerow(
                    [lats[i], lons[i]])

    def barPLots(self, lons, optVar, orVar ,plot):

        font = {'family': 'normal',
                'weight': 'normal',
                'size': 23}

        plt.rc('font', **font)

        if plot=='foc':
            lons =  np.round(lons,3)
            df = pd.DataFrame({
                'Different Sailing Positions': lons,
                'Optimal Fuel Consumption (MT/day)': optVar,
                'Original Fuel Consumption (MT/day)': orVar
            })
            #fig, ax = plt.subplots()
            axisXBarLabels=[
                str(lons[0])[1:]+u"\N{DEGREE SIGN}" +'W',
                str(lons[4])[1:]+u"\N{DEGREE SIGN}" +'W',
                str(lons[8])[1:]+u"\N{DEGREE SIGN}" +'W',
                str(lons[13])[1:]+u"\N{DEGREE SIGN}" +'W',
                str(lons[17])[1:]+u"\N{DEGREE SIGN}" +'W',
                str(lons[len(lons)-1])[1:]+u"\N{DEGREE SIGN}" +'W']
            # plotting graph
            df.plot(x="Different Sailing Positions", y=["Optimal Fuel Consumption (MT/day)",
                    "Original Fuel Consumption (MT/day)"], kind="bar")

            plt.xticks([0,4,8,12,16,20],axisXBarLabels)
            plt.xticks(rotation=360)
            plt.ylabel("Fuel Oil Consumption (MT/day)")
            plt.show()
            x = 0
        else:
            lons = np.round(lons, 3)
            df = pd.DataFrame({
                'Different Sailing Positions': lons,
                'Optimal Sailing Speed (knots)': optVar,
                'Original Sailing Speed (knots)': orVar
            })
            # fig, ax = plt.subplots()
            axisXBarLabels = [
                str(lons[0])[1:] + u"\N{DEGREE SIGN}" + 'W',
                str(lons[4])[1:] + u"\N{DEGREE SIGN}" + 'W',
                str(lons[8])[1:] + u"\N{DEGREE SIGN}" + 'W',
                str(lons[13])[1:] + u"\N{DEGREE SIGN}" + 'W',
                str(lons[17])[1:] + u"\N{DEGREE SIGN}" + 'W',
                str(lons[len(lons) - 1])[1:] + u"\N{DEGREE SIGN}" + 'W']
            # plotting graph
            df.plot(x="Different Sailing Positions", y=["Optimal Sailing Speed (knots)",
                                                        "Original Sailing Speed (knots)"], kind="bar", color=['b','g'])

            plt.xticks([0, 4, 8, 12, 16, 20], axisXBarLabels)
            plt.xticks(rotation=360)
            plt.ylabel("Sailing Speed (knots)")
            plt.show()
            x = 0

    def evaluateOptimalRoute(self):

        originalRoute = pd.read_csv('./data/DANAOS/EXPRESS ATHENS/EXPRESS ATHENSCoor1.csv')
        optimalRoute = pd.read_csv('./optimalRoute.csv')

        latsOR = originalRoute['Latitude'].values
        lonsOR = originalRoute['Longitude'].values

        latsOptR = optimalRoute['Latitude'].values
        lonsOptR = optimalRoute['Longitude'].values

        optSpeed = []
        optFoc = []
        orFoc = []
        orSpeed = []
        evalLons = []
        for i in range(0,len(optimalRoute)):
            optLat = latsOptR[i]
            optLon = lonsOptR[i]
            for k in range(0,len(latsOR)):
                if np.round(latsOR[k],2) == np.round(optLat,2):
                    optSpeed.append(optimalRoute['speed'][i])
                    optFoc.append(optimalRoute['foc'][i])

                    orSpeed.append(originalRoute['speed'][k])
                    orFoc.append(originalRoute['foc'][k])
                    evalLons.append(optLon)
                    break
        self.barPLots(evalLons, optFoc, orFoc,'foc')
        self.barPLots(evalLons, optSpeed, orSpeed,'speed')

    def findAdjacentPolygons(self,initialPolygon,polygons,j):

        ##findAdjacentPolygonsToAnother
        x1 = initialPolygon.bounds[0]
        y1 = initialPolygon.bounds[1]
        x2 = initialPolygon.bounds[2]
        y2 = initialPolygon.bounds[3]
        adjacentList = []
        for i in range(0, len(polygons)):
            if i != j:
                x1C = polygons[i].bounds[0]
                y1C = polygons[i].bounds[1]
                x2C = polygons[i].bounds[2]
                y2C = polygons[i].bounds[3]
                if ((x1, y1) == (x1C, y2C) and (x2, y1) == (x2C, y2C)) or \
                        ((x1, y1) == (x2C, y1C) and (x1, y2) == (x2C, y2C)) or \
                        ((x1, y2) == (x1C, y1C) and (x2, y2) == (x2C, y1C)) or \
                        ((x2, y2) == (x1C, y2C) and (x2, y1) == (x1C, y1C)) or \
                        ((x1 , y1 ) == (x2C , y2C)) or (( x2 , y1 ) == (x1C , y2C)) or ((x1 , y2 ) == (x2C , y1C)) or ((x2 , y2 ) == (x1C , y1C)):
                    adjacentList.append(polygons[i])

        return adjacentList

    def getStraightLineBetweenAB(self,A,B):

        points = [A, B]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        print("Line Solution is y = {m}x + {c}".format(m=m, c=c))
        return m, c

    def getPointsOfInitialRoute(self,initialPoint, endingPoint, m, c):

        pointsInLine = []
        lonsRange = np.linspace(initialPoint[0], endingPoint[0], 23)
        for lon in lonsRange:
            lat_ = m * lon + c
            lon_ = lon
            if globe.is_ocean(lat_, lon_):  pointsInLine.append((lat_, lon_))

        pointsInLine = np.array(pointsInLine[1:])
        with open('./initialStraightLineRoute.csv', mode='w') as dataw:
            data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['Latitude', 'Longitude'])
            for i in range(0, len(pointsInLine)):
                data_writer.writerow(
                    [pointsInLine[i][0], pointsInLine[i][1]])

        return pointsInLine

    ##step1 of optimization process. Particles initilization
    def initializeParticles(self,grid,route):

        self.distanceAB = route[0][0]
        ## find particle that starting point A of the route is inside and construct rectangles of grid
        draft = route[0][3]
        self.meandDraft = draft

        ###construct rectangles of grid
        polygons = []
        for k in range(0,55):
            for i in range(k,len(grid)-56,55):

                lons_vect = []
                lats_vect = []

                lons_vect.append([grid[i][1], grid[i + 55][1], grid[i + 56][1], grid[i + 1][1]])
                lats_vect.append([grid[i][0], grid[i + 55][0], grid[i + 56][0], grid[i + 1][0]])

                lons_lats_vect = np.column_stack((lons_vect[0], lats_vect[0]))  # Reshape coordinates
                polygon = Polygon(lons_lats_vect)  # create polygon
                x, y = polygon.exterior.xy
                #plt.plot(x, y)
                #plt.show()
                polygons.append(polygon)

        ###
        self.initialPoint = Point(route[0][1], route[0][0])
        self.endingPoint = Point(route[len(route)-10][1], route[len(route)-10][0])
        self.distanceAB = geopy.distance.distance((route[0][0], route[0][1]),(route[len(route)-1][0], route[len(route)-1][1])).km
        initialDt = route[0][2]
        lastDt = route[len(route)-1][2]
        elapsedTime = (datetime.datetime.strptime(route[len(route)-1][2] , '%Y-%m-%d %H:%M') - datetime.datetime.strptime(route[0][2] , '%Y-%m-%d %H:%M'))
        self.totalSailingTimeHours = divmod(elapsedTime.total_seconds(), 60)[0] / 60

        print(self.initialPoint)
        ##find polygon that initial point A is into
        for p in polygons:
            if p.contains(self.initialPoint):
                print(p.bounds)
                pol = gpd.GeoSeries(p)
                pol.plot()

                #plt.show()
                initialPolygon = p
                j = polygons.index(p)
                print("TRUE")  # check if a point is in the polygon
            if p.contains(self.endingPoint):
                print(p.bounds)
                endingPolygon = p
                e = polygons.index(p)
                print("TRUE")

        ##get adjacent polygons to initial polygon of starting point A

        adjacentPolygons = self.findAdjacentPolygons(initialPolygon,polygons,j)

        ### find and remove polygons from adjacency list that their centroid is close to land with resolution 1km
        for p in adjacentPolygons:
            lon , lat = tuple(np.array(p.centroid))
            is_in_land = globe.is_land(lat, lon)
            #island = bm.is_land(lon,lat)
            if is_in_land==False :
                adjacentPolygons.remove(p)

        ### find and remove polygons from all the polygons that construct the grid that their centroid is close to land  with resolution 1km
        polygonsNew = []
        for p in polygons:
            lon, lat = tuple(np.array(p.centroid))
            #is_in_ocean = globe.is_ocean(np.round(lat),np.round(lon))
            is_in_ocean_ = globe.is_ocean(lat, lon)
            #island = bm.is_land(lon,lat)
            if  is_in_ocean_==True:
                polygonsNew.append(p)


        ##find equation of straght line that connects points A -> B
        initialPoint = tuple(np.array(self.initialPoint))
        endingPoint = tuple(np.array(self.endingPoint))
        m, c  = self.getStraightLineBetweenAB(initialPoint, endingPoint)

        ##get points(lat,lon) with M-2 dimensions corresponing to initial route (particles)
        initialPointsRoute = self.getPointsOfInitialRoute(initialPoint, endingPoint, m, c)
        ##find polygons that each initial particle belongs to
        ########################
        m_1_polygons=[]
        for p in initialPointsRoute:
            p = Point(p[1],p[0])
            for pol in polygons:
                if pol.contains(p) or p.touches(pol):
                    m_1_polygons.append(pol)
        ##initialize particles in grid and assign velocities randomly from a uniform dist [velMin,velMax]
        polygons = polygonsNew ##polygonsNew conttains the polygons thath their centroid is in ocean and not in land

        adjacentPolygons.insert(0,initialPolygon)
        particlesPos={'particles':[]}
        lonInit, latInit = initialPoint[0] , initialPoint[1]
        dtInit = datetime.datetime.strptime(route[0][2], '%Y-%m-%d %H:%M')
        speedInit = route[0][4]
        draftInit = route[0][3]
        item = {"polygon":initialPolygon,"lat": latInit, "lon": lonInit, 'speed': speedInit, "dt": dtInit,
                "draft": draftInit,'hoursFromPrevPoint':0, }
        #particlesPos['particles'].append(item)

        for i in range(0,len(initialPointsRoute)):

                lon , lat = tuple([initialPointsRoute[i][1],initialPointsRoute[i][0]])
                ##random velocity initialization
                speed = np.random.uniform(13,15)
                dist = geopy.distance.distance((latInit,lonInit),(lat,lon)).km
                timeToNextParticle = dist / speed
                timeToNextParticle = timeToNextParticle * 0.54 # hours

                fwd_azimuth, back_azimuth, distance = self.geodesic.inv(lonInit, latInit, lon, lat)
                dt = route[0][2] if i==0 else dtInit +  datetime.timedelta(hours=timeToNextParticle)
                item={"polygon":m_1_polygons[i],"lat":lat,"lon":lon,'speed':speed , "dt":dt,"draft": draft ,
                      'hoursFromPrevPoint': timeToNextParticle , "bearing": fwd_azimuth }


                particlesPos['particles'].append(item)
        #particlesPos['particles'][0]["bearing"] = fwd_azimuth
        self.evaluateAdjacentParticles(particlesPos)

    #step2
    def evaluateAdjacentParticles(self,particlesPos):

        initLat = particlesPos['particles'][0]['lat']
        initLon = particlesPos['particles'][0]['lon']
        initDt = particlesPos['particles'][0]['dt']
        initSpeed = particlesPos['particles'][0]['speed']
        initDraft = particlesPos['particles'][0]['draft']
        windSpeed, relWindDir, swellSWH, relSwelldir, wavesSWH, relWavesdDir, combSWH, relCombWavesdDir, currentSpeed, relCurrentDir = \
            self.mapWeatherData(initDt, np.round(initLat), np.round(initLon))
        windSpeedbft = self.ConvertMSToBeaufort(windSpeed)
        initFocParticle = self.calculateConsPavlos(initSpeed, windSpeedbft, relWindDir, swellSWH, initDraft)
        particlesPos['particles'][0]['totalFoc'] = initFocParticle[0]


        ###calculate FOC for each particle initially generated
        focsParticles = []
        for i in range(0,len(particlesPos['particles'])):#len(particlesPos['particles'])
            lat = particlesPos['particles'][i]['lat']
            lon = particlesPos['particles'][i]['lat']
            dt = particlesPos['particles'][i]['dt']
            speed = particlesPos['particles'][i]['speed']
            draft = particlesPos['particles'][i]['draft']
            windSpeed, relWindDir, swellSWH, relSwelldir, wavesSWH, relWavesdDir, combSWH, relCombWavesdDir, currentSpeed, relCurrentDir = \
                self.mapWeatherData(dt, np.round(lat), np.round(lon))
            windSpeedbft = self.ConvertMSToBeaufort(windSpeed)
            focParticle = self.calculateConsPavlos(speed,windSpeedbft,relWindDir,swellSWH,draft)[0]
            #totalFoc = focParticle #+ initFocParticle
            particlesPos['particles'][i]['totalFoc'] = focParticle
            #np.random.randint(50,120)
            particlesPos['particles'][i]['local_best'] = particlesPos['particles'][i]
            focsParticles.append(particlesPos['particles'][i]['totalFoc'])
        ###find best local position of particles by taking the min of cost function on each adjacent list of polygons
        ##init

        globalBestParticleInd = focsParticles.index(min(focsParticles))
        globalBestParticle = particlesPos['particles'][globalBestParticleInd]
        count = 0

        lats = []
        lons = []
        #iterations until gonvergence
        #t=0
        optimalParticles = particlesPos['particles']
        ##set up hyperparameters for inertia weight update
        self.it_max = 15
        self.w_max = 1
        self.w_min = 0.7
        self.c1 = 0.5
        self.c2 = 0.5
        self.vmax = 20
        self.vmin = 11
        newTotalSailingTime = 100
        t=0
        thresholdSpeeds = [1]
        while t < self.it_max :#or  newTotalSailingTime > self.totalSailingTimeHours or len(thresholdSpeeds)>0
            if t > 0 :
                optimalParticles, globalBestParticle = self.selectOptimalParticles(particlesPos, t)

            ###update location, velocity
            particlesPos, newTotalSailingTime, speedsArray  = self.velocityLocationUpdate(optimalParticles, globalBestParticle, t)
            thresholdSpeeds = [k for k in speedsArray if k > self.vmax or k < self.vmin]
            #print(thresholdSpeeds)
            #print(newTotalSailingTime)
            #while newTotalSailingTime > self.totalSailingTimeHours or len(thresholdSpeeds)>0:
                #particlesPos, newTotalSailingTime, speedsArray = self.velocityLocationUpdate(optimalParticles, globalBestParticle, t)
            t += 1
        print("optimal route sailing time: " + str(newTotalSailingTime))
        print("original route sailing time: " + str(self.totalSailingTimeHours))
        ### save optimal route #########
        lats=[]
        lons=[]
        speeds = []
        focs = []
        for i in range(0,len(optimalParticles)):
            lats.append(optimalParticles[i]['local_best']['lat'])
            lons.append(optimalParticles[i]['local_best']['lon'])
            speeds.append(optimalParticles[i]['local_best']['speed'])
            focs.append(optimalParticles[i]['local_best']['totalFoc'])
        with open('./optimalRoute.csv', mode='w') as dataw:
                data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(['Latitude', 'Longitude','speed','foc'])
                for i in range(0, len(lats)):
                    data_writer.writerow(
                        [lats[i],lons[i],speeds[i],focs[i]])

        return

    ##step 2 select the optimal values of particle depending on cost function
    def selectOptimalParticles(self,particlesPos,iteration):

        ##caluclate FOC (cost function)  to determine local best and global best for particles
        totalFocLB = 0
        totalFocNP = 0
        focsParticles = []

        for i in range(0,len(particlesPos)):

                newpos = particlesPos[i]['newpos']
                lat = newpos['lat']
                lon = newpos['lon']
                dt = newpos['dt']
                speed = newpos['speed']
                draft = self.meandDraft
                windSpeed, relWindDir, swellSWH, relSwelldir, wavesSWH, relWavesdDir, combSWH, relCombWavesdDir, currentSpeed, relCurrentDir = \
                self.mapWeatherData(dt, np.round(lat), np.round(lon))
                windSpeedbft = self.ConvertMSToBeaufort(windSpeed)
                newPosFocParticle = self.calculateConsPavlos(speed,windSpeedbft,relWindDir,swellSWH,draft)[0]
                newpos['totalFoc'] = newPosFocParticle
                localBestFoc = particlesPos[i]['local_best']['totalFoc']
                totalFocLB += localBestFoc
                totalFocNP += newPosFocParticle
                focsParticles.append(localBestFoc)
                '''if newPosFocParticle < localBestFoc:
                    particlesPos[i]['local_best'] = newpos
                    particlesPos[i]['local_best']['totalFoc'] = newPosFocParticle'''



        if totalFocNP <= totalFocLB:
            for i in range(0, len(particlesPos)):
                newpos = particlesPos[i]['newpos']
                particlesPos[i]['local_best'] = newpos
                focLB = particlesPos[i]['local_best']['totalFoc']
                focsParticles[i]= focLB


        '''for i in range(0,len(particlesPos)):

                local_best = particlesPos[i]['local_best']
                focLB = local_best['totalFoc']
                #totalFoc += focLB
                focsParticles.append(focLB)'''

        globalBestParticleInd = focsParticles.index(min(focsParticles))
        globalBestParticle = particlesPos[globalBestParticleInd]
        return particlesPos , globalBestParticle

    ##step3
    def velocityLocationUpdate(self, particlesPos, globalBestParticle, it_curr):

            w = 0.8
            r1 = np.random.uniform(0, 1)
            r2 = np.random.uniform(0, 1)
            totalSpeed = 0
            newSpeeds = []
            newTotalSailingTime = 0
            for i in range(0, len(particlesPos)):
                locPrev = self.initialPoint if i == 0 else (
                particlesPos[i - 1]['lat'], particlesPos[i - 1]['lon'])
                lat = particlesPos[i]['lat']
                lon = particlesPos[i]['lon']
                dt = particlesPos[i]['dt']
                timeFromPrev = particlesPos[i]['hoursFromPrevPoint']
                speed = particlesPos[i]['speed']

                bearing = particlesPos[i]['bearing']
                vectorSpeed = [speed, bearing]
                locCurr = (lat, lon)
                local_best = particlesPos[i]['local_best']
                p_best = np.array([local_best['lat'] ,local_best['lon']])
                x_k = np.array([lat, lon])
                g_best = np.array([globalBestParticle['lat'], globalBestParticle['lon']])

                g_best_speed = globalBestParticle['speed']
                g_best_bearing = globalBestParticle['bearing']
                g_best_vector_speed = np.array([g_best_speed,g_best_bearing])

                p_best_speed = local_best['speed']
                p_best_bearing = local_best['bearing']
                p_best_vector_speed = np.array([p_best_speed, p_best_bearing])

                R = 6371  # EarthRadius in Km

                ##convert decimal degrees to cartesian

                ###############################

                #vectorSpeed = np.array([vectorSpeed[0] * 0.514, math.radians(vectorSpeed[1])]) ##m/s / radians
                ##inertia weight update
                w = self.w_max - (self.w_max - self.w_min) * (it_curr / self.it_max)

                ##vector speed update

                #speed = speed *  1.15 #miles/h
                speed = speed * 1.85  # km/h
                #speed = speed / 32.39  # km/min
                #speed = speed * 0.514 #m/s
                x_speed_curr = np.cos(bearing) * speed
                y_speed_curr = np.sin(bearing) * speed

                p_best_meters = np.array([p_best[0] *  111 , p_best[1] * 111])
                g_best_meters = np.array([g_best[0] *  111, g_best[1] *  111])
                x_k_meters = np.array([x_k[0] *  111 , x_k[1] *  111 ])

                fwd, bear1, distance = self.geodesic.inv(p_best[1], p_best[0], x_k[1], x_k[0])
                fwd, bear2, distance = self.geodesic.inv(g_best[1], g_best[0], x_k[1], x_k[0])

                x_speed_curr1 = np.cos(bear1) * speed
                y_speed_curr1 = np.sin(bear1) * speed

                x_speed_curr2 = np.cos(bear2) * speed
                y_speed_curr2 = np.sin(bear2) * speed

                e1 = np.array([np.cos(bear1), np.sin(bear1)])
                    #np.array([x_speed_curr1, y_speed_curr1]) / np.sqrt(x_speed_curr1 ** 2 + y_speed_curr1 ** 2)
                e2 = np.array([np.cos(bear2), np.sin(bear2)])
                    #np.array([x_speed_curr2, y_speed_curr2]) / np.sqrt(x_speed_curr2 ** 2 + y_speed_curr2 ** 2)
                v1 =  (geopy.distance.distance(p_best,x_k).km ) * e1
                v2 = (geopy.distance.distance(g_best, x_k).km) * e2

                decomposedVectorSpeed = [x_speed_curr, y_speed_curr]
                vectorSpeed = [speed, bearing]
                #speedNew = np.array([k * w for k in decomposedVectorSpeed ]) + \
                           #(self.c1 * r1 * (p_best_meters - x_k_meters) ) + \
                           #(self.c2 * r2 * (g_best_meters - x_k_meters) )
                speedNew = np.array([k * w for k in decomposedVectorSpeed]) + \
                    (self.c1 * r1 * v1) + \
                    (self.c2 * r2 * v2)

                speedNewValue = np.sqrt(speedNew[0] ** 2 + speedNew[1] ** 2)
                speedNewBearing = np.arctan(speedNew[1]/speedNew[0])
                speedNewBearing = speedNewBearing
                speedNewComposed = [speedNewValue ,speedNewBearing]


                #locCurrXY = (lat *  111, lon *  111)

                locNewX = x_k_meters[0] + speedNew[0]
                locNewY = x_k_meters[1] + speedNew[1]

                locNewXY = (locNewX,locNewY)
                locNewDecDegrees = (locNewXY[0]/  111  , locNewXY[1]/  111 )

                ###############################################################

                speedMS = speedNew[0] * 0.514
                bearingNew = speedNewComposed[1]
                speedkmS = speedNew[0] * 0.000514
                distance = speedNewComposed[0] #km distance travelled in 1 hours with this speed
                lat2 = math.asin(
                    math.sin(math.pi / 180 * lat) * math.cos(distance / R) + math.cos(math.pi / 180 * lat) * math.sin(
                        distance / R) * math.cos(math.pi / 180 * bearingNew))

                lon2 = math.pi / 180 * lon + math.atan2(
                    math.sin(math.pi / 180 * bearingNew) * math.sin(distance / R) * math.cos(math.pi / 180 * lat),
                    math.cos(distance / R) - math.sin(math.pi / 180 * lat) * math.sin(lat2))

                latNew , lonNew = [180 / math.pi * lat2, 180 / math.pi * lon2]

                #locNew = locNewDecDegrees
                locNew = [latNew, lonNew]

                ##update position & velocity of particle
                speedNew = [(speedNewComposed[0] / 1.85) , speedNewComposed[1]] ##knots degrees

                dt = datetime.datetime.strptime(dt,  '%Y-%m-%d %H:%M') if type(dt) is str else dt
                newDt = dt +  datetime.timedelta(hours=2)
                newPos = {"lat":locNew[0],"lon":locNew[1],'speed':speedNew[0] ,"bearing":speedNew[1] , "dt":newDt ,
                      'hoursFromPrevPoint':2 }
                particlesPos[i]['newpos'] = newPos
                newSpeeds.append(speedNew[0])
                totalSpeed += speedNew[0]

                '''particlesPos[i]['lat'] = locNew[0]
                particlesPos[i]['lon'] = locNew[1]
                particlesPos[i]['speed'] = speedNew
                particlesPos[i]['hoursFromPrevPoint'] = newTime'''

            meanSpeed = totalSpeed /  len(particlesPos)
            meanSpeedKm_h =  meanSpeed * 1.852  #km/h
            newTotalSailingTime = self.distanceAB / meanSpeedKm_h ##hours

            #if newTotalSailingTime > self.totalSailingTimeHours:
                #self.velocityLocationUpdate(particlesPos, globalBestParticle, )


            return particlesPos , newTotalSailingTime , newSpeeds

    def selectOptimalParticlesFromAdjacencyList(self,currentPolygon,particlesPos,polygons):

        listParticles = []
        ###find best local position of particles by taking the min of cost function on each adjacent list of polygons
        # for i in range(0, len(particlesPos['particles'])):
        #currentPolygon = particlesPos['particles'][0]['polygon']
        currPolygonInd = polygons.index(currentPolygon)
        adjacentPolygons = self.findAdjacentPolygons(currentPolygon, polygons, currPolygonInd)
        adjacentPolygonsLons = [np.array(k.centroid)[0] for k in adjacentPolygons]
        currLon = np.array(currentPolygon.centroid)[0]
        adjacentPolygonsLats = [np.array(k.centroid)[1] for k in adjacentPolygons]
        currLat = np.array(currentPolygon.centroid)[1]
        toBeDeletedAdj = []
        for i in range(0,len(adjacentPolygonsLons)):
            if adjacentPolygonsLons[i] > currLon or adjacentPolygonsLons[i]==currLon:
                toBeDeletedAdj.append(adjacentPolygons[i])


        for k in toBeDeletedAdj:
            adjacentPolygons.remove(k)

        listParticles = [k for k in particlesPos['particles'] if k['polygon'] in adjacentPolygons]
        if len(listParticles)==0:
            return []
        listParticlesFocs = [k['totalFoc'] for k in listParticles]
        particlesMinFoc = min(listParticlesFocs)
        minIndexFocAdjacentList = listParticlesFocs.index(particlesMinFoc)
        visitedPolygons = [currentPolygon , listParticles[minIndexFocAdjacentList]['polygon']]
        polygons.remove(visitedPolygons[0])
        #polygons.remove(visitedPolygons[1])

        return listParticles[minIndexFocAdjacentList]


    def mapWeatherData(self,dt ,LAT , LON  ):

        #LAT , LON  = self.convertLATLONfromDegMinSec(lat, lon ,latDir , lonDir)

        ##FORM WEATHER HISTORY DB NAME TO SEARCH
        date = str(dt).split(" ")[0]
        day = int(date.split("-")[2])
        month = date.split("-")[1]
        year = date.split("-")[0]
        hhMM = str(dt).split(" ")[1]
        h = int(hhMM.split(":")[0])
        M = int(hhMM.split(":")[1])
        if int(M) > 30:
            h += 1
        if h % 3 == 2:
            h += 1
        else:
            h = h - h % 3

        day = day + 1 if h == 24 else day
        day = '0' + str(day) if str(day).__len__() == 1 else str(day)
        h = 0 if h == 24 else h

        h = '0' + str(h) if str(h).__len__() == 1 else str(h)
        dbDate = year + month + day + h
        #####END WEATHER HISTORY DB NAME

        connWEATHERHISTORY = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER= 10.2.4.84;'
                                            'DATABASE=WeatherProdData;'
                                            'UID=sa;'
                                            'PWD=sa1!')

        cursor = connWEATHERHISTORY.cursor()


        ##INITIALIZE WEATHER HISTORY DB FIELDS
        # relWindSpeed = 0
        windSpeed = -1
        windDir = -1
        relWindDir = -1
        combSWH = -1
        combDir = -1
        relCombWavesdDir=-1
        swellSWH = -1
        swellDir = -1
        relSwelldir=-1
        wavesSWH =-1
        wavesDir =-1
        relWavesdDir=-1
        currentSpeed=-1
        currentDir=-1
        relCurrentDir=-1


        #####################
        if cursor.tables(table='d' + dbDate, tableType='TABLE').fetchone():

            # IF WEATHER HISTORY TABLE EXISTS => SELECT FIELDS FROM WEATHER HISTORY DB
            cursor.execute(
                'SELECT * FROM d' + dbDate + ' WHERE lat=' + str(LAT) + 'and lon=' + str(
                    LON))

            for row in cursor.fetchall():
                windSpeed = float(row.windSpeed)
                windDir = float(row.windDir)

                combSWH = float(row.combSWH)
                combDir = float(row.combDir)
                swellSWH = float(row.swellSWH)
                swellDir = float(row.swellDir)
                wavesSWH = float(row.wavesSWH)
                wavesDir = float(row.wavesDir)
                currentSpeed = float(row.currentSpeed)
                currentDir = float(row.currentDir)


                relWindDir = windDir#self.getRelativeDirectionWeatherVessel(Vcourse,windDir)
                relSwelldir = swellDir#self.getRelativeDirectionWeatherVessel(Vcourse, swellDir)
                relWavesdDir = wavesDir#self.getRelativeDirectionWeatherVessel(Vcourse, wavesDir)
                relCombWavesdDir = relCombWavesdDir#self.getRelativeDirectionWeatherVessel(Vcourse, combDir)
                relCurrentDir = currentDir#self.getRelativeDirectionWeatherVessel(Vcourse, currentDir)


        return windSpeed ,relWindDir ,swellSWH, relSwelldir , wavesSWH , relWavesdDir ,combSWH , relCombWavesdDir , currentSpeed , relCurrentDir


    def constructGrid(self,data):
        lats = data[:, 6]
        lons = data[:, 7]
        lonmin = min(lons)
        lonmax = max(lons)
        latmin = min(lats)
        latmax = max(lats)
        gridpoints=[]
        stepsizeLon = float(np.linalg.norm(lonmax-lonmin) / 24)
        stepsizeLat = float(np.linalg.norm(latmax-latmin) / 54)
        for x in np.arange(lonmin, lonmax, stepsizeLon):
            for y in np.arange(latmin, latmax, stepsizeLat):
                gridpoints.append([x,y])

        with open('./data/DANAOS/EXPRESS ATHENS/testout.csv', 'w') as of:
            of.write('Latitude,Longitude\n')
            for p in gridpoints:
                of.write('{:f},{:f}\n'.format(p[1], p[0]))

    def PLotTrajectory(self,df, vesselName):
        class VesselTrajectory(param.Parameterized):
            alpha = param.Magnitude(default=1, doc="Map tile opacity")
            cmap = param.ObjectSelector('fire', objects=['fire', 'bgy', 'bgyw', 'bmy', 'gray', 'kbc'])
            location = param.ObjectSelector(default='dropoff', objects=['dropoff', 'pickup'])

            def make_view(self, **kwargs):
                df.to_parquet('./data/DANAOS/EXPRESS ATHENS/EXPRESS ATHENSCoorTest.parquet')
                topts = dict(width=1000, height=800, bgcolor='black', title=vesselName, xaxis=None, yaxis=None,
                             show_grid=True)
                tiles = EsriImagery.clone(crs=crs.GOOGLE_MERCATOR).options(**topts)
                dopts = dict(width=3000, height=1000, x_sampling=1.5, y_sampling=1.5, )

                route = dd.read_parquet('./data/DANAOS/EXPRESS ATHENS/EXPRESS ATHENSCoorTest.parquet').persist()
                pts = hv.Points(route, ['x', 'y'],)


                trips = dynspread(datashade(pts, cmap=palette[self.cmap], **dopts),threshold=1.0)
                #dynspread(datashade(trips))
                layout = tiles * trips
                #layout = figure(title="Basic Title", plot_width=300, plot_height=300)

                #show(hv.render(layout), )
                #export_png(hv.render(layout), filename="plot.png")
                return tiles.options(alpha=self.alpha) * trips

        df.loc[:, 'x'], df.loc[:, 'y'] = lnglat_to_meters(df.Longitude, df.Latitude)

        explorer = VesselTrajectory(name=vesselName)
        # comment out to launch interactive maps in server
        pn.Row(explorer.param, explorer.make_view).servable().show()
        #explorer.make_view()
        #plt.show()

    def createGridMeters(self):

        p_ll = pyproj.Proj(init='epsg:4326')
        p_mt = pyproj.Proj(init='epsg:3857')  # metric; same as EPSG:900913

        # Create corners of rectangle to be transformed to a grid
        sw = shapely.geometry.Point((-5.0, 40.0))
        ne = shapely.geometry.Point((-4.0, 41.0))

        stepsize = 5000  # 5 km grid step size

        # Project corners to target projection
        transformed_sw = pyproj.transform(p_ll, p_mt, sw.x, sw.y)  # Transform NW point to 3857
        transformed_ne = pyproj.transform(p_ll, p_mt, ne.x, ne.y)  # .. same for SE

        # Iterate over 2D area
        gridpoints = []
        x = transformed_sw[0]
        while x < transformed_ne[0]:
            y = transformed_sw[1]
            while y < transformed_ne[1]:
                p = shapely.geometry.Point(pyproj.transform(p_mt, p_ll, x, y))
                gridpoints.append(p)
                y += stepsize
            x += stepsize

        with open('testout.csv', 'w') as of:
            of.write('Latitude,Longitude\n')
            for p in gridpoints:
                of.write('{:f},{:f}\n'.format(p.x, p.y))



route = Routing()
data = pd.read_csv('./data/DANAOS/EXPRESS ATHENS/4_leg.csv', ).values
#initdata = data[1100:1900,:]
#data = data[1150:1900,:]
#initdata = data[9950:11050,:]
#data = data[10000:11000,:]
initdata = data[3100:3900,:]
data = data[3150:3900,:]

lats = data[:, 6]
lons = data[:, 7]
dt = data[:,8]
speeds = data[:,0]
draft = data[:,9]
bearings = data[:,]
focs = data[:,2]
with open('./data/DANAOS/EXPRESS ATHENS/EXPRESS ATHENSCoor1.csv', mode='w') as dataw:
    data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    data_writer.writerow(['Latitude', 'Longitude','dt','draft','speed','foc'])
    for i in range(0, len(data)):
        data_writer.writerow(
            [lats[i], lons[i],dt[i],draft[i],speeds[i], focs[i]])


#route.constructGrid(initdata)
df1 = pd.read_csv('./data/DANAOS/EXPRESS ATHENS/EXPRESS ATHENSCoor1.csv')

df2 = pd.read_csv('./data/DANAOS/EXPRESS ATHENS/testout.csv')


route.initializeParticles(df2.values,df1.values)

#df3 = pd.read_csv('./optRoute.csv')
df5 = pd.read_csv('./initialStraightLineRoute.csv')
df4 = pd.read_csv('./optimalRoute.csv')


#route.evaluateOptimalRoute()

arr1 = np.array(df1.values)
arr2 = np.array(df2.values)
'''arr2 = np.array([
    [ 51.020692,1.175194],
    [51.020692, 2.586421 ],
    [51.356709, 2.586421 ],
    [51.356709,1.175194 ],]
                )'''
#arr = np.append(arr1.reshape(-1,2),arr2,axis=0)
#df = pd.DataFrame({'Latitude':arr[:,0],'Longitude':arr[:,1]})
merged = pd.concat([df1,df4])
route.PLotTrajectory(merged, 'EXPRESS ATHENS')
#x=0
