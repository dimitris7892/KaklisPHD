import geopandas as gpd
from shapely.geometry import Point, box
from random import uniform
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm
import cartopy
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import shapely
import geopandas

lon = np.arange(129.4, 153.75+0.05, 0.25)
lat = np.arange(-43.75, -10.1+0.05, 0.25)

precip = 10 * np.random.rand(len(lat), len(lon))


ds = xr.Dataset({"precip": (["lat", "lon"], precip)},coords={"lon": lon,"lat": lat})

ds['precip'].plot()

def hv(lonlat1, lonlat2):
    AVG_EARTH_RADIUS = 6371000. # Earth radius in meter

    # Get array data; convert to radians to simulate 'map(radians,...)' part
    coords_arr = np.deg2rad(lonlat1)
    a = np.deg2rad(lonlat2)

    # Get the differentiations
    lat = coords_arr[:,1] - a[:,1,None]
    lng = coords_arr[:,0] - a[:,0,None]

    # Compute the "cos(lat1) * cos(lat2) * sin(lng * 0.5) ** 2" part.
    # Add into "sin(lat * 0.5) ** 2" part.
    add0 = np.cos(a[:,1,None])*np.cos(coords_arr[:,1])* np.sin(lng * 0.5) ** 2
    d = np.sin(lat * 0.5) ** 2 +  add0

    # Get h and assign into dataframe
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return {'dist_to_coastline': h.min(1), 'lonlat':lonlat2}

def get_distance_to_coast(arr, country, resolution='50m'):

    print('Get shape file...')
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # single geom for Norway
    geom = world[world["name"] == "Norway"].dissolve(by='name').iloc[0].geometry

    # single geom for the coastline
    c = gpd.clip(gpd.read_file("./coastlines/Europe_coastline.shp").to_crs('EPSG:4326'),
                         geom.buffer(0.25)).iloc[0].geometry

    c     = gpd.read_file(c)
    c.crs = 'EPSG:4326'

    print('Group lat/lon points...')
    points = []
    i = 0
    for ilat in arr['lat'].values:
        for ilon in arr['lon'].values:
                points.append([ilon, ilat])
                i+=1

    xlist = []
    gdpclip = gpd.clip(c.to_crs('EPSG:4326'), geom.buffer(1))
    for icoast in range(len(gdpclip)):
        print('Get coastline ({}/{})...'.format(icoast+1, len(gdpclip)))
        coastline = gdpclip.iloc[icoast].geometry #< This is a linestring

        if type(coastline) is shapely.geometry.linestring.LineString:
            coastline = [list(i) for i in coastline.coords]
        elif type(coastline) is shapely.geometry.multilinestring.MultiLineString:
            dummy = []
            for line in coastline:
                dummy.extend([list(i) for i in line.coords])
            coastline = dummy
        else:
            print('In function: get_distance_to_coast')
            print('Type: {} not found'.format(type(type(coastline))))
            exit()

        print('Computing distances...')
        result = hv(coastline, points)

        print('Convert to xarray...')
        gdf = gpd.GeoDataFrame.from_records(result)
        lon = [i[0] for i in gdf['lonlat']]
        lat = [i[1] for i in gdf['lonlat']]
        df1 = pd.DataFrame(gdf)
        df1['lat'] = lat
        df1['lon'] = lon
        df1 = df1.set_index(['lat', 'lon'])
        xlist.append(df1.to_xarray())

    xarr = xr.concat(xlist, dim='icoast').min('icoast')
    xarr = xarr.drop('lonlat')

    return xr.merge([arr, xarr])


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

#single geom for Norway
norway = world[world["name"]=="Norway"].dissolve(by='name').iloc[0].geometry

#single geom for the coastline
coastline = gpd.clip(gpd.read_file("./coastlines/Europe_coastline.shp").to_crs('EPSG:4326'),
                     norway.buffer(0.25)).iloc[0].geometry

def make_point(id):
    point = None
    while point is None or not norway.contains(point):
        point = Point(uniform(norway.bounds[0],norway.bounds[2]),
                      uniform(norway.bounds[1],norway.bounds[3]))
    return {"id": id, "geometry": point}

def compute_distance(point):
    point['dist_to_coastline'] = point['geometry'].distance(coastline)
    return point

df = pd.DataFrame({'Longitude': -0.692396, 'Latitude': 62.191318}, index=[1])

gdf = geopandas.GeoDataFrame(df,geometry=geopandas.points_from_xy(df.Longitude,df.Latitude))

print(gdf['geometry'].distance(coastline)[1])
#compute_distance(gdf)
#dist = get_distance_to_coast(ds['precip'], 'Australia')

#plt.figure()
#dist['dist_to_coastline'].plot()
#plt.show()