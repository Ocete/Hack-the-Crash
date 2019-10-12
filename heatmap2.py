import pandas as pd
import geoplotlib
from geoplotlib import layers, core
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import gmplot
from pyproj import Proj, transform

def lemao(row):
    inProj = Proj(init='epsg:3857')
    outProj = Proj(init='epsg:4329')
    print(row['latitude'])
    if row['latitude'] < 40:
        row['latitude'], row['longitude'] = transform(inProj, outProj,row['location_northing_osgr'],row['location_easting_osgr'])

    print(row['latitude'])
    return row

datafile = 'data/accidents.csv'
map = gpd.read_file('data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')[['ADMIN', 'ADM0_A3', 'geometry']]

df = pd.read_csv(datafile)

print(df.loc[df['latitude']<40, ['location_easting_osgr', 'location_northing_osgr']])
df = df[df['latitude']<40]
print(df.describe())

df.apply(lemao, axis=1)
print(df.describe())

geometry = [Point(xy) for xy in zip (df['longitude'], df['latitude'])]
print(geometry[:3])
crs = {"init": "epsg:4329"}
geo_df = gpd.GeoDataFrame(df, crs=crs, geometry= geometry)
print(geo_df.head())

fig,ax = plt.subplots(figsize = (15,15))
map.plot(ax = ax, alpha = 0.4, color="grey")
geo_df.plot(ax = ax, markersize=1)
plt.show()
