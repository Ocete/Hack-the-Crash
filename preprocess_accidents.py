import pandas as pd

# Read CSV from memory
dataset = pd.read_csv(
    'data/accidents.csv',
    header=0
)

# There were some equal rows, all except one are deleted
dataset = dataset.drop_duplicates()

# There were some accidents placed in the coast of Morocco,
# we suppose all of them are noise and they get discarded.
# Moreover, they are all from the majority class, so, it's
# not too problematic to load information with this drop
dataset = dataset[dataset['latitude'] >= 40]

# The information about location is redundant with the police_force
# column, since the whole country is divided in regions controlled
# by each force.
dataset = dataset.drop(
    ['location_easting_osgr', 'location_northing_osgr',
     'lsoa_of_accident_location', 'latitude', 'longitude'],
    axis=1
)

# Variables police_force, local_authority_highway and
# local_authority_district represent the same information, so
# only the most concise information is kept
dataset = dataset.drop(['local_authority_district', 'local_authority_highway'], axis=1)

# Temporal information converted into more relevant data
dataset['weekday'] = pd.to_datetime(dataset['date']).dt.weekday_name
dataset['day_period'] = pd.to_datetime(dataset['time']).dt.hour
dataset['day_period'] = pd.cut(
    dataset['day_period'], bins=[0, 8, 14, 18, 24], right=False
)

dataset = dataset.drop(['date', 'time'], axis=1)

# Variables with too much invalid or non computed values

dataset = dataset.drop(
    [
        'pedestrian_crossing-human_control',
        'pedestrian_crossing-physical_facilities',
        'carriageway_hazards'
    ],
    axis=1
)
# Urban and rural into boolean variable
dataset['urban_area'] = (dataset['urban_or_rural_area'] == 'Urban')*1
dataset = dataset.drop('urban_or_rural_area', axis=1)

# Class is more representative than the number of the road
dataset = dataset.drop(['1st_road_number', '2nd_road_number'], axis=1)
