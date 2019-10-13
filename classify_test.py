import pandas as pd
import numpy as np

from sklearn import (
    metrics,
    model_selection,
    svm,
    ensemble
)

# TRAINING PREPARATION - EXPLAINED IN THE OTHER NOTEBOOK
dataset_accidents = pd.read_csv(
    'data/accidents.csv',
    header=0
)
dataset_accidents = dataset_accidents.drop_duplicates()

dataset_vehicles = pd.read_csv(
    'data/vehicles.csv',
    header=0
)
dataset_vehicles = dataset_vehicles.drop_duplicates()

# Preprocessing of training set (explained in the other notebook)
dataset_accidents = dataset_accidents[
    dataset_accidents['latitude'] >= 40
]

dropped_cols_begin = [
    'location_easting_osgr', 'location_northing_osgr',
    'lsoa_of_accident_location', 'latitude', 'longitude',
    'local_authority_district', 'local_authority_highway',
    'pedestrian_crossing-human_control',
    'pedestrian_crossing-physical_facilities',
    'carriageway_hazards', '1st_road_number', '2nd_road_number'
]

dataset_accidents.drop(dropped_cols_begin, axis=1, inplace=True)

dataset_accidents['weekday'] = pd.to_datetime(
    dataset_accidents['date']
).dt.weekday_name
dataset_accidents['weekend'] = (dataset_accidents['weekday'].isin(
    ['Friday', 'Saturday', 'Sunday']
))*1
dataset_accidents['day_period'] = pd.to_datetime(
    dataset_accidents['time']
).dt.hour
dataset_accidents['day_period'] = pd.cut(
    dataset_accidents['day_period'], bins=[0, 7, 9, 13, 16, 20, 24], right=False
)
dataset_accidents['urban_area'] = 1*(
    dataset_accidents['urban_or_rural_area'] == 'Urban'
)

dropped_cols_intermediate = ['date', 'time', 'weekday', 'urban_or_rural_area']
dataset_accidents.drop(dropped_cols_intermediate, axis=1, inplace=True)

# MERGING BOTH DATASETS USING PRIMARY KEY TO HAVE INFORMATION
# FROM TWO SOURCES
dataset = dataset_accidents.merge(dataset_vehicles, on='accident_id')

dataset = dataset.drop('accident_id', axis=1)

dataset['Towing_and_Articulation'] = (~dataset['Towing_and_Articulation'].isin(
    ['-1', 'No tow/articulation']
))
dataset['Carriageway_Left'] = (dataset['Vehicle_Leaving_Carriageway'] != 'Did not leave carriageway')*1

dataset['Skidding_and_Overturning'] = dataset[
    dataset['Skidding_and_Overturning'].isin(['-1', 'None'])
]

dropped_cols_end = [
    'Sex_of_Driver', 'Vehicle_Reference',
    'Vehicle_IMD_Decile', 'Driver_IMD_Decile',
    'Was_Vehicle_Left_Hand_Drive?', 'Hit_Object_in_Carriageway',
    'Driver_Home_Area_Type', 'Vehicle_Leaving_Carriageway',
    'Junction_Location', 'Journey_Purpose_of_Driver'
]

dataset.drop(dropped_cols_end, axis=1, inplace=True)

# MISSING DATA HANDLING
# Missing values for categorical data are replaced with the mode of the variable (common point imputation)

missing_data = {
    'Propulsion_Code': dataset['Propulsion_Code'].mode()[0],
    'Vehicle_Type': dataset['Vehicle_Type'].mode()[0],
    'Vehicle_Location-Restricted_Lane': dataset[
        'Vehicle_Location-Restricted_Lane'
    ].mode()[0],
    'Skidding_and_Overturning': dataset['Skidding_and_Overturning'].mode()[0],
    'Hit_Object_off_Carriageway': dataset['Hit_Object_off_Carriageway'].mode()[0],
}

for key, val in missing_data.items():
    dataset.loc[dataset[key] == '-1', key] = val

dataset = pd.get_dummies(dataset)
dataset = pd.get_dummies(
    dataset,
    columns=['road_type', 'weather_conditions']
)

dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='-1')))]

dataset = dataset.sample(frac=1)

x = dataset.drop('target', axis=1)
y = dataset['target']

classifier = ensemble.RandomForestClassifier(
    n_estimators=100, max_depth=20, class_weight='balanced'
)

classifier.fit(x, y)

# PREDICTION PHASE
dataset_test = pd.read_csv(
    'data/test.csv',
    header=0
)

dataset_test.drop(dropped_cols_begin, axis=1, inplace=True)

dataset_test['weekday'] = pd.to_datetime(
    dataset_test['date']
).dt.weekday_name
dataset_test['weekend'] = (dataset_test['weekday'].isin(
    ['Friday', 'Saturday', 'Sunday']
))*1
dataset_test['day_period'] = pd.to_datetime(
    dataset_test['time']
).dt.hour
dataset_test['day_period'] = pd.cut(
    dataset_test['day_period'], bins=[0, 7, 9, 13, 16, 20, 24], right=False
)
dataset_test['urban_area'] = 1*(
    dataset_test['urban_or_rural_area'] == 'Urban'
)

dataset_test.drop(dropped_cols_intermediate, axis=1, inplace=True)

dataset_test = dataset_test.merge(dataset_vehicles, on='accident_id')

dataset_ids = dataset_test['accident_id']
dataset_test = dataset_test.drop('accident_id', axis=1)

dataset_test['Towing_and_Articulation'] = (~dataset_test['Towing_and_Articulation'].isin(
    ['-1', 'No tow/articulation']
))
dataset_test['Carriageway_Left'] = (
    dataset_test['Vehicle_Leaving_Carriageway'] != 'Did not leave carriageway'
)*1

dataset_test['Skidding_and_Overturning'] = dataset_test[
    dataset_test['Skidding_and_Overturning'].isin(['-1', 'None'])
]

for key, val in missing_data.items():
    dataset_test.loc[dataset_test[key] == '-1', key] = val

dataset_test = pd.get_dummies(dataset_test)
dataset_test = pd.get_dummies(
    dataset_test,
    columns=['road_type', 'weather_conditions']
)

dataset_test = dataset_test[
    dataset_test.columns.drop(list(dataset_test.filter(regex='-1')))
]

missing_cols = set(x.columns) - set(dataset_test.columns)
for col in missing_cols:
    dataset_test[col] = 0

dataset_test = dataset_test[x.columns]

predictions = classifier.predict(dataset_test)

predictions_df = pd.DataFrame(
    {'id': dataset_ids, 'prediction': predictions}
)

final_preds = predictions_df.groupby('id').mean()
final_preds = final_preds.astype(int).reindex(dataset_ids.unique())
