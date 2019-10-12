import pandas as pd

# Read CSV from memory
dataset = pd.read_csv(
    'data/vehicles.csv',
    header=0
)

# There were some equal rows, all except one are deleted
dataset = dataset.drop_duplicates()

# We delete the sex of the driver on purpose since we prefer not to
# take this information into account
dataset = dataset.drop('Sex_of_Driver', axis=1)

# A code identifier about the car is not important to predict the severity of an accident
dataset = dataset.drop('Vehicle_Reference', axis=1)

# Percentage of serious accidents is approx constant independently of IMD value, so discarded
dataset = dataset.drop(['Vehicle_IMD_Decile', 'Driver_IMD_Decile'], axis=1)

# Left or right hand drive is not useful since almost every value
# is no and the proportion of serious accidents is almost constant
dataset = dataset.drop(['Was_Vehicle_Left_Hand_Drive?'], axis=1)


# Almost every vehicle is not articulated
dataset = dataset.drop(
    'Towing_and_Articulation',
    axis=1
)

# Difficult to influence in this variable
dataset = dataset.drop(
    'Vehicle_Manoeuvre',
    axis=1
)

# Deleting this variable because almost every value is none
dataset = dataset.drop('Hit_Object_in_Carriageway', axis=1)

# Since we don't know the meaning of this variable and we cannot
# obtain it, we decide to drop this variable since it's not very informative
dataset = dataset.drop(
    'Driver_Home_Area_Type',
    axis=1
)

# This variable gets converted into 0-1 encoding whether
# the vehicle has abandoned the carriageway or not
dataset['Carriageway_Left'] = (dataset['Vehicle_Leaving_Carriageway'] != 'Did not leave carriageway')*1

dataset = dataset.drop('Vehicle_Leaving_Carriageway', axis=1)

# Deleting it since it is not very informative variable for the algorithm
dataset = dataset.drop('Journey_Purpose_of_Driver', axis=1)

# Junction_Location collides with another column from accidents. Deleting
dataset = dataset.drop('Junction_Location', axis=1)

# MISSING DATA HANDLING
# Missing values for categorical data are replaced with the mode of the variable (common point imputation)
dataset.loc[dataset['Propulsion_Code'] == '-1', 'Propulsion_Code'] = dataset['Propulsion_Code'].mode()
dataset.loc[dataset['Vehicle_Type'] == '-1', 'Vehicle_Type'] = dataset['Vehicle_Type'].mode()

dataset.loc[
    dataset['Vehicle_Location-Restricted_Lane'] == '-1', 'Vehicle_Location-Restricted_Lane'
] = dataset['Vehicle_Location-Restricted_Lane'].mode()

dataset.loc[
    dataset['Skidding_and_Overturning'] == '-1', 'Skidding_and_Overturning'
] = dataset['Skidding_and_Overturning'].mode()

dataset.loc[
    dataset['Hit_Object_off_Carriageway'] == '-1', 'Hit_Object_off_Carriageway'
] = dataset['Hit_Object_off_Carriageway'].mode()


# # INTERVALS DEFINITION
# # For
# dataset['Age_of_Vehicle'] = pd.cut(
#     dataset['Age_of_Vehicle'], right=False, bins=[0, 5, 10, 20, 100]
# )

# dataset['Engine_Capacity_(CC)'] = pd.cut(
#     dataset['Engine_Capacity_(CC)'], right=False, bins=[0, 500, 1000, 1500, 2000, 3000, 4000]
# )
