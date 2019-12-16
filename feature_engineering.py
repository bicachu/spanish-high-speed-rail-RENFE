import pandas as pd
import numpy as np

# Read in original data set provided from Kaggle
renfedata = pd.read_csv("input/renfe.csv")

# Read in holidays.csv of all national/local holidays/festivals in Spain
holidays_df = pd.read_csv('datasets/holidays.csv')

# Process dates to date times
renfedata['insert_date'] = pd.to_datetime(renfedata['insert_date'])
renfedata['start_date'] = pd.to_datetime(renfedata['start_date'])
renfedata['end_date'] = pd.to_datetime(renfedata['end_date'])

# Create feature for travel time
renfedata['travel_mins'] = renfedata['end_date'] - renfedata['start_date']
renfedata['travel_mins'] = renfedata['travel_mins']/np.timedelta64(1, 'm')


# Helper function to update populations for origin
def originpop(data):
    if data['origin'] == 'MADRID':
        return 6550000
    elif data['origin'] == 'BARCELONA':
        return 5515000
    elif data['origin'] == 'SEVILLA':
        return 1945000
    elif data['origin'] == 'VALENCIA':
        return 2531000
    elif data['origin'] == 'PONFERRADA':
        return 65239


# Helper function to update populations for origin
def destpop(data):
    if data['destination'] == 'MADRID':
        return 6550000
    elif data['destination'] == 'BARCELONA':
        return 5515000
    elif data['destination'] == 'SEVILLA':
        return 1945000
    elif data['destination'] == 'VALENCIA':
        return 2531000
    elif data['destination'] == 'PONFERRADA':
        return 65239


# Set new columns for population size feature and initialize to 0
renfedata['originpop'] = 0
renfedata['destinationpop'] = 0

# Apply population helper function update newly created features highlighting city population
renfedata['originpop'] = renfedata.apply(originpop, axis=1)
renfedata['destinationpop'] = renfedata.apply(destpop, axis=1)

# Create dictionary to store cities as hash keys
holidays_dict = {
    "MADRID": [],
    "BARCELONA": [],
    "VALENCIA": [],
    "SEVILLA": [],
    "PONFERRADA": []
}

# Add all holiday dates to dictionary for respective city
for index, row in holidays_df.iterrows():
    holidays_dict[holidays_df['rel_city'][index]].append(pd.to_datetime(holidays_df['date'][index]))


# Helper function returns number of days away until next holiday based on given date
def daysToHoliday(items_mad, items_other, ticket_date):
    nearest_mad_hol_date = min(items_mad, key=lambda x: (x.date() - ticket_date.date()).days if (x.date() - ticket_date.date()).days >= 0 else float('inf'))
    nearest_otr_hol_date = min(items_other, key=lambda x: (x.date() - ticket_date.date()).days if (x.date() - ticket_date.date()).days >= 0 else float('inf'))

    # Check to see if both cities have at least one holiday date in the future
    # If not, only return the maximum date between the both i.e -  Madrid's next holiday
    if nearest_mad_hol_date.date() >= ticket_date.date() and nearest_otr_hol_date.date() >= ticket_date.date():
        nearest_date = min(nearest_mad_hol_date, nearest_otr_hol_date)
    else:
        nearest_date = max(nearest_mad_hol_date, nearest_otr_hol_date)

    return (nearest_date.date() - ticket_date.date()).days


# Helper function returns number of days that have passed since last holiday based on given date
def daysFromHoliday(items_mad, items_other, ticket_date):
    nearest_mad_hol_date = max(items_mad, key=lambda x: (x.date() - ticket_date.date()).days if (x.date() - ticket_date.date()).days <= 0 else float('-inf'))
    nearest_otr_hol_date = max(items_other, key=lambda x: (x.date() - ticket_date.date()).days if (x.date() - ticket_date.date()).days <= 0 else float('-inf'))

    # Check to see if both cities have at least one holiday date in the past
    # If not, only return the maximum date between the both i.e -  Madrid's last holiday
    if nearest_mad_hol_date.date() <= ticket_date.date() and nearest_otr_hol_date.date() <= ticket_date.date():
        nearest_date = max(nearest_mad_hol_date, nearest_otr_hol_date)
    else:
        nearest_date = min(nearest_mad_hol_date, nearest_otr_hol_date)

    return abs((nearest_date.date() - ticket_date.date()).days)


# Function to compute days to holiday
def computeDaysToHoliday(data):
    if data['origin'] == 'MADRID':
        city_to_check = data['destination']
    else:
        city_to_check = data['origin']

    # Extract list of holiday dates for MADRID and other CITY
    madrid_dates = holidays_dict.get('MADRID')
    local_city_dates = holidays_dict.get(city_to_check)

    # Compute the minimum number of days away to nearest holiday for both and return the minimum of both
    return daysToHoliday(madrid_dates, local_city_dates, pd.to_datetime(data['start_date']))


# Function to compute days from holiday
def computeDaysFromHoliday(data):
    if data['origin'] == 'MADRID':
        city_to_check = data['destination']
    else:
        city_to_check = data['origin']

    # Extract list of holiday dates for MADRID and other CITY
    madrid_dates = holidays_dict.get('MADRID')
    local_city_dates = holidays_dict.get(city_to_check)

    # Compute the minimum number of days away to nearest holiday for both and return the minimum of both
    return daysFromHoliday(madrid_dates, local_city_dates, pd.to_datetime(data['start_date']))


# Apply functions and create two new columns in data frame for days to/from holiday
renfedata['days_to_holiday'] = renfedata.apply(computeDaysToHoliday, axis=1)
renfedata['days_from_holiday'] = renfedata.apply(computeDaysFromHoliday, axis=1)


# Helper function to update distances between cities
def distmeasure(data):
    if data['destination'] == 'MADRID' and data['origin'] == 'BARCELONA':
        return 505
    elif data['destination'] == 'BARCELONA' and data['origin'] == 'MADRID':
        return 505
    elif data['destination'] == 'MADRID' and data['origin'] == 'SEVILLA':
        return 390
    elif data['destination'] == 'SEVILLA' and data['origin'] == 'MADRID':
        return 390
    elif data['destination'] == 'MADRID' and data['origin'] == 'VALENCIA':
        return 302
    elif data['destination'] == 'VALENCIA' and data['origin'] == 'MADRID':
        return 302
    elif data['destination'] == 'MADRID' and data['origin'] == 'PONFERRADA':
        return 338
    elif data['destination'] == 'PONFERRADA' and data['origin'] == 'MADRID':
        return 338


# Create distance feature to compute distance between origin & destination
renfedata['distance'] = renfedata.apply(distmeasure, axis=1)


# Convert start_date to month, date, day of the week, hour, minute
renfedata['month'] = pd.to_datetime(renfedata['start_date']).dt.month
renfedata['date'] = pd.to_datetime(renfedata['start_date']).dt.day
renfedata['dotw'] = pd.to_datetime(renfedata['start_date']).dt.weekday
renfedata['hour'] = pd.to_datetime(renfedata['start_date']).dt.hour
renfedata['minute'] = pd.to_datetime(renfedata['start_date']).dt.minute


# Helper function to determine if train is "overnight"
def isOvernight(data):
    if pd.to_datetime(data['start_date']).day != pd.to_datetime(data['end_date']).day:
        return 1
    else:
        return 0


# Create feature to flag overnight trips
renfedata['is_overnight'] = renfedata.apply(isOvernight, axis=1)


# Helper function to consolidate related fare types
def fareType(data):
    if data['fare'] == 'Promo':
        return 'Promo'
    elif data['fare'] == 'Flexible':
        return 'Flexible'
    elif data['fare'] == 'Adulto ida':
        return 'Adulto Ida'
    elif data['fare'] == 'Promo +':
        return 'Promo +'
    elif data['fare'] == 'Grupos Ida':
        return 'Grupos Ida'
    elif data['fare'] == 'Double Sleeper-Flexible':
        return 'Flexible'
    elif data['fare'] == 'COD.PROMOCIONAL':
        return 'Promo'
    elif data['fare'] == 'Mesa':
        return 'Table'
    elif data['fare'] == 'Individual-Flexible':
        return 'Individual Sleeper-Flexible'
    else:
        return np.nan


# Helper function to consolidate the train classes
def trainClasses(data):
    if data['train_class'] == 'TuristaSólo plaza H':
        return np.nan
    elif data['train_class'] == 'PreferenteSólo plaza H':
        return np.nan
    elif data['train_class'] == 'Turista PlusSólo plaza H':
        return np.nan
    else:
        return data['train_class']


# Helper function to remove negative days to trips or illogical observations
def daystoTrip(data):
    if data['days_to_trip'] < 0:
        return np.nan
    else:
        return data['days_to_trip']


# Apply functions and update fare and train class features
renfedata['fare'] = renfedata.apply(fareType, axis=1)
renfedata['train_class'] = renfedata.apply(trainClasses, axis=1)

# Create features for difference inn days away from trip as of scrape date
renfedata['days_to_trip'] = (renfedata['start_date'].dt.date - renfedata['insert_date'].dt.date).dt.days
renfedata['days_to_trip'] = renfedata.apply(daystoTrip, axis=1)

# Drop nan values in remaining data set
renfedata = renfedata.dropna()

# Save cleaned data set without categorical encoding for use in EDA purposes
renfedata.to_csv("input/not_encoded_clean_data.csv")

# Perform one-hot encoding on categorical features
renfedata = pd.get_dummies(renfedata, columns=['origin', 'destination', 'dotw', 'train_type', 'fare', 'train_class'])

# Save cleaned data set with categorical encoding for use in modeling purposes
renfedata.to_csv("input/cleaned_data.csv")
