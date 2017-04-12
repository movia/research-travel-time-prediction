import numpy as np
import pandas as pd

def load_csv(file):
    data = pd.read_csv(file, sep=';')

    # Initial data-slicing
    data = data[(data.LinkTravelTime > 0) & (data.LineDirectionCode == 1) & (26 <= data.LineDirectionLinkOrder) & (data.LineDirectionLinkOrder <= 32)]

    # Data convertion
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    time = pd.DatetimeIndex(data['DateTime']) 
    data['TimeOfDay'] = time.hour #+ time.minute / 60

    categorial_columns = ['DayType', 'LinkRef']
    numerical_columns = ['TimeOfDay']
    
    output_column = 'LinkTravelTime'

    # Calculate m lag headway and travel time
    m = 3
    grouping = data.groupby(categorial_columns)
    for i in range(1, m + 1):
        data['HeadwayTime_L' + str(i)] = (data['DateTime'] - grouping['DateTime'].shift(i)) / np.timedelta64(1, 's')
        data['LinkTravelTime_L' + str(i)] = grouping['LinkTravelTime'].shift(i)
        numerical_columns += ['HeadwayTime_L' + str(i), 'LinkTravelTime_L' + str(i)]

    # Slice out missing values
    for i in range(1, m + 1):
        data = data[(data['HeadwayTime_L' + str(i)] > 0) & (data['LinkTravelTime_L' + str(i)] > 0)]

    print('Preprosessed data set size:', len(data))

    input_columns = categorial_columns + numerical_columns

    with_dummies = pd.get_dummies(data[input_columns], columns = categorial_columns)

    data_with_dummies = with_dummies.copy()
    data_with_dummies[output_column] = data[output_column]
    data_with_dummies.to_csv('data/pp.csv', sep = ';')

    # Create dummy variables
    X = with_dummies.as_matrix()
    Y = data.as_matrix(columns = [output_column])

    return (X, Y)
