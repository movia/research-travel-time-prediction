import numpy as np
import pandas as pd
import codecs

def safe_filename(filename):
    filename = filename.replace(':', '-')
    filename = filename.replace('->', '_')
    keepcharacters = ('-','.','_')
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()

def load_csv(file, group_columns = [], categorial_columns = [], meta_columns = []):
    data = pd.read_csv(file, sep=';')

    # Initial data-slicing
    data = data[(data.LinkTravelTime > 0) & (data.LineDirectionCode == 1)]

    # Data convertion
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    time = pd.DatetimeIndex(data['DateTime']) 
    data['TimeOfDayClass'] = 'NO_PEEK' 
    data['Hour'] = time.hour
    data.ix[((7 < time.hour) & (time.hour < 9) & (data['DayType'] == 1)), 'TimeOfDayClass'] = 'PEEK' 
    data.ix[((15 < time.hour) & (time.hour < 17) & (data['DayType'] == 1)), 'TimeOfDayClass'] = 'PEEK' 
       
    numerical_columns = []
    
    output_column = 'LinkTravelTime'

    # Calculate m lag headway and travel time for same link, earlier journeys
    m = 3
    grouping = data.groupby(['LinkRef'])
    for i in range(1, m + 1):
        data['HeadwayTime_L' + str(i)] = (data['DateTime'] - grouping['DateTime'].shift(i)) / np.timedelta64(1, 's')
        data['LinkTravelTime_L' + str(i)] = grouping['LinkTravelTime'].shift(i)
        numerical_columns += ['HeadwayTime_L' + str(i), 'LinkTravelTime_L' + str(i)]

    # Slice out missing values
    for i in range(1, m + 1):
        data = data[(data['HeadwayTime_L' + str(i)] > 0) & (data['LinkTravelTime_L' + str(i)] > 0)]

    # Calculate j lag headway and travel time for journey, upstream links
    j = 3
    grouping = data.groupby(['JourneyRef'])
    for i in range(1, j + 1):
        data['LinkTravelTime_J' + str(i)] = grouping['LinkTravelTime'].shift(i)
        numerical_columns += ['LinkTravelTime_J' + str(i)]
    
    # Slice out missing values
    for i in range(1, j + 1):
        data = data[(data['LinkTravelTime_J' + str(i)] > 0)]

    data = data[(26 <= data.LineDirectionLinkOrder) & (data.LineDirectionLinkOrder <= 32)]

    print('Preprosessed data set size:', len(data))

    input_columns = categorial_columns + numerical_columns

    if len(group_columns) > 0:
        grouping = data.groupby(group_columns)
    else:
        grouping = [('all', data)]

    for key, group in grouping:
        with_dummies = pd.get_dummies(group[input_columns], columns = categorial_columns)
        
        # Create dummy variables
        X = with_dummies.as_matrix()
        Y = group.as_matrix(columns = [output_column])
        
        yield (key, X, Y, group[(meta_columns + input_columns + [output_column])])

def root_mean_square_error(y_true, y_pred): 
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def mean_absolute_error(y_true, y_pred): 
    return np.mean(np.abs(y_true - y_pred))

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true))

def write_results_table(data, outfile, group_columns, key_index = 0, true_colomn_name = 'Observed', predicted_column_name = 'Predicted'):
    with open(outfile, 'w', encoding='utf-8') as file:
        # Write header
        file.write("\\begin{tabular}{l|rrr}\n")
        file.write("    Link & MAPE (\\%) & MAE (s) & RMSE (s) \\\\ \\hline \\hline\n")
        grouping = data.groupby(group_columns)
        for key, group in grouping:
            mape = mean_absolute_percentage_error(group[true_colomn_name], group[predicted_column_name])
            mae = mean_absolute_error(group[true_colomn_name], group[predicted_column_name])
            rmse = root_mean_square_error(group[true_colomn_name], group[predicted_column_name])
            file.write(("    %s & %0.1f\\%% & %0.1f & %0.1f \\\\ \\hline \n" % (key[key_index], (mape * 100), mae, rmse)))
        mape = mean_absolute_percentage_error(data[true_colomn_name], data[predicted_column_name])
        mae = mean_absolute_error(data[true_colomn_name], data[predicted_column_name])
        rmse = root_mean_square_error(data[true_colomn_name], data[predicted_column_name])
        file.write("    \\hline \n")
        file.write(("    Overall & %0.1f\\%% & %0.1f & %0.1f \\\\ \\hline \n" % ((mape * 100), mae, rmse)))
        file.write("\end{tabular}\n")
