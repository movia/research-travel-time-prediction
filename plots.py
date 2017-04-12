import pandas as pd
from pp import load_csv

# Load and pre-process data
print('Loading data ...')
data = pd.read_csv('data/4A_201701.csv', sep=';')

# Initial data-slicing
data = data[(data.LinkTravelTime > 0) & (data.LineDirectionCode == 1) & (26 <= data.LineDirectionLinkOrder) & (data.LineDirectionLinkOrder <= 32)]

grouping = data.groupby(['LinkName'])['LinkTravelTime'].mean()