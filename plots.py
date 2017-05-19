import pandas as pd
from pp import load_csv

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

# Load and pre-process data
print('Loading data ...')
data = pd.read_csv('data/4A_201701.csv', sep=';')

# Initial data-slicing
data = data[(data.LinkTravelTime > 0) & (data.LineDirectionCode == 1) & (26 <= data.LineDirectionLinkOrder) & (data.LineDirectionLinkOrder <= 32)]

grouping = data.groupby(['LineDirectionLinkOrder', 'LinkName']).agg({ 'LinkTravelTime': { 'Mean': np.mean, 'Std.' : np.std, 'p5' : percentile(5), 'p95' : percentile(95) } }).reset_index()
grouping = grouping.rename(columns={"LinkTravelTime": "Link Travel Time", "LinkName": "Link Name"})
pd.set_option('display.max_colwidth', -1)
grouping = grouping[[('Link Name', ''), ('Link Travel Time', 'Mean'), ('Link Travel Time', 'Std.'), ('Link Travel Time', 'p5'), ('Link Travel Time', 'p95')]]
grouping.to_latex('paper/data.tex', index = False, float_format = "%1.0f")

list(grouping.columns.values)