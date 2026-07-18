import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# 1. Load the dataset (replace with your local file path if needed)
fish_df = pd.read_csv('Fish-data.csv')

# 2. Select the class column and features you want to plot
features = ['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
cols_to_plot = ['CAT'] + features

# 3. Create the parallel coordinates plot
plt.figure(figsize=(12, 7))
parallel_coordinates(fish_df[cols_to_plot], 'CAT', colormap='tab10')

# 4. Customize the plot
plt.title('Parallel Coordinates of Fish Market Data', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.legend(loc='upper right')

plt.show()

