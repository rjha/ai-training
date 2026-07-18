import pandas as pd 

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# The following code imports the dataset that is used in the colab.

training_df = pd.read_csv(filepath_or_buffer="./california_housing_train.csv")
# The following code returns basic statistics about the data in the dataframe.
print(training_df.describe())


