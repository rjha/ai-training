import pandas as pd
from matplotlib import pyplot as plt
import io


# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format



# The following code defines the plotting functions that can be used to
# visualize the data.

def plot_the_dataset(feature, label, number_of_points_to_plot):
  """Plot N random points of the dataset."""

  # Label the axes.
  plt.xlabel(feature)
  plt.ylabel(label)

  # Create a scatter plot from n random points of the dataset.
  random_examples = training_df.sample(n=number_of_points_to_plot)
  plt.scatter(random_examples[feature], random_examples[label])

  # Render the scatter plot.
  plt.show()

def plot_a_contiguous_portion_of_dataset(feature, label, start, end):
  """Plot the data points from start to end."""

  # Label the axes.
  plt.xlabel(feature + "Day")
  plt.ylabel(label)

  # Create a scatter plot.
  plt.scatter(training_df[feature][start:end], training_df[label][start:end])

  # Render the scatter plot.
  plt.show()

def plot_for_day_of_week(day_of_week):
  start = day_of_week * 50
  end = start + 49
  print("\nDay %d" % day_of_week)
  plot_a_contiguous_portion_of_dataset("calories", "test_score", start, end)


training_df = pd.read_csv('./calories_score.csv', on_bad_lines='warn')
print(training_df.describe())
print(training_df[0:350].describe())

print("Defined the following functions:")
print("  * plot_the_dataset")
print("  * plot_a_contiguous_portion_of_dataset")
# plot_the_dataset("calories", "test_score", number_of_points_to_plot=50)

plot_for_day_of_week(3) 
plot_for_day_of_week(4) 

