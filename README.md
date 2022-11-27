# California Housing Price Project
Agenda: Using various methods to predict the median house value

## Grabbing the data
The data was provided by the website: https://raw.githubusercontent.com/ageron/handson-ml2/master/, so we used `urllib`, `os`, `tarfile` and `pandas` to load the data. This exercise was done by me to prcatice loops and also to understand how to get data from websites.

## Exploring the data
Once the data is loaded we have 10 features: 
1. Longitude
2. Latitude
3. Housing Median Age
4. Total Rooms
5. Total Bedrooms
6. Population
7. Households
8. Median Income
9. Median House Value
10. Ocean Proximity

Another thing we noticed after we saw the information using `.info()`, there are 207 missing values in `total_bedrooms` attribute and `ocean_proximity` is the only categorical variable.

We used `.describe()` to get the summary of each numerical data, however, visualization wouldresult better. So, I went ahead and made a histogram for all my numerical attribute

There are a few things you might notice in these histograms:
1. First, the median income attribute does not look like it is expressed in US dollars (USD). After checking the website, the data has been scaled and capped at 15 (actually, 15.0001) for higher median incomes, and at 0.5 (actually, 0.4999) for lower median incomes. The numbers represent roughly tens of thousands of dollars (e.g., 3 actually means about $30,000).
2. The housing median age and the median house value were also capped. The latter may be a serious problem since it is our target variable. There are again two options:
  a. Collect proper labels for the districts whose labels were capped, which is unfortunately not possible.
  b. Remove those districts from the training set and test set, as our model should not be evaluated poorly if the price goes beyond 500k.
3. These attributes have very different scales
4. Many histograms are tail-heavy: They extend much farther to the right of the median than to the left. This may make it a bit harder for some Machine Learning algorithms to detect patterns. We will try transforming these attributes later on to have more bell-shaped distributions.

## Splitting our data into test and training set
As an exercise, I tried to split without using `sklearn` library, but in the end went with the library's `train_test_split` function. So far we have considered purely random sampling methods. However, one of the most important aspect when buying a house is a person's income. We have a feature called `median_income` which tells us the median income of the place. So, after looking at the histogram, most median value are clustered around 1.5 to 6 ($15k to $60k) and then some move beyond 6. It is important to have a sufficient number of instances in your dataset for each stratum, or elsrse the estimate of a stratum’s importance may be biased. This means that you should not have too many strata, and each stratum should be large enough. So, the first bucket will range from 0 to 1.5, then 1.5 to 3, 3 to 4.5, 4.5 to 6, and lastly to infinity, and to do that we will use `np.inf`. We will use `pd.cut` function to create the categories.
