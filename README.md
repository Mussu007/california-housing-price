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
We then use ` StratifiedShuffleSplit` function from `sklearn.model_selection` to split our data keeping in mind our new income_category. As you can see, the test set generated using stratified sampling has ncome category proportions almost identical to those in the full dataset, whereas the
test set generated using purely random sampling is skewed. After doing that, we will remove the `income_category` column so that we get back our original data. 

## Exploring our data
### 1. Visualization using scatter plot
We will create a copy of our training set, so that we don't harm the original data. We first use scatter plot and use the `alpha` argument to highlight the most densely populated areas. We can clearly see the high-density areas, namely the Bay Area and around Los Angeles and San Diego, plus a long line of fairly high density in the Central Valley, in particular around Sacramento and Fresno. 
Along with that, lets add the price of the houses, The radius of each circle represents the district’s population (option s), and the color represents the price (option c). We will use a predefined color map (option cmap) called jet, which ranges from blue (low values) to red (high prices). The housing prices are very much related to the location e.g., close to the ocean) and to the population density. The ocean proximity attribute may be useful as well, although in Northern California the housing prices in oastal districts are not too high. Alternatively, the site provided the code to import the image of California state, using that we are able to justify our findings above.

### 2. Correlation amongst our variables
The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation; for example, the median house value tends to go up when the median income goes up. When the coefficient is close to –1, it means that there is a strong negative correlation; you can see a small negative correlation between the latitude and the median house value (i.e., prices have a slight tendency to go down when you go north). Finally, coefficients close to 0 mean that there is no linear correlation.
Alternatively,  This scatter matrix plots every numerical attribute against every othernumerical attribute, plus a histogram of each numerical attribute. The most promising attribute to predict the median house value is the median income. The `median_income` vs `median_house_value` plot reveals a few things. First, the correlation is indeed very strong; you can clearly see the upward trend, and the points are not too dispersed. Second, the price cap that we noticed earlier is clearly visible as a horizontal line at $500,000. But this plot reveals other less obvious straight lines: a horizontal line around $450,000, another around $350,000, perhaps one around $280,000, and a few more below that.

### 3. Experimenting with Attribute Combinations
The total number of rooms in a district is not very useful if we don’t know how many households there are. What we really want is the number of rooms per household. Similarly, the total number of bedrooms by itself is not very useful: we probably want to compare it to the number of rooms. And the population per household also seems like an interesting attribute combination to look at. 
The new bedrooms_per_room attribute is much more correlated with the median house value than the total number of rooms or bedrooms. Apparently, houses with a lower bedroom/room ratio tend to be more expensive. The number of rooms per household is also more informative than the total number of rooms in a district—obviously the larger the houses, the more expensive they are.

### 4. Prepare the Data for Machine Learning Algorithm
Let’s revert to a clean training set (by copying strat_train_set once again). Let’s also separate the predictors and the labels, since we don’t necessarily want to
apply the same transformations to the predictors and the target values. 

#### Data Cleaning
We saw earlier that the total_bedrooms attribute has some missing values. We will use `sklearn.impute` and get the `SimpleImputer` suntion to impute the missign value. First, we create an instance specifying that you want to replace each attribute’s missing values with the median of that attribute. Since the median can only be computed on numerical attributes, you need to create a copy of the data without the text attribute ocean_proximity. We fit the imputer instance to the training data using the `fit()` method. The imputer has simply computed the median of each attribute and stored the result in its statistics_ instance variable. Only the total_bedrooms attribute had missing values. Now we can use this “trained” imputer to transform the training set by replacing missing values with the learned medians.

#### Handling Text and Categorical Attributes
Let’s look at text attributes. In this dataset, there is just one: the ocean_proximity attribute. Let’s convert these categories from text to numbers using `sklearn.preprocessing` `OrdinalEncoder` class. We get the list of categories using the categories_ instance variable. It is a list containing a 1D array of categories for each categorical attribute. One issue with this representation is that ML algorithms will assume that two nearby values are more similar than two distant values. To fix this issue, a common solution is to create one binary attribute per category: one attribute equal to 1 when the category is “<1H OCEAN” (and 0 otherwise), another attribute equal to 1 when the category is “INLAND” (and 0 otherwise), and so on. This is called one-hot encoding, because only one attribute will be equal to 1 (hot), while the others will be 0 (cold). We use `OneHotEncoding` class to achieve the following.
After one hot encoding, we get a matrix with thousands of columns, and the matrix is full of 0s except for a single 1 per row. Using up tons of memory mostly to store zeros would be very wasteful, so instead a sparse matrix only stores the location of the nonzero elements. We can use it mostly like a normal 2D array, but we really want to convert it to a (dense) NumPy array, we call the `toarray()` method.
