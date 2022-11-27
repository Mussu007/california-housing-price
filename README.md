# California-housing-price
Agenda: Using various methods to predict the median house value

## Grabbing the data
The data was provided by the website: https://raw.githubusercontent.com/ageron/handson-ml2/master/, so we used urllib, os, tarfile and pandas to load the data. This exercise was done by me to prcatice loops and also to understand how to get data from websites.

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

