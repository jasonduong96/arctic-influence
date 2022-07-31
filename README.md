The whole purpose of this project is to illustrate the influences of temperature fronts on selected U.S cities by month and investigate whether the fronts will influence certain U.S cities moreso than others. The final deliverable is an animation that quanitifes the increase or decrease in the overall average temperature that month plotted over year compared to that of Columbus, Ohio, using the R-Value as a measurement. For example:

The R-Value of Atlanta, Georgia is 0.81 for the month of Janurary. That means that overall, in Janurary of a certain year, if Columbus, Ohio was above-average in terms of temperature, Atlanta was also likely to be above-average as well. An R-Value of 1.00 would mean that whatever fronts swept through Columbus also swept through Atlanta equally the entire time every single Janurary. An R-Value of 0.81 then means that in certain years, sweeping cold fronts may not have made Atlanta as cold as expected due to the prescence of the Gulf of Mexico southernly winds that may have alleivated much of that cold, or that rain and clouds from the gulf of Mexico may have kept Atlanta unusually cold for long periods, while Columbus bakes in full sunshine simulateniously. 

The R-Value of Seattle, Washington is -0.4 for the month of Janurary. This means that Seattle is more likely to be above-average overall in terms of temperarure if Columbus was below-average and vice-versa in a certain year, but not by much.

By understanding climate patterns and the role these fronts play on the U.S as a whole, we will be able to better predict the long-term forecast in coming years and know which U.S regions will be more affected by certain climatic events than others.

Temperature data used in this project was from 1995-2020 from Kaggle.

ON July 10th 2022, an updated version of the project was uploaded. The updates resulted in a more accurate animation that described temperature variation relationships between Columbus and other cities. The biggest change made was adding a function near the beginning that checked the entire dataframe worth of all the temperature data for invalid data (in this case, we searched for in the average temperature column for values of -99.0, as they obivously should not be there). This is really a basic step that should been done a lot earlier, it is just simple data cleaning. Anytime when you start working with a huge dataset, the first thing that should be done is to briefly go over the dataset and make sure the data looks good and/or that there is no missing data. 

Unfortunately, the city_temperatures.csv file could not be uploaded here due to its vast size of over 137,000 KB.
