import pandas as pd

###################################BASICS

# .head() returns the first few rows (the “head” of the DataFrame).
# .info() shows information on each of the columns, such as the data type and number of missing values.
# .shape returns the number of rows and columns of the DataFrame.
# .describe() calculates a few summary statistics for each column.

# To better understand DataFrame objects, it's useful to know that they consist of three components, stored as attributes:

# .values: A two-dimensional NumPy array of values.
# .columns: An index of columns: the column names.
# .index: An index for the rows: either row numbers or row names.

# Sort on …	Syntax
# one column	df.sort_values("breed")
# multiple columns	df.sort_values(["breed", "weight_kg"])

# # Sort homelessness by descending family members
# homelessness_fam = homelessness.sort_values("family_members", ascending=False)

# # Print the top few rows
# print(homelessness_fam.head())

#homelessness.sort_values(by=["region", "family_members"], ascending=[True, False])

# # Select only the individuals and state columns, in that order
# ind_state = homelessness[["individuals", "state"]]

# There are many ways to subset a DataFrame, perhaps the most common is to use relational operators to return True or False for each row, then pass that inside square brackets.

# dogs[dogs["height_cm"] > 60]
# dogs[dogs["color"] == "tan"]

# You can filter for multiple conditions at once by using the "bitwise and" operator, &.

# dogs[(dogs["height_cm"] > 60) & (dogs["color"] == "tan")]

# Subsetting rows by categorical variables
# colors = ["brown", "black", "tan"]
# condition = dogs["color"].isin(colors)
# dogs[condition]

#################################### Adding new columns

# # Create indiv_per_10k col as homeless individuals per 10k state pop
# homelessness["indiv_per_10k"] = 10000 * (homelessness["individuals"] / homelessness["state_pop"])

# # Subset rows for indiv_per_10k greater than 20
# high_homelessness = homelessness[homelessness["indiv_per_10k"] > 20]

# # Sort high_homelessness by descending indiv_per_10k
# high_homelessness_srt = high_homelessness.sort_values(by=["indiv_per_10k"], ascending=[False])

# # From high_homelessness_srt, select the state and indiv_per_10k cols
# result = high_homelessness_srt[["state" ,"indiv_per_10k"]]

######################################SUM FUNCTIONS

# AGG (SUM BY YOUR CUSTOM FUNCTION)

# # A custom IQR function
# def iqr(column):
#     return column.quantile(0.75) - column.quantile(0.25)

# # Update to print IQR of temperature_c, fuel_price_usd_per_l, & unemployment
# print(sales[["temperature_c", "fuel_price_usd_per_l", "unemployment"]].agg(iqr))

# OTHER SUM FUNCTIONS
# .min()
# .max()
# .median()

##################################Cumulative statistics

# # Sort sales_1_1 by date
# sales_1_1 = sales_1_1.sort_values("date")

# # Get the cumulative sum of weekly_sales, add as cum_weekly_sales col
# sales_1_1["cum_weekly_sales"] = sales_1_1["weekly_sales"].cumsum()

# # Get the cumulative max of weekly_sales, add as cum_max_sales col
# sales_1_1["cum_max_sales"] = sales_1_1["weekly_sales"].cummax()

# # See the columns you calculated
# print(sales_1_1[["date", "weekly_sales", "cum_weekly_sales", "cum_max_sales"]])

################################COUNTING AND DROP DUPLICATES
# # Drop duplicate store/type combinations
# store_types = sales.drop_duplicates(subset=["store", "type"])
# print(store_types.head())

# # Drop duplicate store/department combinations
# store_depts = sales.drop_duplicates(subset=["store", "department"])
# print(store_depts.head())

# # Subset the rows where is_holiday is True and drop duplicate dates
# holiday_dates = sales[sales["is_holiday"]].drop_duplicates("date")

# # Print date col of holiday_dates
# print(holiday_dates["date"])

# # Count the number of stores of each type
# store_counts = store_types["type"].value_counts()
# print(store_counts)

# # Get the proportion of stores of each type
# store_props = store_types["type"].value_counts(normalize=True)
# print(store_props)

# # Count the number of stores for each department and sort
# dept_counts_sorted = store_types["department"].value_counts(sort=True)
# print(dept_counts_sorted)

# # Get the proportion of stores in each department and sort
# dept_props_sorted = store_types["department"].value_counts(sort=True, normalize=True)
# print(dept_props_sorted)


######################GROUP SUMMARY STATISTICS

##without groupby method:
# # Calc total weekly sales
# sales_all = sales["weekly_sales"].sum()

# # Subset for type A stores, calc total weekly sales
# sales_A = sales[sales["type"] == "A"]["weekly_sales"].sum()

# # Subset for type B stores, calc total weekly sales
# sales_B = sales[sales["type"] == "B"]["weekly_sales"].sum()

# # Subset for type C stores, calc total weekly sales
# sales_C = sales[sales["type"] == "C"]["weekly_sales"].sum()

# # Get proportion for each type
# sales_propn_by_type = [sales_A, sales_B, sales_C] / sales_all
# print(sales_propn_by_type)

##WITH GROUPBY

# # Group by type; calc total weekly sales
# sales_by_type = sales.groupby("type")["weekly_sales"].sum()

# # Get proportion for each type
# sales_propn_by_type = sales_by_type / sum(sales_by_type)
# print(sales_propn_by_type)


# # From previous step
# sales_by_type = sales.groupby("type")["weekly_sales"].sum()

# # Group by type and is_holiday; calc total weekly sales
# sales_by_type_is_holiday = sales.groupby(["type", "is_holiday"])["weekly_sales"].sum()
# print(sales_by_type_is_holiday)


#########MULTIPLE GROUP STATS

# # Import numpy with the alias np
# import numpy as np

# # For each store type, aggregate weekly_sales: get min, max, mean, and median
# sales_stats = sales.groupby("type")["weekly_sales"].agg([np.min, np.max, np.mean, np.median])

# # Print sales_stats
# print(sales_stats)

# # For each store type, aggregate unemployment and fuel_price_usd_per_l: get min, max, mean, and median
# unemp_fuel_stats = sales.groupby("type")["unemployment", "fuel_price_usd_per_l"].agg([np.min , np.max, np.mean, np.median])

# # Print unemp_fuel_stats
# print(unemp_fuel_stats)


#########PIVOT TABLES

#sales.pivot_table(values="weekly_sales", index="type")

#sales.pivot_table(values="weekly_sales", index="type" aggfunc=[np.median, np.mean])

# # Pivot for mean weekly_sales by store type and holiday 
# mean_sales_by_type_holiday = sales.pivot_table(values="weekly_sales", index="type", columns="is_holiday")

# # Print mean_sales_by_type_holiday
# print(mean_sales_by_type_holiday)

# # Print mean weekly_sales by department and type; fill missing values with 0
# print(sales.pivot_table(values="weekly_sales", index="department", columns="type", fill_value=0))

# # Print the mean weekly_sales by department and type; fill missing values with 0s; sum all rows and cols
# print(sales.pivot_table(values="weekly_sales", index="department", columns="type", fill_value=0, margins=True))

# # Add a year column to temperatures
# temperatures["year"] = temperatures["date"].dt.year

# # Pivot avg_temp_c by country and city vs year
# temp_by_country_city_vs_year = temperatures.pivot_table(values="avg_temp_c", index=["country", "city"], columns="year", fill_value=0)

# # See the result
# print(temp_by_country_city_vs_year)


##### INDEXES

# # Look at temperatures
# print(temperatures)

# # Set the index of temperatures to city
# temperatures_ind = temperatures.set_index("city")

# # Look at temperatures_ind
# print(temperatures_ind)

# # Reset the temperatures_ind index, keeping its contents
# print(temperatures_ind.reset_index())

# # Reset the temperatures_ind index, dropping its contents
# print(temperatures_ind.reset_index(drop=True))

# # Make a list of cities to subset on
# cities = ["Moscow", "Saint Petersburg"]

# # Subset temperatures using square brackets
# print(temperatures[temperatures["city"].isin(cities)])

# # Subset temperatures_ind using .loc[]
# print(temperatures_ind.loc[cities])

# # Index temperatures by country & city
# temperatures_ind = temperatures.set_index(["country", "city"])

# # List of tuples: Brazil, Rio De Janeiro & Pakistan, Lahore
# rows_to_keep = [("Brazil", "Rio De Janeiro"), ("Pakistan", "Lahore")]

# # Subset for rows to keep
# print(temperatures_ind.loc[rows_to_keep])

# # Sort temperatures_ind by index values
# print(temperatures_ind.sort_index())

# # Sort temperatures_ind by index values at the city level
# print(temperatures_ind.sort_index(level="city"))

# # Sort temperatures_ind by country then descending city
# print(temperatures_ind.sort_index(level=["country", "city"], ascending=[True, False]))

# # Sort the index of temperatures_ind
# temperatures_srt = temperatures_ind.sort_index()

#------------------ SLICING PIVOT TABLES ----------------------------

# # Subset for Egypt to India
# temp_by_country_city_vs_year.loc["Egypt" : "India"]

# # Subset for Egypt, Cairo to India, Delhi
# temp_by_country_city_vs_year.loc[("Egypt", "Cairo") : ("India",  "Delhi")]

# # Subset for Egypt, Cairo to India, Delhi, and 2005 to 2010
# temp_by_country_city_vs_year.loc[("Egypt", "Cairo") : ("India",  "Delhi"), "2005" : "2010"]

#------------ Calculating on Pivots
# # Get the worldwide mean temp by year
# mean_temp_by_year = temp_by_country_city_vs_year.mean()

# # Filter for the year that had the highest mean temp
# print(mean_temp_by_year[mean_temp_by_year == mean_temp_by_year.max()])

# # Get the mean temp by city
# mean_temp_by_city = temp_by_country_city_vs_year.mean(axis="columns")

# # Filter for the city that had the lowest mean temp
# print(mean_temp_by_city[mean_temp_by_city == mean_temp_by_city.min()])

#####################SLICING

# # Subset rows from Pakistan to Russia
# print(temperatures_srt.loc["Pakistan" : "Russia"])

# # Try to subset rows from Lahore to Moscow
# print(temperatures_srt.loc["Lahore" : "Moscow"])

# # Subset rows from Pakistan, Lahore to Russia, Moscow
# print(temperatures_srt.loc[("Pakistan", "Lahore") : ("Russia", "Moscow")])

# # Subset rows from India, Hyderabad to Iraq, Baghdad
# print(temperatures_srt.loc[("India", "Hyderabad") : ("Iraq", "Baghdad")])

# # Subset columns from date to avg_temp_c
# print(temperatures_srt.loc[:, "date" : "avg_temp_c"])

# # Subset in both directions at once
# print(temperatures_srt.loc[("India", "Hyderabad") : ("Iraq", "Baghdad"), "date" : "avg_temp_c"])

# # Use Boolean conditions to subset temperatures for rows in 2010 and 2011
# temperatures_bool = temperatures[(temperatures["date"] >= "2010-01-01") & (temperatures["date"] <= "2011-12-31")]
# print(temperatures_bool)

# # Set date as the index and sort the index
# temperatures_ind = temperatures.set_index("date").sort_index()

# # Use .loc[] to subset temperatures_ind for rows in 2010 and 2011
# print(temperatures_ind.loc["2010" : "2011"])

# # Use .loc[] to subset temperatures_ind for rows from Aug 2010 to Feb 2011
# print(temperatures_ind.loc["2010-08" : "2011-02"])

# # Get 23rd row, 2nd column (index 22, 1)
# print(temperatures.iloc[22:23, 1:2])

# # Use slicing to get the first 5 rows
# print(temperatures.iloc[:5])

# # Use slicing to get columns 3 to 4
# print(temperatures.iloc[:, 2:4])

# # Use slicing in both directions at once
# print(temperatures.iloc[:5, 2:4])


#------------------- GRAPHS, IMPORT AND EXPORT DATA

# # Import matplotlib.pyplot with alias plt
# import matplotlib.pyplot as plt

# # Look at the first few rows of data
# print(avocados.head())

# # Get the total number of avocados sold of each size
# nb_sold_by_size = avocados.groupby("size")["nb_sold"].sum()
# print(nb_sold_by_size)

# # Create a bar plot of the number of avocados sold by size
# nb_sold_by_size.plot(kind="bar")

# # Show the plot
# plt.show()

# # Import matplotlib.pyplot with alias plt
# import matplotlib.pyplot as plt

# # Get the total number of avocados sold on each date
# nb_sold_by_date = avocados.groupby("date")["nb_sold"].sum()


# # Create a line plot of the number of avocados sold by date
# nb_sold_by_date.plot(kind="line")

# # Show the plot
# plt.show()

# # Scatter plot of avg_price vs. nb_sold with title
# avocados.plot(kind="scatter", x="nb_sold", y="avg_price", title="Number of avocados sold vs. average price")

# # Show the plot
# plt.show()

# # Modify histogram transparency to 0.5 
# avocados[avocados["type"] == "conventional"]["avg_price"].hist(alpha=0.5)

# # Modify histogram transparency to 0.5
# avocados[avocados["type"] == "organic"]["avg_price"].hist(alpha=0.5)

# # Add a legend
# plt.legend(["conventional", "organic"])

# # Show the plot
# plt.show()

# # Modify bins to 20
# avocados[avocados["type"] == "conventional"]["avg_price"].hist(alpha=0.5, bins=20)

# # Modify bins to 20
# avocados[avocados["type"] == "organic"]["avg_price"].hist(alpha=0.5, bins=20)

# # Add a legend
# plt.legend(["conventional", "organic"])

# # Show the plot
# plt.show()


#----------------------MISSING VALUES

# # Import matplotlib.pyplot with alias plt
# import matplotlib.pyplot as plt

# # Check individual values for missing values
# print(avocados_2016.isna())

# # Check each column for missing values
# print(avocados_2016.isna().sum())

# # Bar plot of missing values by variable
# avocados_2016.isna().sum().plot(kind="bar")

# # Show plot
# plt.show()

# # Remove rows with missing values
# avocados_complete = avocados_2016.dropna()

# # Check if any columns contain missing values
# print(avocados_complete.isna().any())

# From previous step
# cols_with_missing = ["small_sold", "large_sold", "xl_sold"]
# avocados_2016[cols_with_missing].hist()
# plt.show()

# # Fill in missing values with 0
# avocados_filled = avocados_2016.fillna(0)

# # Create histograms of the filled columns
# avocados_filled[cols_with_missing].hist()

# # Show the plot
# plt.show()


#------------------------CREATE DATAFRAMES

# # Create a list of dictionaries with new data
# avocados_list = [
#     {"date": "2019-11-03", "small_sold": 10376832, "large_sold": 7835071},
#     {"date": "2019-11-10", "small_sold": 10717154, "large_sold": 8561348},
#     ]

# # Convert list into DataFrame
# avocados_2019 = pd.DataFrame(avocados_list)

# # Print the new DataFrame
# print(avocados_2019)

# Create a dictionary of lists with new data
# avocados_dict = {
#   "date": ["2019-11-17", "2019-12-01"],
#   "small_sold": [10859987, 9291631],
#   "large_sold": [7674135, 6238096]
# }

# # Convert dictionary into DataFrame
# avocados_2019 = pd.DataFrame(avocados_dict)

# # Print the new DataFrame
# print(avocados_2019)


# ----------------------- READING AND WRITING CSV

# # Read CSV as DataFrame called airline_bumping
# airline_bumping = pd.read_csv("airline_bumping.csv")

# # Take a look at the DataFrame
# print(airline_bumping)

# # From previous steps
# airline_bumping = pd.read_csv("airline_bumping.csv")
# print(airline_bumping.head())
# airline_totals = airline_bumping.groupby("airline")[["nb_bumped", "total_passengers"]].sum()

# # Create new col, bumps_per_10k: no. of bumps per 10k passengers for each airline
# airline_totals["bumps_per_10k"] = airline_totals["nb_bumped"] / airline_totals["total_passengers"] * 10000

# print(airline_totals)

# # Create airline_totals_sorted
# airline_totals_sorted = airline_totals.sort_values("bumps_per_10k", ascending=False)

# # Print airline_totals_sorted
# print(airline_totals_sorted)

# # Save as airline_totals_sorted.csv
# airline_totals_sorted.to_csv("airline_totals_sorted.csv")