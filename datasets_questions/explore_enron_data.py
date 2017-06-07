#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# Size of data
print("Number of employees: " + str(len(enron_data))) # 146

# Number of features/employee
print("Number of features: " + str(len(enron_data.values()[0]))) # 21

# Number of POIs (enron_data[employee][poi] == 1)
count = 0
for employee in enron_data.values():
    if employee["poi"] == 1:
        count += 1

print("Number of POIs: " + str(count)) # 18

# How many POIs are in ../final_project/poi_names.txt
# 35
# We are missing some, the main issue with this is not having enough
# data to effectively learn the patterns.

#### Querying some of the data

# What is the total value of the stock belonging to James Prentice?
print("James Prentice total stock value: " + str(enron_data["PRENTICE JAMES"]["total_stock_value"])) # 1095040
# How many email messages do we have from Wesly Colwell to POIs?
print("Wesly Colwell emails to POIs: " + str(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])) # 11
# What's the value of stock options exercised by Jeff Skilling?
print("Jeff Skilling exercised stock options: "
        + str(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])) # 19250000
# Some People:
# Jeffrey Skilling - CEO
# Kenneth Lay - Chairman
# Andrew Fastow - CFO

# Of the three above, who took home the most money (total_payments)? How much?
print("J.Skilling take home pay: " + str(enron_data["SKILLING JEFFREY K"]["total_payments"])) # 8682716
print("K.Lay take home pay: " + str(enron_data["LAY KENNETH L"]["total_payments"])) # 103559793
print("A.Fastow take home pay: " + str(enron_data["FASTOW ANDREW S"]["total_payments"])) # 2424083
# Lay took home a lot of money =/

# For nearly every person in the dataset, not every feature has a value. How is it denoted when a feature
# doesn't have a well-defined value

# print(enron_data.values()[0])
# NaN

# How many folks in this dataset have a quantified salary? Known email address?
numEmails = 0
numSalary = 0
for employee in enron_data.values():
    if employee["salary"] != "NaN":
        numSalary += 1
    if employee["email_address"] != "NaN":
        numEmails += 1

print("Number of employees with salary: " + str(numSalary)) # 95
print("Number of employees with emails: " + str(numEmails)) # 111

#### The dataset is missing some values and some POIs. While it would be straight forward to add
#### them into our dataset, this could introduce a subtle problem (walkthrough). Note there is
#### limited data (not as much as the spreadsheet)

# What percentage of people in the dataset have "NaN" for their total payments?
numNoPayments = 0.0
for employee in enron_data.values():
    if employee["total_payments"] == "NaN":
        numNoPayments += 1

print("Percentage of employees with missing payments: " + str(numNoPayments/146 * 100)) # 21/146 = 14.38%

# What percentage of POIs in the dataset have "NaN" for their total payments?
numNoPayments = 0.0
for employee in enron_data.values():
    if employee["poi"] == 1 and employee["total_payments"] == "NaN":
        numNoPayments += 1

print("Percentage of POIs with missing payments: " + str(numNoPayments/18 * 100)) # 0/18 = 0.0%

# If a ML algorithm were to use total_payments as a feature, would you expect it to associate a "NaN"
# value with POIs or non-POIs
# Non-POIs because there are no NaN values for existing POIs

# If you add 10 data points which were all POIs and put NaN in the total pamyents, what is the new number
# of people? what is the new number of folks with "NaN" for total payments
# Num People = 156, Num with NaN = 31

# What is the new number of POIs in the dataset?
# 28
# What is the new number of POIs with NaN for total_payments?
# 10

# With the new data points, would a supervised learning algorithm classify a "NaN" as a POI?
# Yes - it could

## Summary - because we are using financial data to classify whether an individual is a POI
## or not a POI, this introduces bias into our classified because the two classes are being imported
## from different sources (POI and non-POI from spreadsheet, POI by hand). The learning algorithm
## can't differentiate between the data sources and might think that missing financial data is an indicator
## that an individual is a POI. Be careful when mixing data sources!!
