from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr
import math

plt.style.use('ggplot')

def regression(df, x, y):

    X = df.loc[:, x].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df.loc[:, y].values.reshape(-1, 1)  # -1 means you aren't sure the number of rows, but should have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions

    plt.plot(X, Y_pred, color='red')

    plt.show()

    x_list = df[x].values.tolist()
    y_list = df[y].values.tolist()

    print(pearsonr(x_list,y_list))

def currency(x, pos):
    """The two args are the value and tick position"""
    if x >= 1e6:
        s = '${:1.1f}M'.format(x*1e-6)
    else:
        s = '${:1.0f}K'.format(x*1e-3)
    return s

def get_cost_index():

    df_all = pd.read_csv('cost_of_living.CSV')

    df = df_all[['State', 'costIndex']].copy()

    return df

def get_salaries():

    df_all = pd.read_csv('teacher_salaries.CSV')

    df = df_all[['State', 'salary']].copy()

    df_dict = df.to_dict('index')

    state_array = []
    salary_array = []

    for key in df_dict:
        state = df_dict[key]['State']
        salary = df_dict[key]['salary']

        salary = salary.replace(',','')

        salary_float = float(salary)

        state_array.append(state)
        salary_array.append(salary_float)

    final_dict = {'State':state_array, 'salary':salary_array}

    df_final = pd.DataFrame.from_dict(final_dict)

    return df_final

def get_cost_per_student():

    df = pd.read_csv('edspending_state.CSV')

    return df

def merge_by_state():

    df_cost_index = get_cost_index()
    df_cost_per_student = get_cost_per_student()
    df_merge = pd.merge(df_cost_index, df_cost_per_student, on='State')

    return df_merge

def merge_by_state_salary():

    df_cost_per_student = get_cost_per_student()
    df_salary = get_salaries()
    df_merge = pd.merge(df_cost_per_student, df_salary, on='State')

    return df_merge

def get_cost_per_student_county():

    df_all = pd.read_csv('edspending_county.CSV')

    df = df_all[['CONUM','PPCSTOT', "ENROLL"]].copy()

    df.rename(columns={"CONUM": "county_num", "PPCSTOT": "per_student_cost","ENROLL":'enrollment'}, inplace = True)

    df_dict = df.to_dict('index')

    county_array = []
    cost_array = []
    enrollment_array = []

    for key in df_dict:
        county_num = df_dict[key]['county_num']
        per_student_cost = df_dict[key]['per_student_cost']
        enrollment = df_dict[key]['enrollment']
        if per_student_cost < 50000 and per_student_cost != 0 and enrollment != 0:
            county_array.append(county_num)
            cost_array.append(per_student_cost)
            enrollment_array.append(enrollment)

    final_dict = {'county_num':county_array, 'per_student_cost':cost_array,'enrollment':enrollment_array}

    df_final = pd.DataFrame.from_dict(final_dict)

    return df_final

def get_poverty_by_county():
    
    df_all = pd.read_csv('poverty_county.CSV')

    df = df_all[['County Code','Poverty Percent, All Ages','Median Household Income']]

    df.rename(columns={'County Code':'county_num','Poverty Percent, All Ages':'poverty_percent','Median Household Income':'median_income'}, inplace = True)

    county_array = []
    poverty_array = []
    median_array = []

    df_dict = df.to_dict('index')

    for key in df_dict:
        county_num = df_dict[key]['county_num']
        poverty = df_dict[key]['poverty_percent']
        median = df_dict[key]['median_income']

        median = median.replace(',','')
        if len(poverty)<=1:
            continue

        poverty_float = float(poverty)
        median_float = float(median)

        county_array.append(county_num)
        poverty_array.append(poverty_float)
        median_array.append(median_float)

    final_dict = {'county_num':county_array, 'poverty_percent':poverty_array, 'median_income':median_array}

    df_final = pd.DataFrame.from_dict(final_dict)

    return df_final

def merge_by_county():
    df_cost_per_county = get_cost_per_student_county()
    df_poverty_by_county = get_poverty_by_county()
    df_merge = pd.merge(df_cost_per_county, df_poverty_by_county, on='county_num')

    return df_merge

def plot_cost_by_county(df):

    ax = df.plot.scatter(x='poverty_percent', y='per_student_cost', title='Poverty Rate and Cost per Pupil by County')

    plt.xlabel('County Poverty Rate')
    plt.ylabel('Cost per Student')

    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)

    ax.yaxis.set_major_formatter(currency)

    plt.show()

def plot_cost_by_enrollment(df):

    ax = df.plot.scatter(x='enrollment', y='per_student_cost', title='Enrollment and Cost per Pupil by County')

    plt.xlabel('Enrollment')
    plt.ylabel('Cost per Student')

    ax.yaxis.set_major_formatter(currency)

    plt.show()

def plot_cost_by_median(df):

    ax = df.plot.scatter(x='median_income', y='per_student_cost', title='Median Income and Cost per Pupil by County')

    plt.xlabel('Median Income')
    plt.ylabel('Cost per Student')

    ax.xaxis.set_major_formatter(currency)
    ax.yaxis.set_major_formatter(currency)

    plt.show()

def plot_cost_by_state(df):

    ax = df.plot.scatter(x='costIndex', y='amountPerPupil', title='Cost of Living Index and Cost per Pupil')

    plt.xlabel('Cost of Living Index')
    plt.ylabel('Cost per Student')

    ax.yaxis.set_major_formatter(currency)

    regression(df, 'costIndex', 'amountPerPupil')

    plt.show()

def plot_cost_by_salary(df):

    ax = df.plot.scatter(x='salary', y='amountPerPupil', title='Teacher Salaries and Cost per Pupil')

    plt.xlabel('Teacher Salaries')
    plt.ylabel('Cost per Student')

    ax.xaxis.set_major_formatter(currency)
    ax.yaxis.set_major_formatter(currency)

    regression(df, 'salary', 'amountPerPupil')

    plt.show()

df = merge_by_state()
plot_cost_by_state(df)

df = merge_by_state_salary()
plot_cost_by_salary(df)

df = merge_by_county()
plot_cost_by_county(df)
plot_cost_by_median(df)

plot_cost_by_enrollment(df)