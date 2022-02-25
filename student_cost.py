import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr
import math

plt.style.use('ggplot')

def get_cost_index():

    df_all = pd.read_csv('cost_of_living.CSV')

    df = df_all[['State', 'costIndex']].copy()

    return df

def get_cost_per_student():

    df = pd.read_csv('edspending_state.CSV')

    return df

def add_zero_to_county_num(df, id, measure):

    raw_dict = df.to_dict('index')

    conum_array = []
    measure_array = []

    for key in raw_dict:
        conum = raw_dict[key][id]
        measure_instance = raw_dict[key][measure]
        if len(conum) == 4:
            conum = '0' + conum
        
        conum_array.append(conum)
        measure_array.append(measure_instance)

    final_dict = {id:conum_array, measure: measure_array}

    df_final = pd.DataFrame.from_dict(final_dict)

    return df_final

def get_cost_per_student_county():

    df_all = pd.read_csv('edspending_county.CSV')

    df = df_all[['CONUM','PPCSTOT']].copy()

    df.rename(columns={"CONUM": "county_num", "PPCSTOT": "per_student_cost"}, inplace = True)

    #df = df.astype({"county_num": str})

    #df_final = add_zero_to_county_num(df, 'county_num', 'per_student_cost')

    return df

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

    print(df_merge.dtypes)

    return df_merge

def plot_cost_by_county(df):

    ax = df.plot.scatter(x='poverty_percent', y='per_student_cost', title='Poverty Rate and Cost per Pupil by County')

    #myLocator = mtick.MultipleLocator(20)
    #ax.xaxis.set_major_locator(myLocator)

    plt.xlabel('County Poverty Rate')
    plt.ylabel('Cost per Student')

    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)

    #regression(df, 'costIndex', 'amountPerPupil')

    plt.show()

def merge_by_state():

    df_cost_index = get_cost_index()
    df_cost_per_student = get_cost_per_student()
    df_merge = pd.merge(df_cost_index, df_cost_per_student, on='State')

    return df_merge

def plot_cost_by_state(df):

    ax = df.plot.scatter(x='costIndex', y='amountPerPupil', title='Cost of Living Index and Cost per Pupil')

    plt.xlabel('Cost of Living Index')
    plt.ylabel('Cost per Student')

    #fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    #yticks = mtick.FormatStrFormatter(fmt)
    #ax.yaxis.set_major_formatter(yticks)

    regression(df, 'costIndex', 'amountPerPupil')

    plt.show()

def regression(df, x, y):

    X = df.iloc[:, 1].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df.iloc[:, 2].values.reshape(-1, 1)  # -1 means you aren't sure the number of rows, but should have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions

    plt.plot(X, Y_pred, color='red')

    plt.show()

    x_list = df[x].values.tolist()
    y_list = df[y].values.tolist()

    print(pearsonr(x_list,y_list))

#df = merge_by_state()
#plot_cost_by_state(df)

df = merge_by_county()
plot_cost_by_county(df)