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

def add_zero_to_county_num(df, id, cost):

    raw_dict = df.to_dict('index')

    conum_array = []
    per_student_cost_array = []

    for key in raw_dict:
        conum = raw_dict[key][id]
        per_student_cost = raw_dict[key][cost]
        if len(conum) == 4:
            conum = '0' + conum
        
        conum_array.append(conum)
        per_student_cost_array.append(per_student_cost)

    final_dict = {id:conum_array, cost:per_student_cost_array}

    df_final = pd.DataFrame.from_dict(final_dict)

    return df_final

def get_cost_per_student_county():

    df_all = pd.read_csv('edspending_county.CSV')

    df = df_all[['CONUM','PPCSTOT']].copy()

    df.rename(columns={"CONUM": "county_num", "PPCSTOT": "per_student_cost"}, inplace = True)

    df = df.astype({"county_num": str})

    df_final = add_zero_to_county_num(df, 'county_num', 'per_student_cost')

    print(df_final.head())

get_cost_per_student_county()

def merge_by_state():

    df_cost_index = get_cost_index()
    df_cost_per_student = get_cost_per_student()
    df_merge = pd.merge(df_cost_index, df_cost_per_student, on='State')

    return df_merge

def plot_cost_by_state(df):

    ax = df.plot.scatter(x='costIndex', y='amountPerPupil', title='Cost of Living Index and Cost per Pupil')

    plt.xlabel('Cost of Living Index')
    plt.ylabel('Cost per Pupil')

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