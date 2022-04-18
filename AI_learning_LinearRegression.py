from turtle import pd
from sklearn.linear_model import LinearRegression
from openpyxl import load_workbook
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split 


wb = load_workbook(r"C:\Users\alqas\Research\SHEtable2Vdc.xlsx")
ws = wb.active

x_vdc1=[]
y_vdc2=[]
z_t1=[[],[],[],[],[],[]]
#collecting the data from the columns we need
for col in ws.iter_cols(min_row=2, max_col=2 ,min_col=2, values_only=True):
    for row in col:
        x_vdc1.append(row)
for col in ws.iter_cols(min_row=2, max_col=3 ,min_col=3, values_only=True):
    for row in col:
        y_vdc2.append(row)
for g in range(1,7):
    for col in ws.iter_cols(min_row=2, max_col=g+3 ,min_col=g+3, values_only=True):
        for row in col:
            z_t1[g-1].append(row)

#reorganizing the data for ease of axis (not needed could just call the array itself)
Data = {'x': x_vdc1,
        'y': y_vdc2,
        'z_1': z_t1[0],
        'z_2': z_t1[1],
        'z_3': z_t1[2],
        'z_4': z_t1[3],
        'z_5': z_t1[4],
        'z_6': z_t1[5],
        }
df = pd.DataFrame(Data,columns=['x','y','z_1','z_2','z_3','z_4','z_5','z_6'])
input=df[['x','y']]

#in order to look at more outputs, just add the Z_number value for it to the list bellow
output = df[['z_1','z_2','z_3']]
regression=linear_model.LinearRegression()
regression.fit(input.values,output.values)

#The print statment bellow can be uncommented in order to allow it to print the itercepts and coeficents of the function
#print(regression.intercept_, regression.coef_)

#in here are the input values used to predict the function
new_x =112
new_y = 120
print(regression.predict([[new_x, new_y]]))

# it is missing the test/train split
# add a plot of z_1 from linear regression model vs actual
# need a measure of error, performance: lets look at MSE first
