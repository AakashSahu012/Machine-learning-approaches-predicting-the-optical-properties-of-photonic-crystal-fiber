################# Import the library ############################

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
############# for check values in excel data file #####################

df=pd.read_excel("/content/sample_data/pcf_data.xlsx", sheet_name="SiO2-air-rings-4-dBYp-0.7")
dataset=df.values
# print(dataset.shape)
# xls=pd.ExcelFile("/aeftent/pcf_modeSoln_data_1(
# xls.sheet_names
sheet_names=['SiO2-air-rings-4-dBYp-0.8',
                   'SiO2-air-rings-4-dBYp-0.9',
                   'SiO2-air-rings-4-dBYp-0.6',
                   'SiO2-air-rings-5-dBYp-0.7',
                   'SiO2-air-rings-5-dBYp-0.8',
                   'SiO2-air-rings-5-dBYp-0.9',
                   'SiO2-air-rings-5-dBYp-0.6',]
for sheetname in sheet_names:
  dataset_sheet=pd.read_excel("/content/sample_data/pcf_data.xlsx", sheet_name=sheetname)
  df_sheet=dataset_sheet.values
  # print(dataset_sheet.shape)
  dataset=np.concatenate((dataset, df_sheet), axis=0)
# print(dataset)
###########  save combin data into excel file ###############
# print(dataset.shape)
# df_excel=pd.DataFrame(dataset)
# df_excel.to_excel("output_data.xlsx",index=False)

######### preprocessing the data ###########
####### for effective index #########
num_out_var=1
nef_out=dataset[:,6:7]
nef_out=nef_out.reshape((-1,num_out_var))
len(nef_out)
# print(nef_out)


########### for aeffintment loss #########
aef_out=dataset[:,9:10]
aef_out=aef_out.reshape((-1,num_out_var))
len(aef_out)
# print(aef_out)


########## standardize the data (fit) ###########
scale_data1=StandardScaler()
scale_data2=StandardScaler()
scale_data3=StandardScaler()
scale_data4=StandardScaler()
scale_data1.fit(dataset)
scale_data2.fit(nef_out)
scale_data3.fit(aef_out)
scale_data4.fit(dataset[:,0:6])
########### First standardize the data (fit) then transform the data ############
scaler_dataset=scale_data1.transform(dataset)
x=scaler_dataset[:,0:6]
x=x.reshape(-1,6)
y=scaler_dataset[:,[6,7]]
y=y.reshape(-1,2)
x,y=shuffle(x,y)

# # print(x)
# print(y)
y_nef=y[:,0:1]

############## import linear Reg. and split it into train and test data #############
lr_model=LinearRegression()


######## trainig and validating data ###########
x_train,x_validation,y_train,y_validation=train_test_split(x,y_nef,test_size=0.2)
lr_model.fit(x_train,y_train)
# print(lr_model.intercept_)
# print(np.mean(lr_model.coef_))
y_nef_pred=lr_model.predict(x_validation)
# print(y_nef_pred)
# y_val_pred=lr_model.predict(x_validation)
# mse_val=mean_squared_error(y_validation,y_val_pred)
# print("val mse :",mse_val)
# r2=r2_score(y_validation,y_val_pred)
# print("r2",r2)
mse= mean_squared_error(y_validation,y_nef_pred)
mae=mean_absolute_error(y_validation,y_nef_pred)
r_square=r2_score(y_validation,y_nef_pred)
rmse=np.sqrt(mse)
print("mean_square_error with nef:",mse)
print("mean_absolute_error with nef:",mae)
print("rmse with nef :",rmse)
print("r2 score with nef:",r_square)




# ################# another feature effective mode area #############
y_aef=y[:,[1]]
# print(y_aef)
x_aef_train,x_aef_validation,y_aef_train,y_aef_validation=train_test_split(x,y_aef,test_size=0.2)
lr_model.fit(x_aef_train,y_aef_train)
# print(lr_model.intercept_)
# print(np.mean(lr_model.coef_))
y_aef_pred=lr_model.predict(x_aef_validation)
# print(y_aef_pred)
mse_aef= mean_squared_error(y_aef_validation,y_aef_pred)
mae_aef=mean_absolute_error(y_aef_validation,y_aef_pred)
r_square_aef=r2_score(y_aef_validation,y_aef_pred)
rmse_aef=np.sqrt(mse_aef)
print("mean_square_error with aef:",mse_aef)
print("mean_absolute_error with aef:",mae_aef)
print("rmse with aef :",rmse_aef)
print("r2 score with aef:",r_square_aef)

# print(max(y_nef_pred))


########## Taking manual data for testing the model ##############
# df2=pd.read_excel("/content/sample_data/pcf_modeSoln_data_manual_1.xlsx",sheet_name="Sheet1")
# dataset2=df.values
# # print(dataset2)

# scaler_data_man=scale_data1.transform(dataset2)
# # print(scaler_data_man)
# x_test=scaler_data_man[:,0:6]
# x_test=x_test.reshape(-1,6)
# y_test=scaler_data_man[:,6:11]
# y_test=y_test.reshape(-1,5)
# y_test_nef=y_test[:,0:1]
# y_test_pred=lr_model.predict(x_test)
# mse1= mean_squared_error(y_test_nef,y_test_pred)
# # mae=mean_absolute_error(y_test_nef,y_pred)
# r_square1=r2_score(y_test_nef,y_test_pred)
# # # rmse=np.sqrt(mse)
# print("mean_square_error with nefhih:",mse1)
# # print("mean_absolute_error with nef:",mae)
# # print("rmse:",rmse)
# print("r2 score with nef:",r_square1)
# # print(y_validation.shape)

# print(out_data1)




################# Decision Tree Regressor##################

####### effective index #######
dt_model=DecisionTreeRegressor(criterion="squared_error")
x_dt_train,x_dt_test,y_dt_train,y_dt_test=train_test_split(x,y_nef,test_size=0.2)
dt_model.fit(x_dt_train,y_dt_train)
y_dt_pred=dt_model.predict(x_dt_test)
mse_dt=mean_squared_error(y_dt_test,y_dt_pred)
print("mse_dt",mse_dt)
mae_dt=mean_absolute_error(y_dt_test,y_dt_pred)
print("mae_dt",mae_dt)
rmse_dt=np.sqrt(mse_dt)
print("rmse_dt",rmse_dt)
r_score_dt=r2_score(y_dt_test,y_dt_pred)
print("r2_score",r_score_dt)
# print(y_dt_pred)

####### effective mode of area #######
x_dt_a_train,x_dt_a_test,y_dt_a_train,y_dt_a_test=train_test_split(x,y_aef,test_size=0.2)
dt_model.fit(x_dt_a_train,y_dt_a_train)
y_dt_a_pred=dt_model.predict(x_dt_a_test)
mse_dt_a=mean_squared_error(y_dt_a_test,y_dt_a_pred)
print("mse_area",mse_dt_a)
mae_dt_a=mean_absolute_error(y_dt_a_test,y_dt_a_pred)
print("mae_area",mae_dt_a)
rmse_dt_a=np.sqrt(mse_dt_a)
print("rmse_dt_a",rmse_dt_a)
r_score_dt_a=r2_score(y_dt_a_test,y_dt_a_pred)
print("r_score_dt_a",r_score_dt_a)





############## Ploting nef b/w actual and predict ##################

# Rescale
y_nef_pred_rescaled = scale_data2.inverse_transform(y_nef_pred)
y_validation_rescaled = scale_data2.inverse_transform(y_validation)
x_wv_validation_rescaled = scale_data2.inverse_transform(x_validation[:,[5]])
plt.figure(figsize=(8,5),facecolor='yellow')
plt.scatter(y_validation_rescaled, y_nef_pred_rescaled, color='red',marker='o',facecolors='red', label='Predicted nef',edgecolor='blue',s=65)
plt.plot([min(y_validation_rescaled), max(y_validation_rescaled)],
         [min(y_validation_rescaled), max(y_validation_rescaled)],
         color='green', linestyle='--', label='Perfectly predicted')
plt.title('Actual nef vs Predicted nef',fontsize=20)
plt.xlabel('Actual nef',fontsize=15)
plt.ylabel('Predicted nef',fontsize=15)
plt.legend(loc="best",fontsize=15)
plt.xlim(1.30,1.47)
plt.ylim(1.30,1.52)
plt.grid(True)
plt.savefig("/content/sample_data/plot1.png")
plt.show()

################# Plotting Aef- predicted and actual ##########

# rescaled the values

y_aef_pred_rescaled = scale_data2.inverse_transform(y_aef_pred)
y_aef_validation_rescaled = scale_data2.inverse_transform(y_aef_validation)
plt.figure(figsize=(8,5),facecolor='yellow')
# ax=plt.figure()
plt.scatter(y_aef_validation_rescaled, y_aef_pred_rescaled, color='red',marker='o',facecolors='red', label='Predicted aef',edgecolor='green',s=50)
plt.plot([min(y_aef_validation_rescaled), max(y_aef_validation_rescaled)],
         [min(y_aef_validation_rescaled), max(y_aef_validation_rescaled)],
         color='blue', linestyle='--', label='Perfectly predicted')
plt.title('Actual aef vs Predicted aef',fontsize=20)
plt.xlabel('Actual aef',fontsize=15)
plt.ylabel('Predicted aef',fontsize=15)
plt.legend(loc='best',fontsize=15)
# plt.xlim(1.30,1.60)
# plt.ylim(1.3,1.60)
plt.grid(True)
plt.savefig("/content/sample_data/plot2.png")
plt.show()



#############  Plotting DT nef b/w actual and predict

y_pred_dt_rescale=scale_data2.inverse_transform([y_dt_pred])
y_dt_test_rescale=scale_data2.inverse_transform(y_dt_test)
plt.figure(figsize=(8,5),facecolor="yellow")
plt.scatter(y_dt_test_rescale,y_pred_dt_rescale,color="green",marker="o",facecolor="red",label="predicted_dt_nef",edgecolors="blue",s=50)
plt.plot([min(y_dt_test_rescale), max(y_dt_test_rescale)],
         [min(y_dt_test_rescale), max(y_dt_test_rescale)],color='blue',linestyle='--',label='prefecttly predicted')
plt.title('actual dt nef vs predicted dt nef', fontsize=20)
plt.ylabel('Predicted dt nef',fontsize=15)
plt.legend(loc='best',fontsize=15)
# # plt.xlim(1.30,1.60)
# # plt.ylim(1.3,1.60)
plt.grid(True)
plt.savefig("/content/sample_data/plot3.png")
plt.show()


############## Plotting DT aef b/w actual and predict ##########

y_pred_dt_rescale22=scale_data2.inverse_transform([y_dt_a_pred])
y_dt_test_rescale=scale_data2.inverse_transform(y_dt_a_test)
plt.figure(figsize=(8,5),facecolor="yellow")
plt.scatter(y_dt_test_rescale,y_pred_dt_rescale22,color="green",marker="o",facecolor="red",label="predicted_dt_aef",edgecolors="green",s=50)
plt.plot([min(y_dt_test_rescale), max(y_dt_test_rescale)],
         [min(y_dt_test_rescale), max(y_dt_test_rescale)],color='blue',linestyle='--',label='prefecttly predicted')
plt.title('actual dt aef vs predicted dt aef', fontsize=20)
plt.ylabel('Predicted dt aef',fontsize=15)
plt.legend(loc='best',fontsize=15)
# # plt.xlim(1.30,1.60)
# # plt.ylim(1.3,1.60)
plt.grid(True)
plt.savefig("/content/sample_data/plot4.png")
plt.show()






############ Plotting b/w regression and decision tree of nef ##########

# plt.figure(figsize=(8,5),facecolor="yellow")
# plt.scatter(y_validation_rescaled,y_nef_pred_rescaled,color="green",marker="^",facecolor="blue",label="predicted_dt_aef",edgecolors="green",s=50)

# plt.scatter(y_validation_rescaled,y_pred_dt_rescale,color="green",marker="o",facecolor="red",label="predicted_dt_aef",edgecolors="green",s=50)
# plt.plot([min(y_validation_rescaled), max(y_validation_rescaled)],
#          [min(y_validation_rescaled), max(y_validation_rescaled)],color='blue',linestyle='--',label='prefecttly predicted')
# plt.title('actual dt aef vs predicted dt aef', fontsize=20)
# plt.ylabel('Predicted dt aef',fontsize=15)
# plt.legend(loc='best',fontsize=15)
# # # plt.xlim(1.30,1.60)
# # # plt.ylim(1.3,1.60)
# plt.grid(True)
# plt.show()


# mse_bw_algo=mean_absolute_error(y_nef_pred,y_dt_pred)
# print(mse_bw_algo)
