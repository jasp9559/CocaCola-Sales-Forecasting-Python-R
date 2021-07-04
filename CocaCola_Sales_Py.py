import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
# from datetime import datetime

cocacola = pd.read_excel("C:/Data Science/Data Science/Assignments/Ass24. Forecasting/Coca Cola/CocaCola_Sales_Rawdata.xlsx")

cocacola.Sales.plot() # time series plot 

# Centering moving average for the time series
cocacola.Sales.plot(label = "org")
for i in range(2, 9, 2):
    cocacola["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(cocacola.Sales, model = "additive", period = 4)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(cocacola.Sales, model = "multiplicative", period = 4)
decompose_ts_mul.plot()

# ACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cocacola.Sales, lags = 4)
# tsa_plots.plot_pacf(cocacola.Sales, lags=4)

# splitting the data into Train and Test data
# Recent 4 time period values are Test data

Train = cocacola.head(38)
Test = cocacola.tail(4)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,4),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_ses, Test.Sales) 

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.Sales) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.Sales) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.Sales) 

# Visualization of Forecasted values for Test data set using different methods 
plt.plot(Train.index, Train["Sales"], label='Train',color="black")
plt.plot(Test.index, Test["Sales"], label='Test',color="blue")
plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")
plt.legend(loc='best')

# Final Model on 100% Data
hwe_modelori_mul_add = ExponentialSmoothing(cocacola["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()

# Load the new data which includes the entry for future 4 values
new_data = pd.read_excel("C:/Data Science/Data Science/Assignments/Ass24. Forecasting/Coca Cola/Newdata_CocaCola_Sales.xlsx")

newdata_pred = hwe_modelori_mul_add.predict(start = new_data.index[42], end = new_data.index[-1])
newdata_pred

##########################################################
##################################################
#################################

import pandas as pd
cocacola = pd.read_excel("C:/Data Science/Data Science/Assignments/Ass24. Forecasting/Coca Cola/CocaCola_Sales_Rawdata.xlsx")
Quarters = ['Q1','Q2','Q3','Q4'] 

# Pre processing
import numpy as np

cocacola["t"] = np.arange(0, 42)

cocacola["t_square"] = cocacola["t"]*cocacola["t"]
cocacola["log_Sales"] = np.log(cocacola["Sales"])
cocacola.columns

p = cocacola["Quarter"][0]
p[0:2]

cocacola['Quarters'] = 0

for i in range(41):
    p = cocacola["Quarter"][i]
    cocacola['Quarters'][i]= p[0:2]
    
Qtr_dummies = pd.DataFrame(pd.get_dummies(cocacola['Quarters']))
cocacola1 = pd.concat([cocacola, Qtr_dummies], axis = 1)

# Visualization - Time plot
cocacola1.Sales.plot()

# Data Partition
Train = cocacola1.head(38)
Test = cocacola1.tail(4)

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales ~ t', data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_Sales ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Sales ~ t+t_square', data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Sales ~ Q1 + Q2 + Q3', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Sales ~ Q1 + Q2 + Q3',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

################## Additive Seasonality Linear Trend ############################

add_sea_lin = smf.ols('Sales ~ t + Q1 + Q2 + Q3', data=Train).fit()
pred_add_sea_lin = pd.Series(add_sea_lin.predict(Test[['Q1', 'Q2', 'Q3', 't']]))
rmse_add_sea_lin = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_lin))**2))
rmse_add_sea_lin 

################## Additive Seasonality Quadratic Trend ############################

add_sea_Quad = smf.ols('Sales ~ t + t_square + Q1 + Q2 + Q3',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1', 'Q2', 'Q3', 't', 't_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality Linear Trend  ###########

Mul_Add_sea = smf.ols('log_Sales ~ t + Q1 + Q2 + Q3',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Multiplicative Seasonality Quadratic Trend  ###########

Mul_Add_sea_quad = smf.ols('log_Sales ~ t + t_square + Q1 + Q2 + Q3', data = Train).fit()
pred_Mult_add_sea_quad = pd.Series(Mul_Add_sea_quad.predict(Test))
rmse_Mult_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea_quad)))**2))
rmse_Mult_add_sea_quad 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_lin","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea","rmse_Mult_add_sea_quad"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_lin,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea,rmse_Mult_add_sea_quad])}
table_rmse=pd.DataFrame(data)
table_rmse

# 'rmse_add_sea_lin' has the least value among the models prepared so far Predicting new values 
pred_data = pd.read_csv("C:/Data Science/Data Science/Assignments/Ass24. Forecasting/Coca Cola/cocacolaSales_pred.csv")

model_full = smf.ols('Sales ~ t + Q1 + Q2 + Q3', data = cocacola1).fit()

pred_new  = pd.Series(model_full.predict(pred_data))
pred_new

pred_data["forecasted_Sales"] = pd.Series(pred_new)

###########################################################
#################################################
#############################
import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima_model import ARIMA

cocacola_raw = pd.read_excel("C:/Data Science/Data Science/Assignments/Ass24. Forecasting/Coca Cola/CocaCola_Sales_Rawdata.xlsx")

tsa_plots.plot_acf(cocacola_raw.Sales, lags = 4)
# tsa_plots.plot_pacf(Walmart.Footfalls,lags=12)

model1 = ARIMA(cocacola_raw.Sales, order = (1, 1, 4)).fit(disp = 0)
model2 = ARIMA(cocacola_raw.Sales, order = (0, 1, 1)).fit(disp = 0)
model1.aic
model2.aic

p = 1
q = 0
d = 1

pdq = []
aic = []
for q in range(9):
    try:
        model = ARIMA(cocacola_raw.Sales, order = (p, d, q)).fit(disp = 0)
        x=model.aic
        x1= p,d,q
        aic.append(x)
        pdq.append(x1)
    except:
        pass
            
keys = pdq
values = aic
d = dict(zip(keys, values))
print (d)
