# CocaCola-Sales-Forecasting-Python-R
A forecasting problem solved using the Forecasting models, using Python and R.

A plastics manufacturing plant has recorded their monthly sales data from 1949 to 1953. Perform forecasting on the data and bring out insights from it and forecast the sale for the next year. 

Solution:

a.	We load in the data for Plastic dataset

b.	The same is plotted using plot, which plots a time series plot as the data contains time related data. Here we can see that the data has a positive trend and seasonality, i.e a positive upward trend with additive seasonality since the graph seems to be uniform.

c.	We additionally use seasonal decomposition with additive and multiplicative models for 12 months, to check where the data sits. From the two plots we witness that the data seems to be on the additive seasonality, as the plots suggest and the residuals are also correlated linearly in additive plot.

d.	We split the data into Test and Train data with test data to be selected as the same to the frequency of the data to be predicted.

e.	We assign the error as MAPE that will give the Mean Absolute Percentage Error, and define a custom function to calculate one on the smoothing methods.


f.	We try the Data driven approach first wherein the data would undergo different smoothing methods, including the Simple exponential, Holt’s method i.e Double exponential smoothing, Holt Winter’s i.e, using log, i.e using additive or Multiplicative seasonality within.

g.	We set the first Simple Exponential Smoothing over the train data and fit the same. We check for the predicted values on the test data which in-turn returns a MAPE. 

h.	Similarly we check the Holt’s method, Holt’s Winter method with additive seasonality and again with multiplicative.

i.	The MAPE are as follows:
      
			Simple Exponential Smoothing	17.04
      
			Holt’s Method	102.40
      
			Holt’s Winter method with additive seasonality	11.73
      
			Holt’s Winter method with multiplicative seasonality	14.99

j.	We see that the Holt’s winter method with additive seasonality gives us the least error.

k.	We now set the method on the actual data and build a model to predict the values.

l.	We use this model on the new data which is a manual prepared predicted data having empty rows at the bottom for the next 12 values to be predicted.

m.	We can print the same and see a set of 12 predicted values that set properly.


n.	We additionally try predicting the values using the model based technique also.

o.	We take the raw data again. We take the month names, which we concatenate with the data later.

p.	We set the time variable ‘t’, ranging from 1 to 60, as in the dataset.

q.	We additionally set the t-square column which would be the square of ‘t’ and we set the log of sales column as well. These are set for us to build models based on the various seasonality as under.

r.	Now we split the data into train and test and start building models viz, Linear, Exponential, Quadratic, Additive Seasonality, Multiplicative Seasonality, Additive Seasonality with Quadratic Trend and Multiplicative Seasonality with Quadratic Trend. Using these models we calculate the RMSE values and check for the least value.

s.	To build the above models we  use OLS, i.e Ordinary Least square formula from statsmodel.formula as under:

Linear	Sms.ols	Sales ~ t	260.93

Exponential	Sms.ols	Log_Sales ~ t	268.69

Quadratic	Sms.ols	Sales ~ t + t_square	297.40

Additive Seasonality	Sms.ols	Sales ~ months (i.e seasons)	235.60

Additive Seasonality with linear trend	Sms.ols	Sales ~ t + months	135.55

Additive Seasonality with Quadratic trend	Sms.ols	Sales ~ t + t_square + months	218.19

Multiplicative Seasonality	Sms.ols	log_Sales ~ months	239.65

Multiplicative Seasonality with Linear Trend	Sms.ols	log_Sales ~ t + months	160.68

Multiplicative Seasonality with Quadratic Trend	Sms.ols	log_Sales ~ t + t_square + months	239.60

t.	We notice that the model with Additive seasonality having a linear trend gives the least RMSE value i.e, least error. If we look back, we had seen in the plot and adviced that the time series plot and the seasonal decomposition plot both showed that the linear trend and the additive seasonality was our assumption. It has been now proved right with the calculations giving the least RMSE values.

u.	Hence we infer from the calculation and data that the model with Additive seasonality and linear trend can be used to predict the values which will be having least error/residuals.
