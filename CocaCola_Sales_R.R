library(readxl)
cocacola <- read_excel(file.choose()) # read the cocacola data
View(cocacola) # Seasonality 4 quarters

# Pre Processing
# input t
cocacola["t"] <- c(1:42)
View(cocacola)

cocacola["t_square"] <- cocacola["t"] * cocacola["t"]
cocacola["log_Sales"] <- log(cocacola["Sales"])

# So creating 4 dummy variables for quarters
X <- data.frame(rep(1:4,length = 42))

install.packages("fastDummies")
library(fastDummies)
X <- dummy_cols(X, select_columns = ("rep.1.4..length...42."),
                         remove_first_dummy = FALSE, remove_most_frequent_dummy = FALSE, remove_selected_columns = TRUE)

X$Q1 <- X$rep.1.4..length...42._1
X$Q2 <- X$rep.1.4..length...42._2
X$Q3 <- X$rep.1.4..length...42._3
X$Q4 <- X$rep.1.4..length...42._4

X <- X[ , -c(1, 2, 3, 4)]
attach(X)
View(X)

cocacolaSales <- cbind(cocacola, X)
colnames(cocacolaSales)

View(cocacolaSales)
## Pre-processing completed

attach(cocacolaSales)

# partitioning
train <- cocacolaSales[1:38, ]
test <- cocacolaSales[39:42, ]

########################### LINEAR MODEL #############################

linear_model <- lm(Sales ~ t, data = train)
summary(linear_model)

linear_pred <- data.frame(predict(linear_model, interval = 'predict', newdata = test))
rmse_linear <- sqrt(mean((test$Sales - linear_pred$fit)^2, na.rm = T))
rmse_linear

######################### Exponential ############################

expo_model <- lm(log_Sales ~ t, data = train)
summary(expo_model)
expo_pred <- data.frame(predict(expo_model, interval = 'predict', newdata = test))
rmse_expo <- sqrt(mean((test$Sales - exp(expo_pred$fit))^2, na.rm = T))
rmse_expo

######################### Quadratic ###############################

Quad_model <- lm(Sales ~ t + t_square, data = train)
summary(Quad_model)
Quad_pred <- data.frame(predict(Quad_model, interval = 'predict', newdata = test))
rmse_Quad <- sqrt(mean((test$Sales-Quad_pred$fit)^2, na.rm = T))
rmse_Quad

######################### Additive Seasonality #########################

sea_add_model <- lm(Sales ~ Q1 + Q2 + Q3, data = train)
summary(sea_add_model)
sea_add_pred <- data.frame(predict(sea_add_model, newdata = test, interval = 'predict'))
rmse_sea_add <- sqrt(mean((test$Sales - sea_add_pred$fit)^2, na.rm = T))
rmse_sea_add


######################## Multiplicative Seasonality #########################

multi_sea_model <- lm(log_Sales ~ Q1 + Q2 + Q3, data = train)
summary(multi_sea_model)
multi_sea_pred <- data.frame(predict(multi_sea_model, newdata = test, interval = 'predict'))
rmse_multi_sea <- sqrt(mean((test$Sales - exp(multi_sea_pred$fit))^2, na.rm = T))
rmse_multi_sea

################### Additive Seasonality with Quadratic Trend #################

Add_sea_Quad_model <- lm(Sales ~ t + t_square + Q1 + Q2 + Q3, data = train)
summary(Add_sea_Quad_model)
Add_sea_Quad_pred <- data.frame(predict(Add_sea_Quad_model, interval = 'predict', newdata = test))
rmse_Add_sea_Quad <- sqrt(mean((test$Sales - Add_sea_Quad_pred$fit)^2, na.rm=T))
rmse_Add_sea_Quad

# Preparing table on model and it's RMSE values 
table_rmse <- data.frame(c("rmse_linear", "rmse_expo", "rmse_Quad", "rmse_sea_add", "rmse_Add_sea_Quad", "rmse_multi_sea"), c(rmse_linear, rmse_expo, rmse_Quad, rmse_sea_add, rmse_Add_sea_Quad, rmse_multi_sea))
colnames(table_rmse) <- c("model", "RMSE")
View(table_rmse)

# Additive seasonality with Quadratic Trend has least RMSE value

write.csv(cocacolaSales, file = "cocacolaSales.csv", row.names = F)
getwd()

############### Combining Training & test data to build Additive seasonality using Quadratic Trend ############

Add_sea_Quad_model_final <- lm(Sales ~ t + t_square + Q1 + Q2 + Q3, data = cocacolaSales)
summary(Add_sea_Quad_model_final)

#Lets get the Residuals
resid <- residuals(Add_sea_Quad_model_final)
resid[1:10]

windows()
hist(resid)

windows()
acf(resid,lag.max = 10)

k <- arima(resid, order=c(1,0,0))

windows()
acf(k$residuals,lag.max = 15)
pred_res<- predict(arima(resid,order=c(2,0,0)), n.ahead = 12)
str(pred_res)
pred_res$pred
acf(k$residuals)

#Lets Build a ARIMA model for whole dataset. ARIMA(Auto Regression Integrated Moving Average)

#Lets convert the data into Time Series data

library(tseries)
library(forecast)

cocacola_ts <- ts(cocacola$Sales, frequency = 4, start = c(1986)) #Create a Time Series data
View(cocacola_ts)
plot(cocacola_ts) #Plots the data into a Line chart By default as the data is a Time Series Data

#For Building ARIMA model we the AR coefficient i.e p-value then Integration Coefficient i.e d and Moving Average coefficient i.e q-value
#Lets find p-value, p-value is Obtained by pacf
pacf(cocacola_ts) #Lets Consider it as 1

#Lets Find the q-value by acf
acf(cocacola_ts) #Lets Consider this as 0.5

#Also lets Consider the d-value as 0.5
#now lets build an ARIMA model
a <- arima(cocacola_ts, order = c(1,0.5,0.5), method = "ML")
a
#Lets plot the forecast using the ARIMA model
plot(forecast(a, h=4), xaxt = "n")

#Seeing the plot, we get to know that the forecast done was not accurate and is giving level and a meagre trend

#We can build the model using auto.arima() function.
#This function will analyse the p and q value and build a proper model. Lets Build the Model

ab <- auto.arima(cocacola_ts)

windows()
plot(forecast(ab, h=12), xaxt = "n")
#So now we can see that the forecast was accurate 

prediction <- forecast(ab, h=8) #This will predict for the next 8 Quarters, i.e 2 years

prediction

###########################################################################
###########Applying Exponential smoothing model############################
library(forecast)
library(fpp)
library(smooth) # for smoothing and MAPE
library(tseries)

library(readxl)
CocaCola_Sales_Rawdata <- read_excel("C:/Data Science/Data Science/Book and study material/27. Forecasting/CocaCola_Sales_Rawdata.xlsx")
View(CocaCola_Sales_Rawdata)

# Converting data into time series object
tssales <- ts(CocaCola_Sales_Rawdata$Sales, frequency = 4, start = c(42))
View(tssales)

# dividing entire data into training and testing data 
train <- tssales[1:38]
test <- tssales[39:42]
# Considering only 4 Quarters of data for testing because data itself is Quarterly
# seasonal data

# converting time series object
train <- ts(train, frequency = 4)
test <- ts(test, frequency = 4)

# Plotting time series data
plot(tssales)
# Visualization shows that it has level, trend, seasonality => Additive seasonality

#### USING HoltWinters function ################
# Optimum values
# with alpha = 0.2 which is default value
# Assuming time series data has only level parameter
hw_a <- HoltWinters(train, alpha = 0.2, beta = F, gamma = F)
hw_a
hwa_pred <- data.frame(predict(hw_a, n.ahead = 4))
hwa_pred

# By looking at the plot the forecasted values are not showing any characters of train data 
plot(forecast(hw_a, h = 4))
hwa_mape <- MAPE(hwa_pred$fit, test)*100

# with alpha = 0.2, beta = 0.15
# Assuming time series data has level and trend parameter 
hw_ab <- HoltWinters(train, alpha = 0.2, beta = 0.15, gamma = F)
hw_ab
hwab_pred <- data.frame(predict(hw_ab, n.ahead = 4))
# by looking at the plot the forecasted values are still missing some characters exhibited by train data
plot(forecast(hw_ab, h = 4))
hwab_mape <- MAPE(hwab_pred$fit,test)*100

# with alpha = 0.2, beta = 0.15, gamma = 0.05 
# Assuming time series data has level,trend and seasonality 
hw_abg <- HoltWinters(train, alpha = 0.2, beta = 0.15, gamma = 0.05)
hw_abg
hwabg_pred <- data.frame(predict(hw_abg, n.ahead = 4))
# by looking at the plot the characters of forecasted values are closely following historical data
plot(forecast(hw_abg, h = 4))
hwabg_mape <- MAPE(hwabg_pred$fit, test)*100


# With out optimum values 
hw_na <- HoltWinters(train, beta = F, gamma = F)
hw_na
hwna_pred <- data.frame(predict(hw_na, n.ahead = 4))
hwna_pred
plot(forecast(hw_na,h=4))
hwna_mape <- MAPE(hwna_pred$fit,test)*100


hw_nab <- HoltWinters(train, gamma = F)
hw_nab
hwnab_pred <- data.frame(predict(hw_nab, n.ahead = 4))
hwnab_pred
plot(forecast(hw_nab, h = 4))
hwnab_mape <- MAPE(hwnab_pred$fit, test)*100

hw_nabg <- HoltWinters(train)
hw_nabg
hwnabg_pred <- data.frame(predict(hw_nabg, n.ahead = 4))
hwnabg_pred
plot(forecast(hw_nabg, h = 4))
hwnabg_mape <- MAPE(hwnabg_pred$fit, test)*100


df_mape <- data.frame(c("hwa_mape","hwab_mape","hwabg_mape","hwna_mape","hwnab_mape","hwnabg_mape"),c(hwa_mape,hwab_mape,hwabg_mape,hwna_mape,hwnab_mape,hwnabg_mape))

colnames(df_mape)<-c("MAPE","VALUES")
View(df_mape)

# Based on the MAPE value who choose holts winter exponential tecnique which assumes the time series
# Data level, trend, seasonality characters with default values of alpha, beta and gamma

new_model <- HoltWinters(tssales)
new_model

plot(forecast(new_model, n.ahead = 4))

# Forecast values for the next 4 quarters
forecast_new <- data.frame(predict(new_model, n.ahead = 8))

forecast_new
