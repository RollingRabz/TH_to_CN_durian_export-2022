# Project Name: Compare accuracy of predict model of the export quantity of Durian from Thailand to China
## Project Objectives:
1. To develop and implement prediction model of the export quantity of Durian from 
Thailand to China using Multiple Linear regression model and Long Short-Term Memory 
model.
2. Compare the accuracy of predict model between Multiple Linear regression model 
and Long Short-Term Memory model.
## Project Scopes:
1. Data usage are durian export quantity provided by Ministry of Commerce, CPI of 
China provided by Federal Reserve Economic Data, Temperature and Rain data 
provided by World Bank Group, Durian yield and fertilizer price data provided by 
Office of Agricultural Economics and CNY to THB exchange rate data provided by 
investing.com. The range of monthly time-series secondary data are from January 
2008 to December 2021.
2. Compare the accuracy of the predict model by using Root Mean Square Error 
(RMSE)
3. Perform Time Series cross-validation and Rolling Time Series cross-validation in 
training and testing dataset selecting process for each model.
4. In the process of creating the predict model using LSTM model we observe only
50 - 200 epochs, 1-72 batch sizes and 1-5 dense with using Adam method as an 
optimizer.
## Project Details:
This project uses various data to be applied in a model to forecast the export quantity 
of Thailand's durian to China. by selecting variables, the correlation of each variable was 
tested and then used in the model to compare the forecast performance as shown in the 
picture
![Work_Flow](/Pic/workflow.PNG)

  As shown in the picture, there are 9 steps in total, but there will be an iteration in 
steps 1 and 2 due to the increase or decrease of the factors. To be suitable for the purpose 
of this project the first step is to study the factors affecting the export of durian from 
Thailand to China. Then select a dataset that is expected to be appropriate for the studied 
scope and then align the data to make it ready for use in the model. Then study and 
select a model for forecasting. and testing the relationship of each variable Once the 
tested variables are obtained, they are applied to the model. and fine-tuning the model 
parameters accordingly. Then compare the forecasting performance of each model to 
draw further conclusions.

  In this study the selected data was divided into 2 parts. The inputs data, the variables 
were selected, i.e., output quantity, export volume, rainfall, average temperature. and past 
fertilizer prices and China's purchasing power data the variable was selected. Consumer 
Price Index and exchange rate to be used to predict the export volume of durian using 
the past 12 months to forecast the next month's exports. The models used are Multiple 
linear regression and Long Short-Term Memory from packages in python such as sci-kit 
learn, matplotlib, keras, pandas, and numpy. The expected values of this project are 
forecast accuracy from the Multiple linear regression models and Long Short-Term Memory 
models.

##  Project Results
### Correlation  Metrix of all variables
![corr](/Pic/corrheatmap.png)
### Pearson correlation coefficient test
![corr2](/Pic/corrtest.png)
### Compare predict and actual value with multiple linear regression model
![ML](/Pic/Multipredict.png)
![ERR](/Pic/MultiError.png)
### Compare predict and actual value with LSTM (Long Short-Term Memory) model at 100 epochs, 1 batch sizes and 1 dense with using Adam method as an optimizer.
![LSTM](/Pic/LSTM3.png)
![Err2](/Pic/LSTM3_err.png)

The objective of this project is to compare the accuracy of the prediction model 
between Multi linear regression and LSTM. Firstly, after we are gathering all the data, we need 
we plot the correlation to see the variable relationship and we choose the 
variable by test the correlation coefficient and we see that Exchange rate of CNY – THB is not 
appropriate to put into the model because after testing it we found that its P-value is larger 
than 0.05. As a result, we do not put exchange rate into our model. After that we run the 
predict models which is Multi linear regression and LSTM with all the data except exchange 
rate. As a result, we can see that mean absolute percentage error when predict using multiple 
linear regression model is 1.635 which is higher than predict using LSTM model which is 0.812
and root mean square error when predict using multiple linear regression model is 45513.982
which is higher than predict using LSTM model which is 38036.017 so we can say that to 
predict the export quantity of durian from Thailand to China the LSTM model is provide better 
accuracy than multiple linear regression model.

## Conclusion and Future Work

In conclusion, the forecasting of durian export volume from Thailand to China using 
the multiple linear regression model had the predictive efficiency measured by MAPE and 
RMSE of 1.635 and 45513.982, respectively. The forecasting using the long-short term memory 
model was effective in predicting the export volume of durian from Thailand to China. 
Forecasts measured by MAPE and RMSE were 0.812 and 38036.017, respectively. From MAPE
values it can be concluded that the LSTM model provides better performance than the 
multiple linear regression model in forecasting and from RMSE value it can be concluded that 
the LSTM model provides better performance than the multiple linear regression model. And 
the R-square values, LSTM model gives the source variable a better description of the 
dependent variable. Therefore, it is clear that Long Short-Term Memory model has better 
performance in applying the forecasting of durian export volume from Thailand to China.

In this project there was a problem with data values and variables that are used the 
data values are quite different. Which, when used in forecasting, results in very poor results. 
The range of the data may need to be adjusted to reduce the variance for increase the 
forecasting efficiency of each model. It may also be necessary to study more about the factors 
affecting the export volume of durian to increase the number of initial variables to be able 
for describe the dependent variables better and fine-tuning the parameters in the model to 
increase efficiency Including the study of using other models to bring forecasts. It could be a 
good alternative to optimizing forecasts to be able to bring forecast results including the 
model used for those involved to be able to use it for further benefits.


