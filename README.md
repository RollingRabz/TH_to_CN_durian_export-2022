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

