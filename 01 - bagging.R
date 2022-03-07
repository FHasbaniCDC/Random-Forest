library(dplyr)   
library(ggplot2) 
library(e1071)   #for calculating variable importancei
library(caret)   #for general model fitting
library(rpart)   #for fitting decision trees
library(ipred)   #for fitting bagged decision trees

#load data
data("airquality")
View(airquality)

#view structure of airquality dataset
str(airquality)

#for reproducibility
set.seed(1)

#fit the bagged model
bag <- bagging(
  formula = Ozone ~ .,
  data = airquality,
  nbagg = 150, #bootstrapped samples 
  coob = TRUE, #estimated out-of-bag error
  control = rpart.control(minsplit = 2, cp = 0))

#minsplit = 2: This tells the model to only require 2 observations in a node to split.
#cp = 0. This is the complexity parameter. By setting it to 0, we don't require the model to be able to improve the overall fit by any amount in order to perform a split.

#display fitted bagged model
bag

#17.4973=out-of-bag estimated RMSE: the average difference between the predicted value for Ozone and the actual observed value.

#bagged models tend to provide more accurate predictions compared to individual decision trees, it's difficult to interpret and visualize the results of fitted bagged models.

#calculate variable importance
VI <- data.frame(var=names(airquality[,-1]), imp=varImp(bag))

#sort variable importance descending
VI_plot <- VI[order(VI$Overall, decreasing=TRUE),]

#visualize variable importance with horizontal bar plot
barplot(VI_plot$Overall,
        names.arg=rownames(VI_plot),
        horiz=TRUE,
        col='steelblue',
        xlab='Variable Importance')

#Use the Model to Make Predictions
#define new observation for test data
new <- data.frame(Solar.R=150, Wind=8, Temp=70, Month=5, Day=5)

#use fitted bagged model to predict Ozone value of new observation
predict(bag, newdata=new)

#Based on the values of the predictor variables, the fitted bagged model predicts that the Ozone value will be 24.487 on this particular day.