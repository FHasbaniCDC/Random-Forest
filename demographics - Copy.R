#Decision tree
#Decision Trees model regression problems by split data based on different values
#Response variable 'medv' which is the Median Housing Value from the Boston Housing dataset
library(MASS)
data(Boston)
str(Boston)

#create a basic Decision Tree regression model, using rpart
library(rpart)
model = rpart(medv ~ ., data = Boston)
model

#model a ridge regression using the Caret package
#pass the same parameters as above, but in addition we pass the method = 
#'rpart2' model to tell Caret to use a lasso model.
library(caret)
set.seed(1)

model <- train(
  medv ~ .,
  data = Boston,
  method = 'rpart2'
)

# caret automatically trained over multiple hyper parameters
model

#trained over multiple hyper parameters
plot(model)

#preprocess

set.seed(1)

inTraining <- createDataPartition(Boston$medv, p = .80, list = FALSE)
training <- Boston[inTraining,]
testing  <- Boston[-inTraining,]

set.seed(1)
model3 <- train(
  medv ~ .,
  data = training,
  method = 'rpart2',
  preProcess = c("center", "scale")
)
model3

#check our data on the test set.
#use the subset method to get the features and test target. 
#then use the predict method passing in our model from above and the test features.
#calculate the RMSE and r2 to compare to the model above.
#use the subset method to get the features and test target
test.features = subset(testing, select=-c(medv))
test.target = subset(testing, select=medv)[,1]

predictions = predict(model3, newdata = test.features)

# RMSE
sqrt(mean((test.target - predictions)^2))

# R2
cor(test.target, predictions) ^ 2

#cross validation
#k-fold cross-validation that resamples and splits our data many times.
#then train the model on these samples and pick the best model. 
#Caret uses trainControl method to make this easy.

#10-fold cross-validation, pass method = "cv", number = 10 
set.seed(1)
ctrl <- trainControl(
  method = "cv",
  number = 10,
 )

#retrain our model and pass the ctrl response to the trControl parameter.
# set.seed(1)
model4 <- train(
  medv ~ .,
  data = training,
  method = 'rpart2',
  preProcess = c("center", "scale"),
  trControl = ctrl
)
model4
#The final value used for the model was maxdepth = 3

plot(model4)
#results seemed to have improved our accuracy for our training data

test.features = subset(testing, select=-c(medv))
test.target = subset(testing, select=medv)[,1]

predictions = predict(model4, newdata = test.features)

# RMSE
sqrt(mean((test.target - predictions)^2))

# R2
cor(test.target, predictions) ^ 2

#tune hyperparameters
#ridge model, we can give the model different values of lambda.

set.seed(1)

tuneGrid <- expand.grid(
  degree = 1, 
  nprune = c(2, 11, 10)
)

model5 <- train(
  medv ~ .,
  data = training,
  method = 'earth',
  preProcess = c("center", "scale"),
  trControl = ctrl,
  tuneGrid = tuneGrid
)

model5
#plot the model to see how it performs over different tuning parameters.
plot(model5)

#Random Forest - tree model that uses the bagging technique.
#Many trees are built up in parallel and used to build a single tree model

library(randomForest)

#rfNews() to see new features/changes/bug fixes.

#turn target to factor
str(Boston)
#medv is numeric - regression tree, RMSE
model = randomForest(medv ~ ., data = Boston)
model

#Modeling Random Forest in R with Caret
#how to model a ridge regression using the Caret package
#use the train method. We pass the same parameters as above, but in addition we pass the method = 'rf' model to tell Caret to use a lasso model.

library(caret)

set.seed(1)
model <- train(
  medv ~ .,
  data = Boston,
  method = 'rf'
)
model
#caret automatically trained over multiple hyper parameters. 
#plot those to visualize.
plot(model)

#Preprocessing with Caret
#center and scale our data
preProcess = c("center", "scale")
set.seed(1)

model2 <- train(
  medv ~ .,
  data = Boston,
  method = 'rf',
  preProcess = c("center", "scale") #center and scale
)
model2

set.seed(1)
#split and can check for overfitting
#return indexes that contains 80% of the data that we should use for training
inTraining <- createDataPartition(Boston$medv, p = .80, list = FALSE)
training <- Boston[inTraining,]
testing  <- Boston[-inTraining,]

#fit our model again using only the training data
set.seed(1)
model3 <- train(
  medv ~ .,
  data = training,
  method = 'rf',
  preProcess = c("center", "scale")
)
model3

#check our data on the test set

test.features = subset(testing, select=-c(medv))
test.target = subset(testing, select=medv)[,1]

predictions = predict(model3, newdata = test.features)

#calculate the RMSE and r2 to compare to the model above.
# RMSE
sqrt(mean((test.target - predictions)^2))

cor(test.target, predictions) ^ 2

#10-fold cross-validation
#common to use a data partitioning strategy like k-fold cross-validation that 
#resamples and splits our data many times.
#train the model on these samples and pick the best model
#easy with the trainControl method
set.seed(1)
ctrl <- trainControl(
  method = "cv",
  number = 10, #for 10 fold
)
#retrain the model on these samples and pick the best model
set.seed(1)
model4 <- train(
  medv ~ .,
  data = training,
  method = 'rf',
  preProcess = c("center", "scale"),
  trControl = ctrl
)
model4

plot(model4)

test.features = subset(testing, select=-c(medv))
test.target = subset(testing, select=medv)[,1]

predictions = predict(model4, newdata = test.features)

# RMSE
sqrt(mean((test.target - predictions)^2))

#improved our accuracy for our training data
# R2
cor(test.target, predictions) ^ 2

#Tuning Hyper Parameters
#o tune a random forest model, we can give the model different values of mtry. 
#Caret will retrain the model using different mtry and select the best version.
set.seed(1)
tuneGrid <- expand.grid(
  mtry = c(2:4)
)

model5 <- train(
  medv ~ .,
  data = training,
  method = 'rf',
  preProcess = c("center", "scale"),
  trControl = ctrl,
  tuneGrid = tuneGrid
)
model5

plot(model5)


#second random forest run

#training Sample with 300 observations
train=sample(1:nrow(Boston),300)

Boston.rf=randomForest(medv ~ . , data = Boston , subset = train)
Boston.rf

#Mean Squared Error and Variance explained are calculated using Out of Bag Error Estimation.

#Plotting the Error vs Number of Trees Graph.
plot(Boston.rf)

# Error and the Number of Trees.We can easily notice that how the Error is 
#dropping as we keep on adding more and more trees and average them.

#Random Forest model chose Randomly 4 variables to be considered at each split.
#now try all possible 13 predictors which can be found at each split.
#compare the Out of Bag Sample Errors and Error on Test set

oob.err=double(13)
test.err=double(13)
#mtry is no of Variables randomly chosen at each split

for(mtry in 1:13) 
{
  rf=randomForest(medv ~ . , data = Boston , subset = train,mtry=mtry,ntree=400) 
  oob.err[mtry] = rf$mse[400] #Error of all Trees fitted
  
  pred<-predict(rf,Boston[-train,]) #Predictions on Test Set for each Tree
  test.err[mtry]= with(Boston[-train,], mean( (medv - pred)^2)) #Mean Squared Test Error
  
  cat(mtry," ") #printing the output to the console
  
}

#Test Error
test.err

#Out of Bag Error Estimation
oob.err

#Plotting both Test Error and Out of Bag Error
matplot(1:mtry , cbind(oob.err,test.err), pch=19 , col=c("red","blue"),type="b",ylab="Mean Squared Error",xlab="Number of Predictors Considered at each Split")
legend("topright",legend=c("Out of Bag Error","Test Error"),pch=19, col=c("red","blue"))

#four is the number of predictor variables that produced the smallest mean squared error
#for the fitted trees (Out of Bag error).

#the Red line is the Out of Bag Error Estimates and the Blue Line is the Error calculated on Test Set
#Error Tends to be minimized at around mtry=4 .


