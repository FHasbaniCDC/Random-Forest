#import data https://archive.ics.uci.edu/ml/datasets/Dermatology
data <- read.csv(file.choose(), header=TRUE)

str(data)

table(data$class)
prop.table(table(train$class))

wr.tree1 <-rpart(class ~ ., data=data,
                 method="class",
                 control=rpart.control(minsplit=20, cp=0.001))

dev.new(width=8, height=10, unit="in")
plot(wr.tree1, uniform=T, main="Dermatology")
text(wr.tree1, use.n=T, all=T)

# Default plot of the result
dev.new(width=10, height=5, unit="in")
plot(wr.tree1, uniform=T, branch=0, compress=F, main="Dermatology")
text(wr.tree1, use.n=T, all=T, digits=3, cex=0.7, xpd=T, fancy=T, 
     fwidth=0.3, fheight=0.3, pretty=T)

print(wr.tree1)
rpart.plot(wr.tree1)
#resize to see tree
#can prune back 

#CHANGE AND SET TO A DIFFERENT SEED
set.seed(222)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
train <- data[ind==1,]
test <- data[ind==2,]

######FROM HERE CHANGE THE DATASETS TO TRAIN AND TEST########
#creates the decision tree
#class is the classifer.
wr.tree1 <-rpart(class~., data=train, method="class",  cp = -1)
#cp=-1 makes sure tree is fully grown
rpart.plot(wr.tree1, extra = 104)

#extra=104 class model with a response having more than two levels
#By default, rpart() function uses the Gini impurity measure to split the note. 
#The higher the Gini coefficient, the more different instances within the node

library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(wr.tree1, caption = NULL)

#using cp=-1 will most likely produce an overfitted tree. 
#But you can prune back


print(wr.tree1)
# From the rpart documentation, "An overall measure of variable importance is 
#the sum of the goodness of split measures for each split for which it was the 
#primary variable."
wr.tree1$variable.importance
printcp(wr.tree1)
#rel error of each iteration of the tree is the fraction of mislabeled elements in the
#iteration relative to the fraction of mislabeled elements in the root. 
#The cross validation error rates and standard deviations are displayed in the columns 
#xerror and xstd respectively.

#As a rule of thumb, it's best to prune a decision tree using the cp of smallest 
#tree that is within one standard deviation of the tree with the smallest xerror. 
#pruning tree. Cutting back branches . The branch number is found on the text of the tree
#wr.tree2<-snip.rpart(wr.tree1,c( ,))
#print(wr.tree2)

#0.58480 0.045298 = .62, want the smallest tree with xerror less than .62
#tree with cp = 0.2, so we want to prune our tree with a cp slightly greater than than 0.2.
wr.tree1 <- prune(wr.tree1, cp = 0.205)
fancyRpartPlot(wr.tree1)

#Predict the outcome and the possible outcome probabilities
test$DermClass <- predict(wr.tree1, newdata = test, type = "class")
test$DermProb <- predict(wr.tree1, newdata = test, type = "prob")
test

predict_unseen <-predict(wr.tree1, newdata = test, type = 'class')
table_mat <- table(test$class, predict_unseen)
table_mat

accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for test', accuracy_Test))

accuracy_tune <- function(wr.tree1) {
  predict_unseen <- predict(wr.tree1, test, type = 'class')
  table_mat <- table(test$class, predict_unseen)
  accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
  accuracy_Test
}

control <- rpart.control(minsplit = 4,
                         minbucket = round(5 / 3),
                         maxdepth = 5,
                         cp = 0)
tune_fit <- rpart(class~., data = train, method = 'class', control = control)
accuracy_tune(tune_fit)

#random Forest
library(dplyr)

#change class to factor level
data$class<- as.factor(data$class)
ind <- sample(2, nrow(data), replace=TRUE, prob=c(.7,.3))
train <- data[ind==1,]
test <- data[ind==2,]

glimpse(train)
glimpse(test)

#RandomForest(formula, ntree=n, mtry=FALSE, maxnodes = NULL)
#Arguments:
# - Formula: Formula of the fitted model
#- ntree: number of trees in the forest
#- mtry: Number of candidates draw to feed the algorithm. By default, it is the square of the number of columns.
#- maxnodes: Set the maximum amount of terminal nodes in the forest
#- importance=TRUE: Whether independent variables importance in the random forest be assessed

library(randomForest)
library(caret)
library(e1071)

# Define the control
trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid")

set.seed(1234)
str(data)
train$class <-as.factor(train$class)
# Run the model
rf_default <- train(class~.,
                    data = train,
                    method = "rf",
                    metric = "Accuracy", #accuracy for classification
                    trControl = trControl)
# Print the results
print(rf_default)

#The algorithm uses 500 trees and tested three different values of mtry: 2, 18, 34.

#The final value used for the model was mtry = 2 with an accuracy of 0.rsquared 93%, RMSE= 42%. 
#Let's try to get a higher score.

#Search best mtry
#You can test the model with values of mtry from 1 to 18
set.seed(1234)
tuneGrid <- expand.grid(.mtry = c(2: 18))
rf_mtry <- train(class~.,
                 data = train,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 25,
                 ntree = 300)
print(rf_mtry)

#The best value of mtry is stored in:
rf_mtry$bestTune$mtry

#store it and use it when you need to tune the other parameters.
max(rf_mtry$results$RMSE)

best_mtry <- rf_mtry$bestTune$mtry 
best_mtry

store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 25)) {
  set.seed(1234)
  rf_maxnode <- train(class~.,
                      data = train,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 25,
                      maxnodes = maxnodes,
                      ntree = 300)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)

#The lowest RMSE score is obtained with a value of maxnode equals to 20.

#you have the best value of mtry and maxnode, you can tune the number of trees.
store_maxtrees <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)) {
  set.seed(5678)
  rf_maxtrees <- train(class~.,
                       data = train,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 14,
                       maxnodes = 22,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)

#ntree = 550: 550 trees will be trained
#mtry=18: 4 features is chosen for each iteration
#maxnodes = 22: Maximum 22 nodes in the terminal nodes (leaves)

fit_rf <- train(class~.,
                train,
                method = "rf",
                metric = "Accuracy",
                tuneGrid = tuneGrid,
                trControl = trControl,
                importance = TRUE,
                nodesize = 14,
                ntree = 550,
                maxnodes = 22)

#predict(model, newdata= df)
#argument
#- `model`: Define the model evaluated before. 
#- `newdata`: Define the dataset to make prediction

prediction <-predict(fit_rf, test)

#Use the prediction to compute the confusion matrix and see the accuracy score

confusionMatrix(prediction, test$class)

#You have an accuracy of 0.9762 percent, which is higher than the default value
varImp(fit_rf)
varImpPlot(fit_rf) #may have to change to character not factor



