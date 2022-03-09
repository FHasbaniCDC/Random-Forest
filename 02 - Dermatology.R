library(rpart)
library(rpart.plot)
library(ggplot2)

#import data https://archive.ics.uci.edu/ml/datasets/Dermatology
data <- read.csv(file.choose(), header=TRUE)

table(data$class)
prop.table(table(data$class))

#method 1
prop.table(table(data$class))

#method 2
data %>% 
  count(class) %>% 
  mutate(perc = n/sum(n)*100)

summary(data)

#dev.off()
data %>% 
  ggplot()+
  geom_boxplot(aes(x=class, y=age))

# similar distribution between the two classes for age

data %>% 
  ggplot()+
  geom_bar(aes(x=scalp,fill=class),
           position="fill") +
  facet_wrap(~itching)+
  labs(title = "Scalp % distribution by class",
       subtitle = "Analysis stratified by itching ")

#Classification Tree
#Note that some options that control the rpart algorithm are specified by means of the rpart.control function (see ?rpart.control)
wr.tree1 <-rpart(class ~ ., data=data,
                 method="class",
                 control=rpart.control(minsplit=20, cp=0.001))

#In particular, the cp option, set by default equal to 0.01, represent a pre-pruning step because it prevents the generation of non-significant branches (any split that does not decrease the overall lack of fit by a factor of cp is not attempted). See also the meaning of the minsplit, minbucket and maxdepth options.

# obtain the size of the tree by counting the number of nodes which are classified as leaf nodes
sum(wr.tree1$frame$var=="<leaf>")

dev.new(width=8, height=10, unit="in")
plot(wr.tree1, uniform=T, main="Dermatology")
text(wr.tree1, use.n=T, all=T)

# Default plot of the result
dev.new(width=10, height=5, unit="in")
plot(wr.tree1, uniform=T, branch=0, compress=F, main="Dermatology")
text(wr.tree1, use.n=T, all=T, digits=3, cex=0.7, xpd=T, fancy=T, 
     fwidth=0.3, fheight=0.3, pretty=T)

print(wr.tree1)
# Visualize the decision tree with rpart.plot
rpart.plot(wr.tree1, box.palette="RdBu", shadow.col="gray", nn=TRUE)

rpart.plot(x = wr.tree1, type=0, extra=3)
rpart.plot(x = wr.tree1, type=0, extra=0) #super simplified version
#see help for options

#resize to see tree

#can prune back 

#CHANGE AND SET TO A DIFFERENT SEED
set.seed(222)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
train <- data[ind==1,]
test <- data[ind==2,]

#FROM HERE CHANGE THE DATASETS TO TRAIN AND TEST#
#creates the decision tree
#class is the classifier.
wr.tree1 <-rpart(class~., data=train, method="class",  cp = .01)
#cp=-1 makes sure tree is fully grown
rpart.plot(wr.tree1, extra = 104) #rpart.rules

#extra=104 class model with a response having more than two levels
#By default, rpart() function uses the Gini impurity measure to split the note. 
#The higher the Gini coefficient, the more different instances within the node

#prune back tree

print(wr.tree1)

# From the rpart documentation, "An overall measure of variable importance is 
#the sum of the goodness of split measures for each split for which it was the 
#primary variable."
wr.tree1$variable.importance

#rel error of each iteration of the tree is the fraction of mislabeled elements in the
#iteration relative to the fraction of mislabeled elements in the root. 

wr.tree1$cptable
cptable = data.frame(wr.tree1$cptable)
#select the tree with the lowest cross-validation error and to find the corresponding value of CP and number of splits:
min(cptable$xerror)

which.min(cptable$xerror)
#we have that the best tree has 5 splits before the post-pruning. 

cptable$nsplit[which.min(cptable$xerror)]

#The cross validation error rates and standard deviations are displayed in the columns 
#xerror and xstd respectively.

oneSElimit = min(cptable$xerror) + cptable$xstd[which.min(cptable$xerror)]
oneSElimit

#check which is the smallest tree whose 
#As a rule of thumb, it's best to prune a decision tree using the cp of smallest 
#tree that is within one standard deviation of the tree with the smallest xerror. xerror value is lower than the oneSElimit value:
  
# all the trees with an error below the limit
which(cptable$xerror<oneSElimit)

# take the smallest one
best = min(which(cptable$xerror<oneSElimit))
best
## [1] 5
bestcp = cptable$CP[best]
bestcp 
## [1] 0.01
#With this approach the best pruned has 2 splits and the corresponding value of CP is equal to 

#We can now prune the tree with the selected value of ??, (CP) by using the prune function:
wr.tree1_p = prune(tree = wr.tree1, cp = bestcp)

#plot the pruned tree:
rpart.plot(x = wr.tree1_p) 

#pruning tree. Cutting back branches . The branch number is found on the text of the tree
#wr.tree2<-snip.rpart(wr.tree1,toss=)
#print(wr.tree2)
#wr.tree12 <- prune(wr.tree1, cp = )

#Predict the outcome and the possible outcome probabilities
test$treeClass <- predict(wr.tree1_p, newdata = test, type = "class")
test$treeProb <- predict(wr.tree1_p, newdata = test, type = "prob")
test

predict_unseen <-predict(wr.tree1_p, newdata = test, type = 'class')
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

class_tree_pred_prob = predict (wr.tree1_p,                      newdata=test,
                     type = "prob")

#confirmation
control <- rpart.control(maxdepth = 5,
                         cp = 0.01)
tune_fit <- rpart(class~., data = test, method = 'class', control = control)
accuracy_tune(tune_fit)

#random Forest
library(dplyr)

#change class to factor level
data$class<- as.factor(data$class)
ind <- sample(2, nrow(data), replace=TRUE, prob=c(.7,.3))
train <- data[ind==1,]
test <- data[ind==2,]


#RandomForest(formula, ntree=n, mtry=FALSE, maxnodes = NULL)
#Arguments:
# - Formula: Formula of the fitted model
#- ntree: number of trees in the forest
#- mtry: Number of candidates draw to feed the algorithm. By default, it is the square of the number of columns.
#- maxnodes: Set the maximum amount of terminal nodes in the forest
#- importance=TRUE: Whether independent variables importance in the random forest be assessed
#Bagging corresponds to a particular case of random forest when all the regressors are used. For this reason to implement bagging we use the function randomForest in the randomForest package. As the procedure is based on boostrap resampling we need to set the seed. As usual we specify the formula and the data together with the number of regressors to be used (mtry, in this case equal to the number of columns minus the column dedicated to the response variable). The option importance is set to T when we are interested in assessing the importance of the predictors. Finally, ntree represents the number of independent trees to grow (500 is the default value).

#Bagging: all regressors are used. For this reason to implement bagging we use the function randomForest in the randomForest package. difference is in mtry, in RF=mtry = sqrt(ncol(train)-34)

set.seed(1, sample.kind="Rejection")
class_bag = randomForest(formula = class~ .,    data = train,
   mtry = ncol(train)-34, #minus the response
   importance = T, #to estimate predictors importance
    ntree = 500) #500 by default
class_bag

varImpPlot(class_bag)
#left plot the mean decrease (across all B trees) of accuracy in prediction on the OOB samples when a given variable is excluded. This measures how much accuracy the model losses by excluding each variable.

#right plot the mean decrease (across all B trees) in node purity on the OOB samples when a given variable is excluded. For classification, the node impurity is measured by the Gini index. This measures how much each variable contributes to the homogeneity of the nodes in the resulting random forest.

class_bag_pred$BagClass = predict(class_bag,
              newdata=test,
              type="class")
head(class_bag_pred)

accbag = mean((test$class == class_bag_pred$BagClass)) 
accbag
accuracy_tune #pruned accuracy on test
#bagging performs slightly beter with respect to the single prune tree.

#compute the predicted probabilities:
class_bag_pred_prob = predict(class_bag,
                    newdata=test,
                    type = "prob")

test$bagClass <- predict(class_bag, newdata = test, type = "class")
test$bagProb <- predict(class_bag, newdata = test, type = "prob")
test

#random forest using caret

library(randomForest)
library(caret)
library(e1071)

# Define the control
trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid")

set.seed(1)
str(data)
train$class <-as.factor(train$class)

# Run the model
rf_class <- train(class~.,
                    data = train,
                    importance=T,
                    method = "rf",
                    metric = "Accuracy",
                  #accuracy for classification
                    trControl = trControl
)
                  
# Print the results
print(rf_class)

rf_class$control #prints out the indices

#Accuracy = 98.0%
rf_pred=predict(rf_class,test)

#adding predicted class to test set
rfProbs=predict(rf_class,test, type="prob")

#install.packages("matrixStats")
library(matrixStats)
num = as.numeric(rf_pred)
test$rfProb = rowMaxs(as.matrix(rfProbs))

#Use the prediction to compute the confusion matrix and see the accuracy score

confusionMatrix(rf_pred, test$class)
#The algorithm uses 500 trees and tested three different values of mtry: 2, 18, 34.

#Compute predicted probabilities
class_rf_pred_prob = predict(object = rf_class,
 newdata = test,
  type = "prob")

#ROC
#Let's try to get a higher score.


#Search best mtry
#You can test the model with values of mtry from 1 to 18
set.seed(1)
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
max(rf_mtry$results$Accuracy)

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



