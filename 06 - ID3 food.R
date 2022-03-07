#ID3 example

#Problem Statement: To study a Mushroom data set in order to predict whether a given mushroom is edible or poisonous to human beings.

#Data Set Description: The given data set contains a total of 8124 observations of different kind of mushrooms and their properties such as odor, habitat, population, etc. 

#Logic: To build a Decision Tree model in order to classify mushroom samples as either poisonous or edible by studying their properties such as odor, root, habitat, etc.

#Loading libraries
#library(rpart,quietly = TRUE)
#no message confirming package attaching is printed, and most often, 
#no errors/warnings are printed if package attaching fails.

library(caret,quietly = TRUE)
library(rpart.plot,quietly = TRUE)
#library(rattle) #not available for current verson

#Reading the data set as a dataframe, foodID3.csv
mushrooms <- read.csv(file.choose(), header=TRUE, stringsAsFactors = TRUE)
# structure of the data
str(mushrooms)

# number of rows with missing values
nrow(mushrooms) - sum(complete.cases(mushrooms)) #formula

# analyzing the odor variable
table(mushrooms$class,mushrooms$odor)

number.perfect.splits <- apply(X=mushrooms[-1], MARGIN = 2, FUN = function(col){
  t <- table(mushrooms$class,col)
  sum(t == 0)
})

# Descending order of perfect splits
order <- order(number.perfect.splits,decreasing = TRUE)
number.perfect.splits <- number.perfect.splits[order]

# Plot graph
par(mar=c(10,2,2,2))
barplot(number.perfect.splits,
        main="Number of perfect splits vs feature",
        xlab="",ylab="Feature",las=2,col="wheat")

#data splicing
set.seed(12345)
train <- sample(1:nrow(mushrooms),size = ceiling(0.80*nrow(mushrooms)),replace = FALSE)
# training set
mushrooms_train <- mushrooms[train,]
# test set
mushrooms_test <- mushrooms[-train,]

# penalty matrix
penalty.matrix <- matrix(c(0,1,10,0), byrow=TRUE, nrow=2)

#Building a model

#build a Decision Tree by using the rpart  (Recursive Partitioning And Regression Trees) algorithm:
# building the classification tree with rpart
tree <- rpart(class~.,
              data=mushrooms_train,
              parms = list(loss = penalty.matrix),
              method = "class")

#library(rpart.plot)
# Visualize the decision tree with rpart.plot
rpart.plot(tree, nn=TRUE)

#test our Decision Tree model, we'll be applying the testing data set on our model like so:
#Testing the model
pred <- predict(object=tree,mushrooms_test[-1],type="class")

#Calculating accuracy

t <- table(mushrooms_test$class,pred) 
confusionMatrix(t)
#Confusion Matrix and Statistics


#output shows that all the samples in the test dataset have been correctly classified and we've attained an accuracy of 100% on the test data set with a 95% confidence interval (0.9977, 1). 
#can correctly classify a mushroom as either poisonous or edible using this Decision Tree model
