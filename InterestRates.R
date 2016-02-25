## Predicting Loan Interest Rates for Customers
## By Sammy Yu on 1/31/2016

library(magrittr)
library(mice)
library(VIM)
library(dummies)
library(glmnet)

## READ AND PROCESS THE TRAIN DATA
myData <- read.csv("Data for Cleaning & Modeling.csv", header = T, na.strings = c("NA", ""))
## convert the percentage columns into numeric
for (i in c(1, 30)) {
  myData[,i] <- gsub("%","", myData[,i]) %>% as.numeric
}
## convert the currency columns into numeric
for (i in 4:6) {
  myData[,i] <- gsub("$","", myData[,i], fixed = TRUE) %>% gsub(pattern=",",replacement="") %>% as.numeric
}
## convert some categorical variables into character
for (i in c(10, 15, 16, 18, 19, 23)) {
  myData[,i] <- as.character(myData[,i])
}
## impute missing values using the MICE package (this process will take very long time)
## since imputing all columns at one time requires a lot of memory, here I impute 2~3 columns at a time
tempData01 <- mice(myData[,1:2],  seed=245401) ## X1, X2
tempData02 <- mice(myData[,3:4],  seed=245402) ## X3, X4
tempData03 <- mice(myData[,5:6],  seed=245403) ## X5, X6
tempData04 <- mice(myData[,7:8],  seed=245404) ## X7, X8
tempData05 <- mice(myData[,9:10], seed=245405) ## X9, X10
tempData06 <- mice(myData[,11:12],seed=245406) ## X11, X12
tempData07 <- mice(myData[,13:14],seed=245407) ## X13, X14
tempData08 <- mice(myData[,15:17],seed=245408) ## X15, X16, X17
tempData09 <- mice(myData[,17:19],seed=245409) ## X17, X18, X19
tempData10 <- mice(myData[,20:21],seed=245410) ## X20, X21
tempData11 <- mice(myData[,22:23],seed=245411) ## X22, X23
tempData12 <- mice(myData[,24:25],seed=245412) ## X24, X25
tempData13 <- mice(myData[,26:27],seed=245413) ## X26, X27
tempData14 <- mice(myData[,28:29],seed=245414) ## X28, X29
tempData15 <- mice(myData[,30:32],seed=245415) ## X30, X31, X32
## get the imputed data in dataframe format
completedData01 <- complete(tempData01, 1)
completedData02 <- complete(tempData02, 1)
completedData03 <- complete(tempData03, 1)
completedData04 <- complete(tempData04, 1)
completedData05 <- complete(tempData05, 1)
completedData06 <- complete(tempData06, 1)
completedData07 <- complete(tempData07, 1)
completedData08 <- complete(tempData08, 1)
completedData09 <- complete(tempData09, 1)
completedData10 <- complete(tempData10, 1)
completedData11 <- complete(tempData11, 1)
completedData12 <- complete(tempData12, 1)
completedData13 <- complete(tempData13, 1)
completedData14 <- complete(tempData14, 1)
completedData15 <- complete(tempData15, 1)
## column-bind all small imputed dataframes into one
myData.imp <- cbind(completedData01, completedData02, completedData03, completedData04, completedData05, 
                    completedData06, completedData07, completedData08, completedData09[,2:3], completedData10, 
                    completedData11, completedData12, completedData13, completedData14, completedData15)
### Ridge regression and LASSO requires all variables are either numeric or dummy variables, so the character and date variables are removed from the data
## column X25 and X26 are removed as well because the percentage of missing values in these two columns are so high (> 50%) that would affect the predictability of the models
x.temp <- myData.imp[,c(-1,-10,-15,-16,-18,-19,-23,-25,-26)]
## create dummy variables for the factor variables
x.temp <- dummy.data.frame(x.temp, names = c("X7","X8","X9","X11","X12","X14","X17","X20","X32"))
## create a numeric matrix for the model training
x <- data.matrix(x.temp)
## X1 (interest rate) is the dependent variable
y <- myData.imp$X1
## take 70% of the imputed data for training, 30% for validation
set.seed(1999)
train <- sample (1: nrow(x), nrow(x) * 0.7)
validate <- (-train)


## READ AND PROCESS THE TEST DATA
myTestData <- read.csv("Holdout for Testing.csv", header = T, na.strings = c("NA", ""))
## remove the % symbol and convert to numeric
myTestData[,30] <- gsub("%","", myTestData[,30]) %>% as.numeric
## remove the $ symbol and comma, then convert to numeric
for (i in 4:6) {
  myTestData[,i] <- gsub("$","", myTestData[,i], fixed = TRUE) %>% gsub(pattern=",",replacement="") %>% as.numeric
}
## some factor varibles in the test data are missing some factor levels those exist in the train data
## so we need to add those back for the ones in the test data before running the test
for (i in c(7,8,9,11,12,14,17,20,32)) {
  levels(myTestData[,i]) <- levels(myData[,i])
}
## create the dummy variables for the factors in the test dataframe
myTestData <- dummy.data.frame(myTestData, names = c("X7","X8","X9","X11","X12","X14","X17","X20","X32"), drop=FALSE)
## remove the character and date columns, and others
myTestData$X1 <- NULL
myTestData$X10 <- NULL
myTestData$X15 <- NULL
myTestData$X16 <- NULL
myTestData$X18 <- NULL
myTestData$X19 <- NULL
myTestData$X23 <- NULL
myTestData$X25 <- NULL
myTestData$X26 <- NULL
x.test <- data.matrix(myTestData)


## create a list of the regularization parameter, lambda
grid = 10^seq(10,-2, length=100)


## MODEL 1: Ridge Regression
## train the Ridge regression model using the glmnet package
ridge.mod <- glmnet(x[train,], y[train], alpha=0, lambda=grid, thresh=1e-12)
set.seed(1)
## run cross-validation in 10 folds
ridge.cv = cv.glmnet(x[train,], y[train], alpha=0, nfolds=10)
## get the best lambda from the cross validation
ridge.bestlamda <- ridge.cv$lambda.min; ridge.bestlamda
## validate the model with the validation set
ridge.val  <- predict(ridge.mod, s=ridge.bestlamda, newx=x[validate,])
## RMSE is 2.800686 from Ridge regession
RMSE.ridge <- sqrt(mean((ridge.val - y[validate])^2)); RMSE.ridge
## retrain the model with the full data set
ridge.out <- glmnet(x, y, alpha=0)
## view the coefficients of the model
ridge.coef  <- predict(ridge.out, type = "coefficients", s=ridge.bestlamda); ridge.coef[1:15,]
## get the prediction results for the test data. 30 out of 80000 rows cannot be predicted
ridge.test  <- predict(ridge.out, s=ridge.bestlamda, type="response", newx=x.test); head(ridge.test, 10)


## MODEL 2: LASSO
## train the LASSO model using the glmnet package
lasso.mod <- glmnet(x[train,], y[train], alpha=1, lambda=grid)
set.seed(2)
## run cross-validation in 10 folds
lasso.cv = cv.glmnet(x[train,], y[train], alpha=1, nfolds=10)
## get the best lambda from the cross validation
lasso.bestlamda <- lasso.cv$lambda.min; lasso.bestlamda
## validate the model with the validation set
lasso.val  <- predict(lasso.mod, s=lasso.bestlamda, newx=x[validate,])
## RMSE is 2.802197 from LASSO
lasso.RMSE <- sqrt(mean((lasso.val - y[validate])^2)); lasso.RMSE
## retrain the model with the full data set
lasso.out <- glmnet(x, y, alpha=1, lambda=grid)
## view the coefficients of the model, 69 out of 146 have been shrinked to 0
lasso.coef  <- predict(lasso.out, type = "coefficients", s=lasso.bestlamda); lasso.coef[1:15,]
## get the prediction results for the test data
lasso.test  <- predict(lasso.out, s=lasso.bestlamda, type="response", newx=x.test); head(lasso.test, 10)


## write the prediction results to a csv file
predict.test <- cbind(ridge.test, lasso.test)
write.table(predict.test, file="Results from Sammy Yu.csv", sep=",", row.names=FALSE, col.names=c("Ridge.InterestRate","Lasso.InterestRate"))


#############################################################################################################
# WRITE UP:
# The algorithms I used to develop the models are Ridge Regression and LASSO. Both algorithms use the 
# regularization thru the parameter lambda to minimize the overfitting. In terms of performance,
# Ridge regression has a slightly better RMSE (2.800686) than the one in LASSO (2.802197) for this particular dataset.
#
# However, LASSO has an advantage over Ridge regression that it not only can minimize the overfitting, but
# also force some coefficients to exactly zero. For example, for the LASSO model here, 69 out of its 146 coefficients
# are exactly zero. in other words, the dimensions are almost cut in half. In this case, the model is simpler
# and easier to interpret.
#############################################################################################################
