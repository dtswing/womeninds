setwd("~/meetup")

library(caret)
library(dplyr)
library(tidyr)
library(doMC)
library(randomForest)
library(pROC)
library(ROCR)

registerDoMC(4)


airline <- read.csv("meetup_data/train-0.1m.csv")

#clean the feature names, change "c-" to "c_".  (this will cause some issues if you use reformulate function later)
airline$DayOfWeek <- gsub("c-","c_",airline$DayOfWeek)
airline$DayOfWeek <- as.factor(airline$DayOfWeek)
airline$DayofMonth <- gsub("c-","c_",airline$DayofMonth)
airline$DayofMonth <- as.factor(airline$DayofMonth)
airline$Month <- gsub("c-","c_",airline$Month)
airline$Month <- as.factor(airline$Month)

#vectorize features
system.time({
  airline_vec <-  model.matrix(dep_delayed_15min ~ ., data = airline)
})
dim(airline_vec)

airline_vec <- airline_vec[,-1]
airline_vec <- as.data.frame(airline_vec)
airline_vec$dep_delayed_15min <- airline$dep_delayed_15min

#Splitting the 100K dataset into train, test and (out-of-time) validation set¶
airline_vec_prior_2007 <- airline_vec %>% filter(Year < 2007)
airline_vec_2007 <- airline_vec %>% filter(Year == 2007)

set.seed(3456)
trainIndex <- createDataPartition(airline_vec_prior_2007$dep_delayed_15min, p = .6,
                                  list = FALSE,
                                  times = 1)
head(trainIndex)
airlineTrain <- airline_vec_prior_2007[ trainIndex,]
print(paste0("Number of obs in training: ",nrow(airlineTrain)))
airlineTest  <- airline_vec_prior_2007[-trainIndex,]
print(paste0("Number of obs in test: ",nrow(airlineTest)))

#take 10K random sample for grid search
idx_10k <- sample(nrow(airline_vec_prior_2007),1e4)
airline_10k <- airline_vec_prior_2007[idx_10k,]
#Split 10k sample into training, and test
idx_10k_train <- sample(nrow(airline_10k),floor(nrow(airline_10k)*.75))
airline_train_10k <- airline_10k[idx_10k_train,]
airline_test_10k <- airline_10k[-idx_10k_train,]
print(paste0("Number of obs in training: ",nrow(airline_train_10k)))
print(paste0("Number of obs in training: ",nrow(airline_test_10k)))

#Grid search with Caret
myGrid = expand.grid(mtry = c(1:5)*20)

set.seed(77)
cv_build_fold_idx <- createMultiFolds(airline_train_10k$dep_delayed_15min, k = 3, times = 1)

model_vars <- names(airline_train_10k)[!names(airline_train_10k)%in% c('dep_delayed_15min','(Intercept)')]

#set up your caret fit control, we use 3-fold CV with no repeat here
cvFitControl <- trainControl(method = "repeatedcv", repeats = 1, number = 3, index = cv_build_fold_idx,
                             classProbs = TRUE, summaryFunction = twoClassSummary, returnResamp = 'all', savePredictions = T,
                             allowParallel= T, verboseIter = T)

start = Sys.time()
model_rf <- train(reformulate(model_vars, response="dep_delayed_15min"), data = airline_train_10k, metric = "ROC", preProcess = NULL,
                  method = "rf", ntrees = 200, sampsize = c(70,70),tuneGrid = myGrid, trControl = cvFitControl, importance = T)

end = Sys.time()
duration = end - start
print(duration)

# See your grid search CV results
plot(model_rf)
# Check your CV model results
print(model_rf)

# Training AUC
pred <- prediction(predict(model_rf$finalModel, airline_train, type = "prob")[,2], airline_train$dep_delayed_15min)
performance(pred,"auc")@y.values[[1]]

# Test AUC
pred2 <- prediction(predict(model_rf$finalModel, airline_test, type = "prob")[,2], airline_test$dep_delayed_15min)
performance(pred2,"auc")@y.values[[1]]

#Plot AUC Curves
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col=rainbow(10))
perf2 <- performance(pred2, measure = "tpr", x.measure = "fpr") 
plot(perf2, add = TRUE, colorize = TRUE)
abline(a=0, b= 1)

# Variable importance
varImp(model_rf)

########################## Try 100k sample#############################
###Train best model on 100k data set (this takes ~1hr+ to run...)
myGrid = expand.grid(mtry = c(60))

set.seed(77)
cv_build_fold_idx_final <- createMultiFolds(airlineTrain$dep_delayed_15min, k = 3, times = 1)

model_vars <- names(airlineTrain)[!names(airlineTrain)%in% c('dep_delayed_15min','(Intercept)')]


cvFitControl_final <- trainControl(method = "repeatedcv", repeats = 1, number = 3, index = cv_build_fold_idx_final,
                                   classProbs = TRUE, summaryFunction = twoClassSummary, returnResamp = 'all', savePredictions = T,
                                   allowParallel= T, verboseIter = T)

start = Sys.time()
model_rf_final <- train(reformulate(model_vars, response="dep_delayed_15min"), data = airlineTrain, metric = "ROC", preProcess = NULL,
                        method = "rf", ntrees = 200, sampsize = c(70,70), tuneGrid = myGrid, trControl = cvFitControl, importance = T)

end = Sys.time()
duration = end - start
print(duration)

### performance of 100k 
pred_final <- prediction(predict(model_rf_final$finalModel, airlineTrain, type = "prob")[,2], airlineTrain$dep_delayed_15min)
perf_final <- performance(pred_final, measure = "tpr", x.measure = "fpr") 
print(paste0("Training AUC: ",performance(pred_final,"auc")@y.values[[1]]))

pred2_final <- prediction(predict(model_rf_final$finalModel, airlineTest, type = "prob")[,2], airlineTest$dep_delayed_15min)
perf2_final <- performance(pred2_final, measure = "tpr", x.measure = "fpr") 
print(paste0("Test AUC: ",performance(pred2_final,"auc")@y.values[[1]]))

plot(perf_final, col=rainbow(10))
plot(perf2_final, add = TRUE, colorize = TRUE)
abline(a=0, b= 1)

# Variable importance
varImp(model_rf_final)

# performance on validation set
pred_val_final <- prediction(predict(model_rf_final$finalModel, airline_vec_2007, type = "prob")[,2], airline_vec_2007$dep_delayed_15min)
perf_val_final <- performance(pred_val_final, measure = "tpr", x.measure = "fpr") 
print(paste0("Validation AUC: ",performance(pred_val_final,"auc")@y.values[[1]]))
