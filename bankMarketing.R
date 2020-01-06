#---------------------------------------------------------------------------
# Load data from Internet
#
library(readr)
library(tidyverse)
# Load train dataset
file_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
temp <- tempfile()
download.file(file_url, temp)
unzip(temp, "bank-additional/bank-additional-full.csv")
train_set <- read_csv2("bank-additional/bank-additional-full.csv", col_names = TRUE)
# Load test dataset
unzip(temp, "bank-additional/bank-additional.csv")
test_set <- read_csv2("bank-additional/bank-additional.csv", col_names = TRUE)

#---------------------------------------------------------------------------
# Clean/Normalise data
#
train_set <- train_set %>% mutate(age = as.integer(age), 
                                  job = as.factor(job), 
                                  marital = as.factor(marital), 
                                  education = as.factor(education), 
                                  default = as.factor(default), 
                                  housing = as.factor(housing), 
                                  loan = as.factor(loan), 
                                  contact = as.factor(contact), 
                                  month = as.factor(month), 
                                  day_of_week = as.factor(day_of_week),
                                  duration = as.integer(duration),
                                  campaign = as.integer(campaign),
                                  pdays = as.integer(pdays),
                                  previous = as.integer(previous),
                                  emp.var.rate = as.integer(emp.var.rate),
                                  cons.price.idx = as.integer(cons.price.idx),
                                  cons.conf.idx = as.integer(cons.conf.idx),
                                  euribor3m = as.integer(euribor3m),
                                  nr.employed = as.integer(nr.employed),
                                  poutcome = as.factor(poutcome), 
                                  y = as.factor(y))
test_set <- test_set %>% mutate(age = as.integer(age), 
                                  job = as.factor(job), 
                                  marital = as.factor(marital), 
                                  education = as.factor(education), 
                                  default = as.factor(default), 
                                  housing = as.factor(housing), 
                                  loan = as.factor(loan), 
                                  contact = as.factor(contact), 
                                  month = as.factor(month), 
                                  day_of_week = as.factor(day_of_week),
                                  duration = as.integer(duration),
                                  campaign = as.integer(campaign),
                                  pdays = as.integer(pdays),
                                  previous = as.integer(previous),
                                  emp.var.rate = as.integer(emp.var.rate),
                                  cons.price.idx = as.integer(cons.price.idx),
                                  cons.conf.idx = as.integer(cons.conf.idx),
                                  euribor3m = as.integer(euribor3m),
                                  nr.employed = as.integer(nr.employed),
                                  poutcome = as.factor(poutcome), 
                                  y = as.factor(y))
train_set2 <- train_set
## Remove na's
train_set2$nr.employed[is.na(train_set2$nr.employed)] <- 0

#---------------------------------------------------------------------------
# Executing data modelling
#

# Fit with rpart
library(caret)
library(rpart)
set.seed(1)
#--------------------------------------------------------------------------
# Tuning process followed has been commented out and only optimised instruction is kept
#--------------------------------------------------------------------------
# Tune cp
##-- fit <- train(y~., method = "rpart", tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)), data = train_set2)
cp <- 0.002083333 # fit$bestTune
fit_rpart <- rpart(y~., data = train_set, cp = cp)
y_hat <- predict(fit_rpart, test_set)
y_hat <- ifelse(y_hat[,1]>y_hat[,2],"no", "yes")
y_hat <- factor(y_hat)
#    Calculate performance metrics
acc <- confusionMatrix(y_hat, test_set$y)$overall["Accuracy"]
#    Record results
results <- data.frame(method = "rpart", accuracy = acc)

# Fit with Rborist
library(Rborist)
set.seed(1)
#--------------------------------------------------------------------------
# Tuning process followed has been commented out and only optimised instruction is kept
#--------------------------------------------------------------------------
##-- control <- trainControl(method="cv", number=5, p=0.8)
##-- grid <- expand.grid(minNode=c(6, 12), predFixed=c(5, 9, 12, 16, ncol(train_set2)-1))
##-- fit <- train(y~., method = "Rborist", data = train_set2, type = "class", nTree=50, trControl=control, tuneGrid=grid, nSamp=5000)
pf <- 20 # fit$bestTune$predFixed
mn <- 12 # fit$bestTune$minNode
fit_rf <- Rborist(x = select(train_set2, -y), y = train_set2$y, type="class", nTree=1000, minNode=mn, predFixed=ifelse(pf>ncol(train_set2)-1,ncol(train_set2)-1,pf))
y_hat <- predict(fit_rf, select(test_set, -y))
#    Calculate performance metrics
acc <- confusionMatrix(y_hat$yPred, test_set$y)$overall["Accuracy"]
#    Record results
results <- suppressWarnings(bind_rows(results, data.frame(method = "Rborist", accuracy = acc)))

# Fit with rf
library(randomForest)
set.seed(1)
#--------------------------------------------------------------------------
# Tuning process followed has been commented out and only optimised instruction is kept
#--------------------------------------------------------------------------
##-- nodesize <- seq(1, 51, 10)
##-- ac1 <- sapply(nodesize, function(ns){
##--   train(y ~ ., method = "rf", data = train_set2,
##--         tuneGrid = data.frame(mtry = 2),
##--         nodesize = ns)$results$Accuracy
##-- })
##-- 
##-- qplot(nodesize, ac1, main = paste("Node size for higher rf accuracy is ", (which.max(ac1)-1)*10+1), xlab = "Node size", ylab = "Accuracy")
ns <- 31 # (which.max(ac1)-1)*10+1
fit_rf <- randomForest(y ~ ., data=train_set2, nodesize = ns)
y_hat <- predict(fit_rf, test_set)
#    Calculate performance metrics
acc <- confusionMatrix(y_hat, test_set$y)$overall["Accuracy"]
#    Record results
results <- suppressWarnings(bind_rows(results, data.frame(method = "randomForest", accuracy = acc)))

# Fit with Gradient Boosting Machines gbm_h2o
library(h2o)
h2o::h2o.init()
h2o::h2o.no_progress()
set.seed(1)
fit <- train(y~., method = "gbm_h2o", data = train_set2, verbose = FALSE)
y_hat <- predict(fit, test_set)
#    Calculate performance metrics
acc <- confusionMatrix(y_hat, test_set$y)$overall["Accuracy"]
#    Record results
results <- suppressWarnings(bind_rows(results, data.frame(method = "gbm_h2o", accuracy = acc)))

# Fit with Gradient Boosting Machines gbm
set.seed(1)
fit <- train(y~., method = "gbm", data = train_set2, verbose = FALSE)
y_hat <- predict(fit, test_set)
#    Calculate performance metrics
acc <- confusionMatrix(y_hat, test_set$y)$overall["Accuracy"]
#    Record results
results <- suppressWarnings(bind_rows(results, data.frame(method = "gbm", accuracy = acc)))

library(gbm)
# Fit with gbm bernoulli distribution with tuning
set.seed(1)
#--------------------------------------------------------------------------
# Tuning process followed has been commented out and only optimised instruction is kept
#--------------------------------------------------------------------------
#    Initial fit with test parameters to determine optimal number of boosting interactions
##-- gbm.fit <- gbm(
##--   formula = unclass(y)-1 ~ .,
##--   distribution = "bernoulli",
##--   data = train_set2,
##--   n.trees = 60000,
##--   interaction.depth = 1,
##--   shrinkage = 0.001,
##--   cv.folds = 5,
##--   n.cores = NULL, # will use all cores by default
##--   verbose = FALSE
##-- )  
#    Get the optimal number of boosting iterations for gbm object
##-- best.iter = gbm::gbm.perf(gbm.fit, method="cv")
##-- best.iter
##-- control = trainControl(method="cv", number=5, returnResamp = "all")
#    parameters bernoulli
##-- grid <- expand.grid(.n.trees=best.iter, .shrinkage=c(0.01, 0.05), .interaction.depth=c(1, 3), .n.minobsinnode=1)
##-- fit <- train(y ~ ., data = train_set2, method = "gbm", distribution = "bernoulli", trControl = control, verbose = FALSE, tuneGrid = grid)
##-- fit$bestTune

fit_gbm <- gbm(
  unclass(y)-1 ~ .,
  distribution = "bernoulli",
  data = train_set2,
  n.trees = 59994,
  interaction.depth = 1,
  shrinkage = 0.01, 
  n.minobsinnode = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  
y_hat <- predict(fit_gbm, test_set, n.trees = 59994, type = "response")
y_hat <- factor(ifelse(y_hat>=0.5, "yes", "no"))
#    Calculate performance metrics
acc <- confusionMatrix(y_hat, test_set$y)$overall["Accuracy"]
#    Record results
results <- suppressWarnings(bind_rows(results, data.frame(method = "tuned gbm/bernoulli", accuracy = acc)))

# Fit with gbm adaboost distribution with tuning
set.seed(1)
#--------------------------------------------------------------------------
# Tuning process followed has been commented out and only optimised instruction is kept
#--------------------------------------------------------------------------
#    Initial fit with test parameters to determine optimal number of boosting interactions
##-- gbm.fit <- gbm(
##--   formula = unclass(y)-1 ~ .,
##--   distribution = "adaboost",
##--   data = train_set2,
##--   n.trees = 60000,
##--   interaction.depth = 1,
##--   shrinkage = 0.001,
##--   cv.folds = 5,
##--   n.cores = NULL, # will use all cores by default
##--   verbose = FALSE
##-- )  
#    Get the optimal number of boosting iterations for gbm object
##-- best.iter = gbm::gbm.perf(gbm.fit, method="cv")
##-- best.iter

##-- control = trainControl(method="cv", number=5, returnResamp = "all")
#    parameters adaboost
##-- grid <- expand.grid(.n.trees=best.iter, .shrinkage=0.01, .interaction.depth=c(1, 3), .n.minobsinnode=1)
##-- fit <- train(y ~ ., data = train_set2, method = "gbm", distribution = "adaboost", trControl = control, verbose = FALSE, tuneGrid = grid)
##-- fit$bestTune
fit_gbm <- gbm(
  unclass(y)-1 ~ .,
  distribution = "adaboost",
  data = train_set2,
  n.trees = 59972,
  interaction.depth = 3,
  shrinkage = 0.01, 
  n.minobsinnode = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)
y_hat <- predict(fit_gbm, test_set, n.trees = 59972)
y_hat <- plogis(2*y_hat) # Convert predictions from logit scale to probability
y_hat <- factor(ifelse(y_hat>=0.5, "yes", "no"))
#    Calculate performance metrics
acc <- confusionMatrix(y_hat, test_set$y)$overall["Accuracy"]
#    Record results
results <- suppressWarnings(bind_rows(results, data.frame(method = "tuned gbm/adaboost", accuracy = acc)))

# Display results table and option with best accuracy
results %>% arrange(desc(accuracy))
message("The highest accuracy is delivered by ", results$method[which.max(results$accuracy)], " at ", max(results$accuracy))
