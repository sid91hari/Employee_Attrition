setwd("E:\\Hackerearth\\Employee Attrition\\Dataset")

library(dplyr)
library(caret)
library(mice)
library(Metrics)
library(MLmetrics)
library(MASS)
library(leaps)

# Load dataset

tr <- read.csv("Train.csv",stringsAsFactors = F)
te <- read.csv("Test.csv",stringsAsFactors = F)

# Checking for blanks

colSums(tr=='')
colSums(te=='')

# Checking NAs
round(colSums(is.na(tr))*100/nrow(tr),2)

str(tr)


summary(tr)

imputedValues <- mice(data=tr
                      , seed=42     # keep to replicate results
                      , method="cart" # model you want to use
                      , m=5           # Number of multiple imputations
                      , maxit = 1     # number of iterations
)

# impute the missing values in our tr data.frame
tr <- mice::complete(imputedValues,1) # completely fills in the missing

imputedValues <- mice(data=te
                      , seed=42     # keep to replicate results
                      , method="cart" # model you want to use
                      , m=5           # Number of multiple imputations
                      , maxit = 1     # number of iterations
)

# impute the missing values in our tr data.frame
te <- mice::complete(imputedValues,1) # completely fills in the missing

str(tr)


sapply(tr,function(x){length(unique(x))})

tr2 <- tr
te2 <- te

hist(tr$Attrition_rate)

library(rcompanion)

plotNormalHistogram(tr$Attrition_rate)


qqnorm(tr$Attrition_rate,
       ylab="Attrition Rate")

qqline(tr$Attrition_rate,
       col="red")

pairs(tr[c('Age','Time_of_service','growth_rate','Attrition_rate')])


gender_attrition <- aggregate(Attrition_rate~Gender, data=tr, FUN=mean, na.rm=T)

gender_attrition


relation_attrition <- aggregate(Attrition_rate~Relationship_Status, data=tr, FUN=mean, na.rm=T)

relation_attrition


# Store from backup

# tr <- tr2
# te <- te2

fact_cols <- c('Hometown','Decision_skill_possess','Compensation_and_Benefits','Unit')

layout(matrix(c(1,1,1,2,2,2,3,3,3,4,4,4), 4, 3, byrow = TRUE))

for (i in fact_cols){
  plot_table <- aggregate(Attrition_rate ~ get(i), data=tr, FUN=mean, na.rm=T)
  plot_table <- plot_table %>% arrange(desc(Attrition_rate))
  names(plot_table)[1] <- i
  barplot(plot_table$Attrition_rate,names = plot_table[,i],
          xlab = i, ylab = "Attrition Rate",
          main = paste("Attrition Rate for",i))
}

# One Hot Encoding - Gender and Relationship_Status

tr$Gender <- ifelse(tr$Gender == "F",1,0)
te$Gender <- ifelse(te$Gender == "F",1,0)

tr$Relationship_Status <- ifelse(tr$Relationship_Status == "Married",1,0)
te$Relationship_Status <- ifelse(te$Relationship_Status == "Married",1,0)


tr_id <- tr$Employee_ID
te_id <- te$Employee_ID

tr$Employee_ID <- NULL
te$Employee_ID <- NULL


# Label Encoding 

# for (i in fact_cols){
#   label_table <- aggregate(Attrition_rate ~ get(i), data=tr, FUN=mean, na.rm=T)
#   label_table <- label_table %>% arrange(desc(Attrition_rate))
#   label_table$label <- as.numeric(rownames(label_table))
#   names(label_table) <- c(i,'Attrition_rate',paste(i,'label',sep = '_'))
#   label_table <- label_table[,c(1,3)]
#   tr <- tr %>% inner_join(label_table, by = i)
#   te <- te %>% inner_join(label_table, by = i)
#   }


for (i in fact_cols){
  tr[[i]] <- as.numeric(as.factor(tr[[i]]))
  te[[i]] <- as.numeric(as.factor(te[[i]]))
}

names(tr)

d <- tr

y <- d$Attrition_rate

d$Attrition_rate <- NULL



#### Pre Processing

preProcValues <- preProcess(d, method = c("center","scale","BoxCox"))

d <- predict(preProcValues, d)

# Do for test

preProctest <- preProcess(te, method = c("center","scale","BoxCox"))

te <- predict(preProctest, te)

d$y <- y

############ Modeling ############

# Define Eval Metric

score <- function (data,
                   lev = NULL,
                   model = NULL) {
  out <- 100*max(0,1-RMSE(data$obs, data$pred))
  names(out) <- "Score"
  out
}

# Train control

ctrl <- trainControl(method="cv",     # cross-validation set approach to use
                     number=5,        # k number of times to do k-fold
                     classProbs = FALSE,  # if you want probabilities
                     summaryFunction = score,  # for regression
                     allowParallel=T,
                     verboseIter = T
)


# GBM

gbm_grid <- expand.grid(n.trees = seq(from = 500, to = 1000, by = 100), 
                        interaction.depth = c(1:10), 
                        shrinkage = c(0.01,0.05,0.1),
                        n.minobsinnode=10)


set.seed(42)
gbm_model <- train(y~.,
                   data = d,        # train set used to build model
                   method = "gbm",      # type of model you want to build
                   distribution = "poisson",
                   trControl = ctrl,    # how you want to learn
                   metric = "Score",       # performance measure
                   tuneGrid = gbm_grid)


y_pred1 <- predict(gbm_model,d)

print(RMSE(y_pred1,d$y))

# 0.185127

y_pred1 <- predict(gbm_model,te)

results <- data.frame(cbind(te_id,y_pred1))

names(results) <- c("Employee_ID","Attrition_rate")	

write.table(results, "Submissions\\gbm_1.csv", row.names = F, sep = ",")

# 81.22130 One Hot


##### RF

ctrl <- trainControl(method="cv",     # cross-validation set approach to use
                     number=5,        # k number of times to do k-fold
                     classProbs = FALSE,  # if you want probabilities
                     summaryFunction = score,  # for regression
                     allowParallel=T,
                     verboseIter = T
)


rf_grid <- expand.grid(mtry = c(1:12))

set.seed(42)
rf_model <- train(y~.,
                   data = d,        # train set used to build model
                   method = "rf",      # type of model you want to build
                   trControl = ctrl,    # how you want to learn
                   metric = "Score",       # performance measure
                   tuneGrid = rf_grid)


y_pred2 <- predict(rf_model,d)

print(RMSE(y_pred2,d$y))

# 0.1705622

y_pred2 <- predict(rf_model,te)

results <- data.frame(cbind(te_id,y_pred2))

names(results) <- c("Employee_ID","Attrition_rate")	

write.table(results, "Submissions\\rf_1.csv", row.names = F, sep = ",")

# 81.22178 One Hot

##### XGBoost

ctrl <- trainControl(method="cv",     # cross-validation set approach to use
                     number=5,        # k number of times to do k-fold
                     classProbs = FALSE,  # if you want probabilities
                     summaryFunction = score,  # for regression
                     allowParallel=T,
                     verboseIter = T
)


xgb_grid <- expand.grid(nrounds = seq(from = 200, to = 1000, by = 100),
                        max_depth = c(1:7),
                        colsample_bytree = c(0.8,0.9),
                        eta = c(0.005,0.01,0.05),
                        gamma = 0,
                        min_child_weight = c(1,3,5),
                        subsample = c(0.8,0.9))

set.seed(42)
xgb_model  <- train(y ~ .,               # model specification
                    data = d,        # train set used to build model
                    method = "xgbTree",      # type of model you want to build
                    trControl = ctrl,    # how you want to learn
                    metric = "Score",       # performance measure
                    tuneGrid = xgb_grid,
                    seed = 42)



y_pred3 <- predict(xgb_model,d)

print(RMSE(y_pred3,d$y))

# 0.1852718

y_pred3 <- predict(xgb_model,te)

results <- data.frame(cbind(te_id,y_pred3))

names(results) <- c("Employee_ID","Attrition_rate")	

write.table(results, "Submissions\\xgb_1.csv", row.names = F, sep = ",")

# 81.22266 One Hot

########## LightGBM

#Data partition
set.seed(42) # set a seed so you can replicate your results

inTrain <- createDataPartition(y = d$y,   # outcome variable
                               p = .8,   # % of training data you want
                               list = F)

# create your partitions
train <- d[inTrain,]  # training data set
val <- d[-inTrain,]  # test data set

library(lightgbm)

dtrain <- lgb.Dataset(as.matrix(train[1:ncol(train)-1]), label=train$y)
dval <- lgb.Dataset.create.valid(dtrain, data = as.matrix(val[1:ncol(val)-1]), label = val$y)

valids <- list(train = dtrain,test = dval)

lgb.grid = list(objective = "regression",
                metric = "rmse",
                lambda_l1 = 1,
                lambda_l2 = 1)

set.seed(42)
lgb.model.cv = lgb.cv(params = lgb.grid, data = dtrain, learning_rate = 0.001, num_leaves = 70,
                      num_threads = 4, nrounds = 1000, early_stopping_rounds = 100,
                      eval_freq = 20, nfold = 5, stratified = TRUE)

best.iter = lgb.model.cv$best_iter

set.seed(42)
lgb.model = lgb.train(params = lgb.grid, data = dtrain, learning_rate = 0.001,
                      num_leaves = 70, num_threads = 4 , nrounds = best.iter,
                      eval_freq = 20, valids = valids, verbose = 2)


y_pred_train <- predict(lgb.model, as.matrix(train[1:ncol(train)-1]))


# train's rmse:0.185227	test's rmse:0.186749 

# train's rmse:0.184541	test's rmse:0.18699

# train's rmse:0.183167	test's rmse:0.187013

y_pred4 <- predict(lgb.model, as.matrix(te))

results <- data.frame(cbind(te_id,y_pred4))

names(results) <- c("Employee_ID","Attrition_rate")	

write.table(results, "Submissions\\lgb_1.csv", row.names = F, sep = ",")

# 81.25678 One Hot
# 81.25313 Label
# 81.25949 Mean Encoding
# 81.24928

##### Ridge

# Train control

ctrl <- trainControl(method="cv",     # cross-validation set approach to use
                     number=10,        # k number of times to do k-fold
                     classProbs = FALSE,  # if you want probabilities
                     summaryFunction = score,  # for regression
                     allowParallel=T,
                     verboseIter = T
)

# Set parameters for Ridge

ridge_grid <- expand.grid(alpha = 0, lambda = 1)

set.seed(42)
ridge_model <- train(y~.,
                   data = d,        # train set used to build model
                   method = "glmnet",      # type of model you want to build
                   trControl = ctrl,    # how you want to learn
                   metric = "Score",       # performance measure
                   tuneGrid = ridge_grid)



plot(varImp(ridge_model))

y_pred6_tr <- predict(ridge_model,d)

print(RMSE(y_pred6_tr,d$y))

# 0.1855831


y_pred6 <- predict(ridge_model,te)

results <- data.frame(cbind(te_id,y_pred6))

names(results) <- c("Employee_ID","Attrition_rate")	

write.table(results, "Submissions\\ridge_1.csv", row.names = F, sep = ",")

# 81.26086 One Hot

# 81.26201 Label

# 81.26213 Label No Pre Process

# 81.25866 Mean Encoding No Pre Process

# 81.26026 Ordinal Label No Pre Process

# 81.26217 Label Pre Process z and BoxCox

##### Lasso

lasso_grid <- expand.grid(alpha = 1, lambda = 1)

set.seed(42)
lasso_model <- train(y~.,
                     data = d,        # train set used to build model
                     method = "glmnet",      # type of model you want to build
                     trControl = ctrl,    # how you want to learn
                     metric = "Score",       # performance measure
                     tuneGrid = lasso_grid)

y_pred7_tr <- predict(lasso_model,d)

print(RMSE(y_pred7_tr,d$y))

# 0.18574
# 0.18574

y_pred7 <- predict(lasso_model,te)

results <- data.frame(cbind(te_id,y_pred7))

names(results) <- c("Employee_ID","Attrition_rate")	

write.table(results, "Submissions\\lasso_1.csv", row.names = F, sep = ",")

# 81.26120 One Hot
# 81.26120 Label
# 81.26120 Label No Preprocess