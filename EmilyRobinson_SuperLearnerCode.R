install.packages(c("knitr", "kernlab", "dplyr", "caret", "ggplot2", "car", "randomForest", "corrplot", "ROCR", "SuperLearner", "ranger", "knitcitations", "glmnet", "gam"))
library(dplyr)
library(caret)
library(ggplot2)
library(car)

# Import and Prep Data
setwd("C:/Users/EmilyARobinson/Dropbox/823- Methods III/Project")
Diabetes <- read.csv("diabetes.csv")
colSums(is.na(Diabetes))

# Diagnostics
glm_mod <- glm(Outcome ~., data = Diabetes, family = binomial())
glm_mod$coefficients

#OUTLIERS
outlierTest(glm_mod)

#INFLUENTIAL OBS
# Cook's D plot
# identify D values > 4/(n-k-1) 
cutoff <- 4/((nrow(Diabetes)-length(glm_mod$coefficients)-2)) 
plot(glm_mod, which=4, cook.levels=cutoff)
# Influence Plot 
influencePlot(glm_mod,	id.method="identify", main="Influence Plot", sub="Circle size is proportial to Cook's Distance" )

#COLLINEARITY
panel.cor <- function(x, y, digits=2, prefix="", cex.cor) 
{
  usr <- par("usr"); on.exit(par(usr)) 
  par(usr = c(0, 1, 0, 1)) 
  r <- abs(cor(x, y)) 
  txt <- format(c(r, 0.123456789), digits=digits)[1] 
  txt <- paste(prefix, txt, sep="") 
  if(missing(cex.cor)) cex <- 0.8/strwidth(txt) 
  
  test <- cor.test(x,y) 
  # borrowed from printCoefmat
  Signif <- symnum(test$p.value, corr = FALSE, na = FALSE, 
                   cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                   symbols = c("***", "**", "*", ".", " ")) 
  
  text(0.5, 0.5, txt, cex = cex * r) 
  text(.8, .8, Signif, cex=cex, col=2) 
}
pairs(Diabetes, lower.panel=panel.smooth, upper.panel=panel.cor)

library(corrplot)
corrplot(cor(Diabetes))

# Prep Data
Diabetes <- Diabetes[-c(14, 229, 350, 454, 503, 707),]
outcome  <- Diabetes$Outcome
data     <- subset(Diabetes, select = -Outcome)
str(data)
dim(data)

# Variable Importance
library(randomForest)
rf_mod <- randomForest(factor(Outcome) ~ ., data = Diabetes, family = "binomial")
VI <- as.data.frame(rf_mod$importance)
VI$variable <- rownames(VI)
VI <- VI[order(VI[,1], decreasing = T),]
barplot(VI[,1], names.arg = VI[,2], ylab = "Mean Decrease Gini", beside = F)

# Explore Data
glm_mod2 <- glm(Outcome ~., data = Diabetes, family = binomial())
glm_mod2$coefficients
avPlots(glm_mod2)

# Create train & test sets
set.seed(904)
pct       <- 0.6
train_obs <- sample(nrow(data), floor(nrow(data)*pct))
X_train   <- data[train_obs, ]
X_holdout <- data[-train_obs, ]
Y_train   <- outcome[train_obs]
Y_holdout <- outcome[-train_obs]

# Review the outcome variable distribution.
table(Y_train, useNA = "ifany")

# Misclassification
misclass = function(yhat,y) {
  temp <- table(yhat,y)
  cat("Table of Misclassification\n")
  cat("(row = predicted, col = actual)\n")
  print(temp)
  cat("\n\n")
  numcor <- sum(diag(temp))
  numinc <- length(y) - numcor
  mcr <- numinc/length(y)
  cr  <- sum(diag(temp))/sum(temp)
  cat(paste("Classification Rate = ",format(cr,digits=3)))
  cat("\n")
  cat(paste("Misclassification Rate = ",format(mcr,digits=3)))
  cat("\n")
}

# Review ROC curve
library(ROCR)
ROC_eval <- function(fit, pred, y){
  pred_rocr <- ROCR::prediction(pred, Y_holdout)
  auc       <- ROCR::performance(pred_rocr, measure = "auc", x.measure = "cutoff")@y.values[[1]]
  perf      <- performance( pred_rocr, "tpr", "fpr" )
  plot(perf, main = paste("Fit =", fit, "\n", "AUC = ", round(auc,4)))
  abline(0,1)
}

# SuperLearner Package
citation(package = "SuperLearner", lib.loc = NULL)
library(SuperLearner)

# Review available models.
listWrappers()

# Fit GLM
SL.glm
sl_glm   = SuperLearner(Y = Y_train, X = X_train, newX = X_holdout, family = binomial(), SL.library = "SL.glm")
sl_glm
pred_glm = predict(sl_glm, X_holdout)
ROC_eval("glm", pred_glm$pred, Y_holdout)
conv.preds_glm <- ifelse(pred_glm$pred >= 0.5, 1, 0)
misclass(conv.preds_glm, Y_holdout)

# Fit GAM
SL.gam
sl_gam   = SuperLearner(Y = Y_train, X = X_train, newX = X_holdout, family = binomial(), SL.library = "SL.gam")
sl_gam
pred_gam = predict(sl_gam, X_holdout)
ROC_eval("gam", pred_gam$pred, Y_holdout)
conv.preds_gam <- ifelse(pred_gam$pred >= 0.5, 1, 0)
misclass(conv.preds_gam, Y_holdout)

# Fit Lasso
SL.glmnet
sl_lasso = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = "SL.glmnet")
sl_lasso
pred_lasso = predict(sl_lasso, X_holdout)
ROC_eval("lasso", pred_lasso$pred, Y_holdout)
conv.preds_lasso <- ifelse(pred_lasso$pred >= 0.5, 1, 0)
misclass(conv.preds_lasso, Y_holdout)

# Fit Ridge
learners    = create.Learner("SL.glmnet", params = list(alpha = 0))
learners$names
sl_ridge = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = "SL.glmnet_1")
sl_ridge
pred_ridge = predict(sl_ridge, X_holdout)
ROC_eval("ridge", pred_ridge$pred, Y_holdout)
conv.preds_ridge <- ifelse(pred_ridge$pred >= 0.5, 1, 0)
misclass(conv.preds_ridge, Y_holdout)

# Fit Ranger
library(ranger)
SL.ranger
set.seed(904)
sl_ranger = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = "SL.ranger")
sl_ranger
pred_ranger = predict(sl_ranger, X_holdout)
ROC_eval("ranger", pred_ranger$pred, Y_holdout)
conv.preds_ranger <- ifelse(pred_ranger$pred >= 0.5, 1, 0)
misclass(conv.preds_ranger, Y_holdout)

# Fit ksvm
sl_ksvm = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = "SL.ksvm")
sl_ksvm
pred_ksvm = predict(sl_ksvm, X_holdout)
ROC_eval("ksvm", pred_ksvm$pred, Y_holdout)
conv.preds_ksvm <- ifelse(pred_ksvm$pred >= 0.5, 1, 0)
misclass(conv.preds_ksvm, Y_holdout)

# Fit Mean
SL.mean
sl_mean = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = "SL.mean")
sl_mean
pred_mean = predict(sl_mean, X_holdout)
ROC_eval("mean", pred_mean$pred, Y_holdout)
conv.preds_mean <- ifelse(pred_mean$pred >= 0.5, 1, 0)
misclass(conv.preds_mean, Y_holdout)

# Review Weights
review_weights = function(cv_sl) {
  meta_weights = coef(cv_sl)
  means = colMeans(meta_weights)
  sds = apply(meta_weights, MARGIN = 2,  FUN = function(col) { sd(col) })
  mins = apply(meta_weights, MARGIN = 2, FUN = function(col) { min(col) })
  maxs = apply(meta_weights, MARGIN = 2, FUN = function(col) { max(col) })
  # Combine the stats into a single matrix.
  sl_stats = cbind("mean(weight)" = means, "sd" = sds, "min" = mins, "max" = maxs)
  # Sort by decreasing mean weight.
  sl_stats[order(sl_stats[, 1], decreasing = T), ]
}

# Fit Multiple Models
SL.library <- c("SL.glm", "SL.gam", "SL.glmnet", "SL.glmnet_1", "SL.ranger", "SL.ksvm", "SL.mean")
set.seed(904)
sl_stacked = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = SL.library, 
                          verbose = T, cvControl = list(V = 5), method = "method.NNLS")
sl_stacked

# Fit Super Learner
set.seed(904)
cv_stacked <- CV.SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = SL.library,
                              verbose = T, cvControl = list(V = 5), method = "method.NNLS")
review_weights(cv_stacked)
summary(cv_stacked)
plot(cv_stacked) + theme_bw(base_size = 15)

# Predictions
pred_sl <- predict.SuperLearner(sl_stacked, newdata = X_holdout, Y = Y_train, X = X_train)
par(mfrow = c(2,4))
ROC_eval("stacked", pred_sl$pred, Y_holdout)
ROC_eval("glm",     pred_sl$library.predict[,1], Y_holdout)
ROC_eval("gam",     pred_sl$library.predict[,2], Y_holdout)
ROC_eval("lasso",   pred_sl$library.predict[,3], Y_holdout)
ROC_eval("ridge",   pred_sl$library.predict[,4], Y_holdout)
ROC_eval("ranger",  pred_sl$library.predict[,5], Y_holdout)
ROC_eval("ksvm",    pred_sl$library.predict[,6], Y_holdout)
ROC_eval("mean",    pred_sl$library.predict[,7], Y_holdout)
par(mfrow = c(1,1))

conv.preds_sl <- ifelse(pred_sl$pred >= 0.5, 1, 0)
misclass(conv.preds_sl, Y_holdout)

# Adjust Parameters
SL.ranger
mtry_seq    <- seq(2, 5, 1)
n.trees_seq <- seq(500, 800, 100) 
learners.ranger <- create.Learner("SL.ranger", tune = list(mtry = mtry_seq, num.trees = n.trees_seq))
learners.ranger$names
set.seed(904)
cv_stacked.ranger <- CV.SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = learners.ranger$names,
                               verbose = T, cvControl = list(V = 5), method = "method.NNLS")
review_weights(cv_stacked.ranger)
summary(cv_stacked.ranger)
plot(cv_stacked.ranger) + theme_bw()
SL.ranger_5
SL.ranger_13
SL.ranger_14


SL.glmnet
alpha_seq <- seq(0, 1, 0.1)
learners.glmnet <- create.Learner("SL.glmnet", tune = list(alpha = alpha_seq))
learners.glmnet$names
set.seed(904)
cv_stacked.glmnet <- CV.SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = learners.glmnet$names,
                                     verbose = T, cvControl = list(V = 5), method = "method.NNLS")
review_weights(cv_stacked.glmnet)
summary(cv_stacked.glmnet)
plot(cv_stacked.glmnet) + theme_bw()

SL.gam
deg_seq <- seq(1, 8, 1)
cts_seq <- seq(2, 6, 1)
learners.gam <- create.Learner("SL.gam", tune = list(deg.gam = deg_seq, cts.num = cts_seq))
learners.gam$names
set.seed(904)
cv_stacked.gam <- CV.SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = learners.gam$names,
                                     verbose = T, cvControl = list(V = 5), method = "method.NNLS")
review_weights(cv_stacked.gam)
summary(cv_stacked.gam)
plot(cv_stacked.gam) + theme_bw()
SL.gam_12

SL.library2 <- c("SL.glm", "SL.gam_12", "SL.glmnet_1",  "SL.ranger_5", "SL.mean")
set.seed(904)
cv_stacked2 <- CV.SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = SL.library2,
                               verbose = T, cvControl = list(V = 5), method = "method.NNLS")
review_weights(cv_stacked2)
summary(cv_stacked2)
plot(cv_stacked2) + theme_bw()

sl_stacked2 = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = SL.library2, 
                          verbose = T, cvControl = list(V = 5), method = "method.NNLS")

pred_sl2 <- predict.SuperLearner(sl_stacked2, newdata = X_holdout, Y = Y_train, X = X_train)
par(mfrow = c(1,2))
ROC_eval("stacked- revised", pred_sl2$pred, Y_holdout)
ROC_eval("gam",              pred_sl$library.predict[,2], Y_holdout)
par(mfrow = c(1,1))
