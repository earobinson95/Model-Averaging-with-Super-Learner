Improved Classification with Super Learner
========================================================
width: 1920
height: 1080
author: Emily Robinson
date: December 3, 2018
autosize: true
transition: linear
css: rpres.css



Motivating Example
========================================================
class: center-img

How many Christmas candies are in the jar?

- Some individuals guessed over.
- Some individuals guessed under.
- What if we average all of the guesses?

***

<img src="candy_jar.jpg" height="500px"/>

Types of Ensemble Methods
========================================================
- Model averaging
- Bagging: Averaging trees built to bootstrap samples of the training data.
- Random Forest: Building trees using random selected predictors at each stage.
- Boosting: Observations that are misclassified on a given stage in the boosting process are given more weight in next stage.  Thus the next model in the sequence will "work harder" to classify those observations correctly.
- Bumping: Rather than averaging models together, you fit multple models and select the best.
- Stacking

Ensemble Learning (Stacking)
========================================================
A procedure where multiple learner modules are applied on a dataset to extract multiple predictions, which are then combined into one composite prediction.

Three Steps:
- Ensemble Generation
  - Homogeneous if all base models belong to the same class of models.
  - Heterogeneous if all base models belong to a diverse set of models.
- Ensemble Pruning
- Ensemble Integration

Diversity is considered one of the key success factors of ensembles. 

[Steinki and Mohammad (2015)]

Weighted Average
========================================================
Suppose that $M$ predictors are available, $\hat{f_1}, \hat{f_2}, ..., \hat{f_M}$. Under a loss or risk function, a weight vector $\textbf{w}$ is created and the predictor $$\hat{f}(x)=\sum_{m=1}^{M}w_m\hat{f_m}(x).$$

How do you determine opitimum weights?
- Optimize $\textbf{w}$ over a defined function using cross validation techniques.
- Minimize mean square error.
- Maximize area under the curve (AUC) of receiver operating characteristic (ROC).

[Vardeman (2018)] and [Polley, LeDell, Kennedy, and van der Laan (2018)]

Weighted Average (Mean Squared Error)
========================================================
Consider $M=2$, then $E[y-\hat{f}_1(x)]=0$ and $E[y-\hat{f}_2(x)]=0$. Define $\hat{f}_\alpha = \alpha\hat{f}_1+(1-\alpha)\hat{f}_2.$ Then,

$$E[(y-\hat{f}_\alpha(x))^2] = E[(\alpha\hat{f}_1+(1-\alpha)\hat{f}_2)^2] = Var(\alpha\hat{f}_1+(1-\alpha)\hat{f}_2) = (\alpha, 1-\alpha) \text{COV} \begin{pmatrix} y - \hat{f}_1(x) \\ y -\hat{f}_2(x) \end{pmatrix} \begin{pmatrix} \alpha \\ 1-\alpha \end{pmatrix}$$

Since this is a quadratic function of $\alpha,$ that has a minimum, there is a minimizing $\alpha$ that produces a better expected loss function than either $\hat{f}_1(x)$ or $\hat{f}_2(x)$.

In general, $$\textbf{w}^{stack}=\underset{w}\arg\min\sum_{i=1}^N\left(y_i - \sum_{m=1}^M w_m\hat{f}_m^i(x_i)\right)^2$$
for the $m^{th}$ predictor fit to the training set with the $i^{th}$ case removed.

- optimizes a kind of leave-one-out cross validation
- ad hoc version is closer to k-fold cross validation


```r
library(SuperLearner)
method.NNLS()
method.NNLS2()
method.NNloglik()
method.CC_LS()
method.AUC(nlopt_method=NULL, optim_method="L-BFGS-B", bounds=c(0, Inf), normalize=TRUE)
```

Diabetes Dataset
========================================================
The dataset I have selected for my project is the Pima Indians Diabetes Database found on Kaggle through the UCI data repository https://www.kaggle.com/uciml/pima-indians-diabetes-database, originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 
- Females at least age 21 years old and of Pima Indian heritage
- Target variable, Outcome, is binary and indicates whether or not a patient has diabetes
- The objective is to correctly classify individuals to have diabetes or not based on several medical predictor variables such as the number of pregnancies the patient has had, their BMI, insulin level, age, etc. 


```
[1] "Pregnancies"              "Glucose"                 
[3] "BloodPressure"            "SkinThickness"           
[5] "Insulin"                  "BMI"                     
[7] "DiabetesPedigreeFunction" "Age"                     
[9] "Outcome"                 
```

[Smith, Everhart, Dickson, Knowler, and Johannes (1988)]

Diagnostics
========================================================

```r
glm_mod <- glm(Outcome ~., data = Diabetes, family = binomial())
outlierTest(glm_mod)
```

```
No Studentized residuals with Bonferonni p < 0.05
Largest |rstudent|:
    rstudent unadjusted p-value Bonferonni p
350 2.967294          0.0030043           NA
```

```r
cutoff <- 4/((nrow(Diabetes)-length(glm_mod$coefficients)-2)) 
plot(glm_mod, which=4, cook.levels=cutoff)
```

<img src="ProjectPresentation-figure/diagnostics-1.png" title="plot of chunk diagnostics" alt="plot of chunk diagnostics" style="display: block; margin: auto;" />

Diagnostics
========================================================

```r
influencePlot(glm_mod,	id.method="identify", main="Influence Plot", sub="Circle size is proportial to Cook's Distance" )
```

<img src="ProjectPresentation-figure/diagnostics2-1.png" title="plot of chunk diagnostics2" alt="plot of chunk diagnostics2" style="display: block; margin: auto;" />

```
      StudRes         Hat       CookD
14   1.002281 0.114022424 0.009536169
229 -2.634163 0.020610966 0.057205591
350  2.967294 0.003069042 0.024729204
454 -1.057796 0.078351507 0.007159288
503  2.786460 0.005137660 0.024505179
707  2.306667 0.023944183 0.031920741
```

Collinearity
========================================================

```r
pairs(Diabetes, lower.panel=panel.smooth, upper.panel=panel.cor)
```

<img src="ProjectPresentation-figure/Collinearity-1.png" title="plot of chunk Collinearity" alt="plot of chunk Collinearity" style="display: block; margin: auto;" />

Data Preparation
========================================================

```r
# Create response and predictor sets
Diabetes <- Diabetes[-c(14, 229, 350, 454, 503, 707),]
outcome  <- Diabetes$Outcome
data     <- subset(Diabetes, select = -Outcome)

# Create train & test sets
set.seed(904)
pct = 0.6
train_obs = sample(nrow(data), floor(nrow(data)*pct))
X_train = data[train_obs, ]
X_holdout = data[-train_obs, ]
Y_train = outcome[train_obs]
Y_holdout = outcome[-train_obs]

table(Y_train, useNA = "ifany")
```

```
Y_train
  0   1 
290 167 
```

Explore Dataset
========================================================

```r
glm_mod2 <- glm(Outcome ~., data = Diabetes, family = binomial())
glm_mod2$coefficients
```

```
             (Intercept)              Pregnancies                  Glucose 
            -9.033836090              0.120943571              0.038777313 
           BloodPressure            SkinThickness                  Insulin 
            -0.013052960             -0.001082154             -0.001094749 
                     BMI DiabetesPedigreeFunction                      Age 
             0.092884435              1.146457478              0.014083702 
```

Variable Importance
========================================================
<img src="ProjectPresentation-figure/VarImp-1.png" title="plot of chunk VarImp" alt="plot of chunk VarImp" style="display: block; margin: auto;" />

Added Variable Plots
========================================================

```r
avPlots(glm_mod2)
```

<img src="ProjectPresentation-figure/Explore2-1.png" title="plot of chunk Explore2" alt="plot of chunk Explore2" style="display: block; margin: auto;" />

Receiver Operating Characteristic curve (ROC curve)
========================================================
class: center-img
- A plot of the true positive rate (sensitivity) against the false positive rate (1-specificity) for the different possible cutpoints of a diagnostic test.
- Shows the tradeoff between sensitivity and specificity.
- Compare the area under the ROC curve (AUC)
  - A higher AUC indicates a better fit.

***

<img src="ROCexample.png" height="300px"/>

SuperLearner
========================================================
Super Learner uses V-fold cross-validation to build the optimal weighted combination of predictions from a library of candiate algorithms. [Naimi and Balzer (2018)]


```r
listWrappers()
```

```
 [1] "SL.bartMachine"      "SL.bayesglm"         "SL.biglasso"        
 [4] "SL.caret"            "SL.caret.rpart"      "SL.cforest"         
 [7] "SL.dbarts"           "SL.earth"            "SL.extraTrees"      
[10] "SL.gam"              "SL.gbm"              "SL.glm"             
[13] "SL.glm.interaction"  "SL.glmnet"           "SL.ipredbagg"       
[16] "SL.kernelKnn"        "SL.knn"              "SL.ksvm"            
[19] "SL.lda"              "SL.leekasso"         "SL.lm"              
[22] "SL.loess"            "SL.logreg"           "SL.mean"            
[25] "SL.nnet"             "SL.nnls"             "SL.polymars"        
[28] "SL.qda"              "SL.randomForest"     "SL.ranger"          
[31] "SL.ridge"            "SL.rpart"            "SL.rpartPrune"      
[34] "SL.speedglm"         "SL.speedlm"          "SL.step"            
[37] "SL.step.forward"     "SL.step.interaction" "SL.stepAIC"         
[40] "SL.svm"              "SL.template"         "SL.xgboost"         
[1] "All"
[1] "screen.corP"           "screen.corRank"        "screen.glmnet"        
[4] "screen.randomForest"   "screen.SIS"            "screen.template"      
[7] "screen.ttest"          "write.screen.template"
```

Fit GLM
========================================================


```r
sl_glm = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = "SL.glm")
sl_glm
```

```

Call:  
SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = "SL.glm") 

                Risk Coef
SL.glm_All 0.1524512    1
```

```r
pred_glm = predict(sl_glm, X_holdout)
ROC_eval("glm", pred_glm$pred, Y_holdout)
```

<img src="ProjectPresentation-figure/fitGLM-1.png" title="plot of chunk fitGLM" alt="plot of chunk fitGLM" style="display: block; margin: auto;" />

Glmnet Code
========================================================

```r
SL.glmnet
```

```
function (Y, X, newX, family, obsWeights, id, alpha = 1, nfolds = 10, 
    nlambda = 100, useMin = TRUE, loss = "deviance", ...) 
{
    .SL.require("glmnet")
    if (!is.matrix(X)) {
        X <- model.matrix(~-1 + ., X)
        newX <- model.matrix(~-1 + ., newX)
    }
    fitCV <- glmnet::cv.glmnet(x = X, y = Y, weights = obsWeights, 
        lambda = NULL, type.measure = loss, nfolds = nfolds, 
        family = family$family, alpha = alpha, nlambda = nlambda, 
        ...)
    pred <- predict(fitCV, newx = newX, type = "response", s = ifelse(useMin, 
        "lambda.min", "lambda.1se"))
    fit <- list(object = fitCV, useMin = useMin)
    class(fit) <- "SL.glmnet"
    out <- list(pred = pred, fit = fit)
    return(out)
}
<bytecode: 0x000000002358a3f0>
<environment: namespace:SuperLearner>
```

Fit Lasso Regression
========================================================


```r
sl_lasso = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = "SL.glmnet")
sl_lasso
```

```

Call:  
SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = "SL.glmnet") 

                   Risk Coef
SL.glmnet_All 0.1512258    1
```

```r
pred_lasso = predict(sl_lasso, X_holdout)
ROC_eval("lasso", pred_lasso$pred, Y_holdout)
```

<img src="ProjectPresentation-figure/fitLasso-1.png" title="plot of chunk fitLasso" alt="plot of chunk fitLasso" style="display: block; margin: auto;" />

Fit Ridge Regression
========================================================


```r
set.seed(904)
learners    = create.Learner("SL.glmnet", params = list(alpha = 0))
learners$names
```

```
[1] "SL.glmnet_1"
```

```r
sl_ridge = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = learners$names)
pred_ridge = predict(sl_ridge, X_holdout)
ROC_eval("ridge", pred_ridge$pred, Y_holdout)
```

<img src="ProjectPresentation-figure/fitRidge-1.png" title="plot of chunk fitRidge" alt="plot of chunk fitRidge" style="display: block; margin: auto;" />

Fit Ranger
========================================================


```r
library(ranger)
set.seed(904)
sl_ranger = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = "SL.ranger")
sl_ranger
```

```

Call:  
SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = "SL.ranger") 

                   Risk Coef
SL.ranger_All 0.1549322    1
```

```r
pred_ranger = predict(sl_ranger, X_holdout)
ROC_eval("ranger", pred_ranger$pred, Y_holdout)
```

<img src="ProjectPresentation-figure/fitRanger-1.png" title="plot of chunk fitRanger" alt="plot of chunk fitRanger" style="display: block; margin: auto;" />

Fit Stacked Model
========================================================

```r
SL.library <- c("SL.glm", "SL.gam", "SL.glmnet", "SL.glmnet_1", "SL.ranger", "SL.ksvm", "SL.mean")
set.seed(904)
sl_stacked = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = SL.library, 
                          verbose = F, cvControl = list(V = 5), method = "method.NNLS")
sl_stacked
```

```

Call:  
SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = SL.library,  
    method = "method.NNLS", verbose = F, cvControl = list(V = 5)) 

                     Risk      Coef
SL.glm_All      0.1541189 0.1410684
SL.gam_All      0.1532209 0.3096086
SL.glmnet_All   0.1550442 0.0000000
SL.glmnet_1_All 0.1532624 0.2815630
SL.ranger_All   0.1591219 0.2677600
SL.ksvm_All     0.1664940 0.0000000
SL.mean_All     0.2338019 0.0000000
```

Cross Validation Stacked Model
========================================================

```r
set.seed(904)
cv_stacked <- CV.SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = SL.library,
                              verbose = F, cvControl = list(V = 5), method = "method.NNLS")
summary(cv_stacked)
```

```

Call:  
CV.SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = SL.library,  
    method = "method.NNLS", verbose = F, cvControl = list(V = 5)) 

Risk is based on: Mean Squared Error

All risk estimates are based on V =  5 

       Algorithm     Ave        se     Min     Max
   Super Learner 0.15177 0.0097741 0.13000 0.18402
     Discrete SL 0.15437 0.0105308 0.13306 0.18544
      SL.glm_All 0.15406 0.0107291 0.13255 0.18338
      SL.gam_All 0.15315 0.0106896 0.12697 0.18544
   SL.glmnet_All 0.15457 0.0102180 0.13406 0.18849
 SL.glmnet_1_All 0.15320 0.0094776 0.13306 0.18340
   SL.ranger_All 0.15924 0.0092672 0.13794 0.19448
     SL.ksvm_All 0.16653 0.0103261 0.13918 0.21283
     SL.mean_All 0.23374 0.0061529 0.21603 0.26112
```

Cross Validation Stacked Model
========================================================

```r
plot(cv_stacked, cex.lab = 2) + theme_bw(base_size = 15)
```

<img src="ProjectPresentation-figure/cv_stacked3-1.png" title="plot of chunk cv_stacked3" alt="plot of chunk cv_stacked3" style="display: block; margin: auto;" />

Evaluate Test Set
========================================================
<img src="ProjectPresentation-figure/pred.eval-1.png" title="plot of chunk pred.eval" alt="plot of chunk pred.eval" style="display: block; margin: auto;" />

Adjust Ranger Function
========================================================

```r
SL.ranger
```

```
function (Y, X, newX, family, obsWeights, num.trees = 500, mtry = floor(sqrt(ncol(X))), 
    write.forest = TRUE, probability = family$family == "binomial", 
    min.node.size = ifelse(family$family == "gaussian", 5, 1), 
    replace = TRUE, sample.fraction = ifelse(replace, 1, 0.632), 
    num.threads = 1, verbose = T, ...) 
{
    .SL.require("ranger")
    if (family$family == "binomial") {
        Y = as.factor(Y)
    }
    if (is.matrix(X)) {
        X = data.frame(X)
    }
    fit <- ranger::ranger(`_Y` ~ ., data = cbind(`_Y` = Y, X), 
        num.trees = num.trees, mtry = mtry, min.node.size = min.node.size, 
        replace = replace, sample.fraction = sample.fraction, 
        case.weights = obsWeights, write.forest = write.forest, 
        probability = probability, num.threads = num.threads, 
        verbose = verbose)
    pred <- predict(fit, data = newX)$predictions
    if (family$family == "binomial") {
        pred = pred[, "1"]
    }
    fit <- list(object = fit, verbose = verbose)
    class(fit) <- c("SL.ranger")
    out <- list(pred = pred, fit = fit)
    return(out)
}
<bytecode: 0x0000000020e565a0>
<environment: namespace:SuperLearner>
```

Adjust Ranger Function
========================================================

```r
mtry_seq    <- seq(2, 5, 1)
n.trees_seq <- seq(500, 800, 100) 
learners.ranger <- create.Learner("SL.ranger", tune = list(mtry = mtry_seq, num.trees = n.trees_seq))
learners.ranger$names
```

```
 [1] "SL.ranger_1"  "SL.ranger_2"  "SL.ranger_3"  "SL.ranger_4" 
 [5] "SL.ranger_5"  "SL.ranger_6"  "SL.ranger_7"  "SL.ranger_8" 
 [9] "SL.ranger_9"  "SL.ranger_10" "SL.ranger_11" "SL.ranger_12"
[13] "SL.ranger_13" "SL.ranger_14" "SL.ranger_15" "SL.ranger_16"
```

```r
set.seed(904)
cv_stacked.ranger <- CV.SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = learners.ranger$names,verbose = F, cvControl = list(V = 5), method = "method.NNLS")
```

Adjust Ranger Function
========================================================

```r
plot(cv_stacked.ranger) + theme_bw()
SL.ranger_5
```

```
function (...) 
SL.ranger(..., mtry = 2, num.trees = 600)
<bytecode: 0x0000000024f08b90>
```

<img src="ProjectPresentation-figure/Adjust5-1.png" title="plot of chunk Adjust5" alt="plot of chunk Adjust5" style="display: block; margin: auto;" />

Revised Stacked Model
========================================================



```r
SL.gam_12
```

```
function (...) 
SL.gam(..., deg.gam = 4, cts.num = 3)
```

```r
SL.glmnet_1
```

```
function (...) 
SL.glmnet(..., alpha = 0)
```

```r
SL.ranger_5
```

```
function (...) 
SL.ranger(..., mtry = 2, num.trees = 600)
<bytecode: 0x0000000024f08b90>
```

```r
SL.library2 <- c("SL.glm", "SL.gam_12", "SL.glmnet_1", "SL.ranger_5", "SL.mean")
set.seed(904)
cv_stacked2 <- CV.SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = SL.library2,
                               verbose = F, cvControl = list(V = 5), method = "method.NNLS")
```

Revised Stacked Model
========================================================

```
                mean(weight)         sd        min       max
SL.glmnet_1_All    0.3402132 0.20271323 0.00000000 0.4964136
SL.gam_12_All      0.2979238 0.13611385 0.13996798 0.4445352
SL.ranger_5_All    0.1857092 0.07049154 0.07854268 0.2665017
SL.glm_All         0.1761539 0.26247687 0.00000000 0.5935303
SL.mean_All        0.0000000 0.00000000 0.00000000 0.0000000
```

<img src="ProjectPresentation-figure/Adjust33-1.png" title="plot of chunk Adjust33" alt="plot of chunk Adjust33" style="display: block; margin: auto;" />

Evaluate Revised Stacked Model
========================================================
<img src="ProjectPresentation-figure/Adjust4-1.png" title="plot of chunk Adjust4" alt="plot of chunk Adjust4" style="display: block; margin: auto;" />

Reflection
========================================================
- Time constraint and computing power
- No capability to look into the components of the models through the super learner package
- The optimization method for AUC did not work correctly

Overall, due to a statistician's toolbox having so many potential models in it, I found super learner to be a simple way to compare all the models.

Resources
========================================================


```
[1] A. I. Naimi and L. B. Balzer. "Stacked generalization: an
introduction to super learning". In: _European Journal of
Epidemiology_ 33.5 (May. 2018), pp. 459-464. ISSN: 1573-7284. DOI:
10.1007/s10654-018-0390-z. <URL:
https://doi.org/10.1007/s10654-018-0390-z>.

[2] E. Polley, E. LeDell, C. Kennedy, et al. _SuperLearner: Super
Learner Prediction_. R package version 2.0-24. 2018. <URL:
https://CRAN.R-project.org/package=SuperLearner>.

[3] J. W. Smith, J. E. Everhart, W. C. Dickson, et al. "Using the
ADAP learning algorithm to forecast the onset of diabetes
mellitus". In: _Proceedings of the Annual Symposium on Computer
Application in Medical Care_. Ed. by NA. American Medical
Informatics Association, 1988, pp. 261-265.

[4] O. Steinki and Z. Mohammad. "Introduction to ensemble
learning". In: _European Journal of Epidemiology_ (Aug. 2015).
DOI: 10.2139/ssrn.2634092. <URL:
http://dx.doi.org/10.2139/ssrn.2634092>.

[5] S. B. Vardeman. "Lecture Notes on Modern Multivariate
Statistical Learning-Version II". In: _European Journal of
Epidemiology_ (Jul. 2018), pp. 109-112. <URL:
http://www.analyticsiowa.com/wp-content/uploads/2018/07/StatLearningNotesII.pdf>.
```
