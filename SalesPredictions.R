# Title: C2T3 Sales Predictions

# Last update: 2019.01.15

# File: SalesPredictions.R
# Project name: C2T3 Predicting sales of four different product types


###############
# Project Notes
###############

# Summarize project: 
# Predict the sales of four different product types 
# Assessing the effects service and customer reviews have on sales.

# Summarize top model and/or filtered dataset
# The top model was rfFit2 used with readyDataFS.


###############
# Housekeeping
###############

# Clear objects if necessary
#rm(list = ls())

# get working directory
getwd()
?getwd  # get help
# set working directory
setwd("/Users/muift/documents/R_Projects/C2T3")
dir()


################
# Load packages
################


library(caret)
library(corrplot)
library(C50)
library(doParallel)
library(mlbench)
library(readr)
library(parallel)
library(plyr)
#library(knitr)



#####################
# Parallel processing
#####################

# detect number of cores
detectCores()  
# select number of cores
cl <- makeCluster(2)  
# register cluster
registerDoParallel(cl) 
# confirm number of cores being used by RStudio
getDoParWorkers()  
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)


###############
# Import data
##############

#--- Load raw datasets ---#

## Load Train/Existing data (Dataset 1)
existingProduct <- read.csv("existingProduct2017.csv", stringsAsFactors = FALSE, header=T)
class(existingProduct)  # "data.frame"
str(existingProduct)


## Load Predict/New data (Dataset 2) ---#
newProduct <- read.csv("newproduct2017.csv", stringsAsFactors = FALSE, header=T)


################
# Evaluate data
################

#--- Dataset 1 ---#
str(existingProduct)  # 80 obs. of  18 variables 
names(existingProduct)
head(existingProduct)
tail(existingProduct)
summary(existingProduct)

# plot
hist(existingProduct$Volume)
plot(existingProduct$ProductType, existingProduct$Volume)
qqnorm(existingProduct$Volume) # Be familiar with this plot, but don't spend a lot of time on it

# check for missing values 
anyNA(existingProduct)
is.na(existingProduct)


#--- Dataset 2 ---#
str(newProduct)  # 24 obs. of 18 variables 

# check for missing values 
anyNA(newProduct) # FALSE


#############
# Preprocess
#############

#--- Dataset 1 ---#

# Dummify the data
dumexProdwoID <- dummyVars(" ~ .", data = exProdwoID)
readyData <- data.frame(predict(dumexProdwoID, newdata = exProdwoID))
str(readyData)
summary(readyData)
anyNA(readyData)
is.na(readyData)

# Delete any attribute that has missing information
readyData$BestSellersRank <- NULL


#--- Dataset 2 ---#

# If there is a dataset with unseen data to make predictions on, then preprocess here
# to make sure that it is preprossed the same as the training dataset.

# Dummify the data
dumNewProdwoID <- dummyVars(" ~ .", data = newProdwoID)
newReadyData <- data.frame(predict(dumNewProdwoID, newdata = newProdwoID))
str(newReadyData) # 24 obs. of  28 variables:
summary(readyData)
anyNA(readyData)
is.na(readyData)

# Delete any attribute that has missing information
# Even though the new data does not have missing data in BestSellerRank, it is removed to match Dataset 1
newReadyData$BestSellersRank <- NULL


#################
# Feature removal
#################

#--- Dataset 1 ---#

# remove ID and obvious features
exProdwoID <- existingProduct
exProdwoID$ProductNum <- NULL   # remove ID
str(exProdwoID) # 'data.frame':	80 obs. of  17 variables:


# remove based on Feature Selection (FS)
# 6 attributes removed
readyDataFS <- readyData
readyDataFS$x1StarReviews <- NULL
readyDataFS$x3StarReviews <- NULL
#readyDataFS$x4StarReviews <- NULL
readyDataFS$x5StarReviews <- NULL
#readyDataFS$NegativeServiceReview <- NULL
#readyDataFS$ProductTypeExtendedWarranty <- NULL

str(readyDataFS)

#--- Dataset 2 ---#

# remove ID and obvious features
newProdwoID <- newProduct
newProdwoID$ProductNum <- NULL   # remove ID
str(newProdwoID) # 'data.frame':	24 obs. of  17 variables:


# remove based on Feature Selection (FS)
# 6 attributes removed
newReadyDataFS <- newReadyData
newReadyDataFS$x2StarReviews <- NULL
newReadyDataFS$x3StarReviews <- NULL
newReadyDataFS$x4StarReviews <- NULL
newReadyDataFS$x5StarReviews <- NULL
newReadyDataFS$NegativeServiceReview <- NULL
newReadyDataFS$ProductTypeExtendedWarranty <- NULL

str(newReadyDataFS) # 24 obs. of  21 variables:

###############
# Save datasets
###############

# after ALL preprocessing, save a new version of the dataset
# Dataset 1
write.csv(readyData, file="readyData.csv")
write.csv(readyDataFS, file="readyDataFS.csv")

# Dataset 2
write.csv(newReadyData, file="newReadyData.csv")
write.csv(newReadyDataFS, file="newReadyDataFS.csv")

# OPEN DATASET 1
read.csv("readyData.csv")
read.csv("readyDataFS.csv")

# OPEN DATASET 2
read.csv("newReadyData.csv")
read.csv("newReadyDataFS.csv")

################
# Sampling
################

# ---- Sampling ---- #

# Sampling not required for small dataset


##########################
# Feature Selection (FS)
##########################

# Three primary methods
# 1. Filtering
# 2. Wrapper methods (e.g., RFE caret)
# 3. Embedded methods (e.g., varImp)

###########
# Filtering
###########

# good for num/int data 

# calculate correlation matrix for all vars
corrData <- cor(readyData[,1:27])
corrDataFS <- cor(readyDataFS[,1:24])

# summarize the correlation matrix
corrData
corrDataFS

# plot correlation matrix (general, no modifications done)
corrplot(corrData)
corrplot(corrDataFS)

# order = "hclust" - sorts based on level of collinearity
corrplot(corrData, method = "square", order = "hclust")
corrplot(corrDataFS, method = "square", order = "hclust")

# find IVs that are highly corrected (ideally > 0.80)
corrIV <- cor(readyData[,1:27])
corrIVFS <- cor(readyDataFS[,1:24])

# summarize the correlation matrix
corrIV
corrIVFS

# create object with indexes of highly corr features, and get column names
corrIVhigh <- findCorrelation(corrIV, cutoff=0.8, names = TRUE)


# confirm there are no more highly correlated features left
corrIVhighFS <- findCorrelation(corrIVFS, cutoff=0.8, names = TRUE)

# print indexes of highly correlated attributes
corrIVhigh
#[1] "x3StarReviews"               "x4StarReviews"               "x5StarReviews"               "x2StarReviews"              
#[5] "NegativeServiceReview"       "ProductTypeExtendedWarranty"

corrIVhighFS

############
# caret RFE (recursive feature elimination)
############

# lmFuncs - linear model
# rfFuncs - random forests
# nbFuncs - naive Bayes
# treebagFuncs - bagged trees


## ---- lm ---- ##

# define refControl using a linear model selection function (regression only)
LMcontrol <- rfeControl(functions=lmFuncs, method="cv", number=10)
# run the RFE algorithm
set.seed(7)
LMresults <- rfe(readyData[,1:26], readyData[,27], sizes=c(1:26), rfeControl=LMcontrol)
LMresults

# readyDataFS performance metrics
#Recursive feature selection
#Outer resampling method: Cross-Validated (10 fold) 
#Resampling performance over subset size:
  
#Variables      RMSE Rsquared       MAE    RMSESD RsquaredSD     MAESD Selected
#        1 2.659e-13        1 2.128e-13 2.954e-13          0 2.005e-13        *
#        2 2.707e-13        1 2.157e-13 2.943e-13          0 2.000e-13         
#        3 2.810e-13        1 2.134e-13 2.864e-13          0 1.850e-13         
#        4 2.894e-13        1 2.177e-13 2.745e-13          0 1.705e-13         
#        5 3.202e-13        1 2.349e-13 2.911e-13          0 1.794e-13         
#        6 3.228e-13        1 2.396e-13 2.885e-13          0 1.789e-13         
#        7 3.248e-13        1 2.325e-13 2.809e-13          0 1.706e-13         
#        8 3.652e-13        1 2.569e-13 2.830e-13          0 1.730e-13         
#        9 3.705e-13        1 2.556e-13 2.732e-13          0 1.655e-13         
#        10 4.131e-13        1 2.972e-13 3.111e-13          0 2.222e-13         
#        11 3.775e-13        1 2.833e-13 2.574e-13          0 2.045e-13         
#        12 4.281e-13        1 3.215e-13 3.867e-13          0 3.226e-13         
#        13 4.138e-13        1 3.262e-13 3.742e-13          0 3.286e-13         
#        14 4.419e-13        1 3.326e-13 3.905e-13          0 3.145e-13         
#        15 4.694e-13        1 3.501e-13 3.829e-13          0 2.852e-13         
#        16 5.416e-13        1 3.799e-13 4.702e-13          0 3.188e-13         
#        17 5.853e-13        1 4.210e-13 4.854e-13          0 3.417e-13         
#        18 4.813e-13        1 3.522e-13 4.319e-13          0 2.909e-13         
#        19 4.633e-13        1 3.573e-13 3.604e-13          0 2.884e-13         
#        20 4.729e-13        1 3.768e-13 3.572e-13          0 3.014e-13         
#        21 4.719e-13        1 3.708e-13 4.278e-13          0 3.463e-13         
#        22 5.947e-13        1 4.281e-13 4.518e-13          0 2.954e-13         
#        23 6.059e-13        1 4.452e-13 4.210e-13          0 2.962e-13         
#        24 6.244e-13        1 4.321e-13 4.118e-13          0 2.429e-13         
#        25 5.877e-13        1 4.024e-13 4.925e-13          0 2.704e-13         
#        26 6.720e-13        1 4.333e-13 5.452e-13          0 2.679e-13         

#The top 1 variables (out of 1):
#  x5StarReviews        

# Notes of performance metrics
# Rsquared is generally very low and RMSE is high
# Model did not perform well, it only deemed 5 variables as important


plot(LMresults, type=c("g", "o"))  

# show predictors used
predictors(LMresults) 
#[1] "x5StarReviews"

varImp(LMresults)
# Note results.
#                       Overall
#x5StarReviews                4


## ---- rf ---- ##

# define the control using a random forest selection function (regression or classification)
RFcontrol <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
set.seed(7)
RFresults <- rfe(readyData[,1:26], readyData[,27], sizes=c(1:26), rfeControl=RFcontrol)
RFresults

#Recursive feature selection
#Outer resampling method: Cross-Validated (10 fold) 
#Resampling performance over subset size:
  
#  Variables  RMSE Rsquared   MAE RMSESD RsquaredSD MAESD Selected
#          1 282.7   0.9791 114.0  659.3    0.05268 243.1        *
#          2 390.4   0.9772 158.1  742.7    0.02242 269.0         
#          3 409.8   0.9562 175.6  733.6    0.06543 296.8         
#          4 511.6   0.9332 218.9  777.3    0.07797 314.0         
#          5 553.8   0.9232 237.7  776.2    0.08193 317.1         
#          6 537.9   0.9319 227.3  762.1    0.07482 308.4         
#          7 599.0   0.9190 247.4  819.0    0.07842 316.3         
#          8 601.6   0.9063 256.8  804.8    0.08370 314.2         
#          9 559.1   0.9225 235.3  783.6    0.07489 307.5         
#          10 575.1   0.9142 245.2  787.5    0.07756 307.8         
#          11 593.8   0.9038 256.7  754.3    0.07447 296.3         
#          12 571.3   0.9240 242.2  783.6    0.07220 307.6         
#          13 577.9   0.9182 245.9  780.4    0.07076 301.5         
#          14 588.4   0.9155 249.4  779.3    0.06709 296.7         
#          15 550.1   0.9226 236.6  751.5    0.07630 302.4         
#          16 585.1   0.9157 248.2  789.0    0.07243 305.9         
#          17 600.5   0.9081 254.7  790.8    0.06941 300.8         
#          18 582.5   0.9264 245.6  795.4    0.06935 304.5         
#          19 579.6   0.9227 247.4  787.9    0.07669 308.3         
#          20 579.0   0.9168 248.5  785.4    0.08326 308.7         
#          21 554.3   0.9279 237.4  760.3    0.07270 300.5         
#          22 565.4   0.9250 241.3  764.3    0.06953 295.3         
#          23 582.8   0.9185 250.6  793.9    0.07780 309.6         
#          24 567.4   0.9253 244.5  770.3    0.07622 303.8         
#          25 573.3   0.9244 243.9  778.3    0.07792 303.6         
#          26 587.5   0.9212 251.4  771.9    0.07134 298.9         

#The top 1 variables (out of 1):
#  x5StarReviews

# plot the results
plot(RFresults, type=c("g", "o"))

# show predictors used
predictors(RFresults)
#[1] "x5StarReviews" 

varImp(RFresults)
# Note results.
#                             Overall
#x5StarReviews                13.70815
#PositiveServiceReview        11.58929


## ---- treebag ---- ##
# treebagFuncs - bagged trees
TBControl <- rfeControl(functions=treebagFuncs, method="cv", number=10)
# run the RFE algorithm
set.seed(7)
TBresults <- rfe(readyData[,1:26], readyData[,27], sizes=c(1:26), rfeControl=TBControl)
TBresults
#Recursive feature selection
#Outer resampling method: Cross-Validated (10 fold) 
#Resampling performance over subset size:
  
#  Variables  RMSE Rsquared   MAE RMSESD RsquaredSD MAESD Selected
#          1 732.1   0.9011 321.0  944.2     0.1143 328.7         
#          2 725.1   0.9072 325.4  939.2     0.1122 321.0         
#          3 693.5   0.9106 309.0  938.8     0.1118 327.8         
#          4 705.5   0.9097 313.0  930.9     0.1130 327.5         
#          5 651.4   0.9087 295.0  904.1     0.1139 323.8         
#          6 690.3   0.9145 308.7  918.9     0.1125 331.6         
#          7 766.9   0.8943 334.3  972.5     0.1108 329.0         
#          8 696.6   0.9125 308.4  951.4     0.1140 328.8         
#          9 698.0   0.9050 306.8  942.8     0.1100 335.2         
#          10 737.7   0.9060 325.5  943.3     0.1093 321.6         
#          11 686.6   0.9146 305.5  941.9     0.1135 325.7         
#          12 637.2   0.9222 291.8  921.4     0.1134 330.5        *
#          13 710.9   0.9114 313.3  962.4     0.1129 331.4         
#          14 696.4   0.9148 317.4  931.3     0.1136 334.0         
#          15 715.9   0.9003 315.9  949.1     0.1080 325.1         
#          16 709.5   0.9105 314.7  954.4     0.1118 325.5         
#          17 691.6   0.9118 308.4  951.1     0.1144 326.5         
#          18 707.4   0.9120 307.9  943.3     0.1117 325.4         
#          19 692.8   0.9090 312.7  950.5     0.1100 323.9         
#          20 655.6   0.9188 300.7  933.9     0.1133 329.7         
#          21 698.5   0.9050 306.0  949.7     0.1165 331.8         
#          22 660.2   0.9177 295.4  968.7     0.1133 334.3         
#          23 661.3   0.9163 297.9  910.1     0.1135 322.9         
#          24 672.5   0.9197 296.8  967.0     0.1172 332.0         
#          25 679.5   0.9123 306.9  928.0     0.1126 324.7         
#          26 706.8   0.9114 312.2  949.6     0.1145 322.4         

#The top 5 variables (out of 12):
#  x5StarReviews, PositiveServiceReview, x4StarReviews, x3StarReviews, x2StarReviews


# plot the results
plot(TBresults, type=c("g", "o"))

# show predictors used
predictors(TBresults)
#[1] "x5StarReviews"               "PositiveServiceReview"       "x4StarReviews"               "x3StarReviews"              
#[5] "x2StarReviews"               "x1StarReviews"               "NegativeServiceReview"       "ProductDepth"               
#[9] "ProductWidth"                "ShippingWeight"              "ProfitMargin"                "ProductTypeExtendedWarranty"

varImp(TBresults)
#                                Overall
#x5StarReviews               1.49854736
#PositiveServiceReview       1.15701991
#x4StarReviews               1.14809720
#x3StarReviews               0.73102849
#x2StarReviews               0.61331556
#x1StarReviews               0.59133407
#NegativeServiceReview       0.26182436
#ProductDepth                0.05178576
#ProductWidth                0.05156579
#ProfitMargin                0.04069718
#ShippingWeight              0.03492338
#ProductTypeExtendedWarranty 0.03007501
#ProductHeight               0.02660633
#Recommendproduct            0.01961117
#Price                       0.01907905
#ProductTypeAccessories      0.00000000


##############################
# Variable Importance (varImp)
##############################

# varImp is evaluated in the model train/fit section


# ---- Conclusion ---- #



##################
# TRAIN / TEST SETS
##################

# CREATE THE TRAINING PARTITION THAT IS 75% OF TOTAL OBS (WITHOUT feature removal)
set.seed(123) # set random seed
inTraining <- createDataPartition(readyData$Volume, 
                                    p = 0.75, 
                                    list = FALSE)

# CREATE THE TRAINING PARTITION THAT IS 75% OF TOTAL OBS (WITH feature removal)
set.seed(123) # set random seed
inTrainingFS <- createDataPartition(readyDataFS$Volume, 
                                  p = 0.75, 
                                  list = FALSE)
str(inTraining)
#int [1:61, 1] 1 4 5 6 8 11 12 13 14 15 ...
#- attr(*, "dimnames")=List of 2
#..$ : NULL
#..$ : chr "Resample1"

# CREATE TRAINING/TESTING DATASET (WITHOUT feature removal)
trainSet <- readyData[inTraining,]  
testSet <- readyData[-inTraining,]  

# CREATE TRAINING/TESTING DATASET (WITH feature removal)
trainSetFS <- readyDataFS[inTraining,]  
testSetFS <- readyDataFS[-inTraining,] 

# VERIFY NUMBER OF OBSERVATIONS
str(trainSet) # 'data.frame':	61 obs. of  27 variables:
str(testSet) # 'data.frame':	19 obs. of  27 variables:


################
# Train control
################

# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)


##############
# Train model
##############

## ------- LM ------- ##

# LM TRAIN/FIT without feature selection
# AUTOMATIC GRID
set.seed(123)
lmFit <- train(Volume~., 
               data = trainSet, 
               method = "lm",
               trControl = fitControl)
lmFit
#Resampling results:
#RMSE          Rsquared  MAE         
#3.713545e-13  1         2.308183e-13


# LM TRAIN/FIT with feature selection
# AUTOMATIC GRID
set.seed(123)
lmFitFS <- train(Volume~., 
               data = trainSetFS, 
               method = "lm",
               trControl = fitControl)
lmFitFS
#RMSE      Rsquared   MAE     
#1136.494  0.5634613  661.8363


## ------- RF ------- ##

# RF train/fit
set.seed(123)
# tuneLength=1 (without feature selection)
rfFit1 <- train(Volume~.,data=trainSet,method="rf",importance=T,trControl=fitControl,tuneLength=1) 
rfFit1
#RMSE      Rsquared   MAE     
#181.2988  0.9505659  115.5905

set.seed(123)
# tuneLength=1 (with feature selection)
rfFit1FS <- train(Volume~.,data=trainSetFS,method="rf",importance=T,trControl=fitControl,tuneLength=1) 
rfFit1FS

#RMSE      Rsquared   MAE     
#312.8753  0.8726878  230.8864

# tuneLength=2 (without feature selection)
set.seed(123)
rfFit2 <- train(Volume~.,data=trainSet,method="rf",importance=T,trControl=fitControl,tuneLength=2) 
rfFit2
#mtry  RMSE       Rsquared   MAE      
#2    259.25017  0.9172593  185.17864
#26     89.90263  0.9885616   50.83148


# tuneLength=2 (with feature selection)
set.seed(123)
rfFit2FS <- train(Volume~.,data=trainSetFS,method="rf",importance=T,trControl=fitControl,tuneLength=2) 
rfFit2FS
#mtry  RMSE      Rsquared   MAE     
#2    374.7817  0.8223467  293.3071
#20    189.2979  0.9211732  112.9112

# Manual grid (without feature selection)
set.seed(123)
rfGrid <- expand.grid(mtry=c(1,3,5,7,9,11,13,15))
rfFitMan <- train(Volume~.,data=trainSet,method="rf",importance=T,trControl=fitControl,tuneGrid=rfGrid) 
rfFitMan
#mtry  RMSE      Rsquared   MAE      
#1    357.7172  0.8932975  287.07063
#3    223.4830  0.9289880  148.56637
#5    192.6026  0.9449934  121.30515
#7    158.3290  0.9619529   97.56700
#9    147.9000  0.9632303   88.04071
#11    131.9378  0.9725871   75.65066
#13    117.7958  0.9790573   67.46027
#15    109.2231  0.9815651   61.85379

#RMSE was used to select the optimal model using the smallest value.
#The final value used for the model was mtry = 15.

# Manual grid (with feature selection)
set.seed(123)
rfFitManFS <- train(Volume~.,data=trainSetFS,method="rf",importance=T,trControl=fitControl,tuneGrid=rfGrid) 
rfFitManFS
#mtry  RMSE      Rsquared   MAE     
#1    463.4091  0.7292317  380.1896
#3    330.9845  0.8626401  250.6270
#5    288.6174  0.8918045  206.6574
#7    254.3569  0.9079133  177.5250
#9    235.9491  0.9077069  159.7716
#11    223.8666  0.9110111  147.6423
#13    211.5629  0.9151475  134.2072
#15    203.8410  0.9173931  125.1126

#RMSE was used to select the optimal model using the smallest value.
#The final value used for the model was mtry = 15.

# Random (without feature selection)
RFfitControl <- trainControl(method="repeatedcv",number=10,repeats=1,search='random')
set.seed(123)
rfFitRan <- train(Volume~.,data=trainSet,method="rf",importance=T,trControl=RFfitControl) 
rfFitRan
#mtry  RMSE       Rsquared   MAE     
#11    126.82709  0.9740230  72.99082
#21     99.42553  0.9839309  56.40105
#23     90.98784  0.9878311  51.86023

#RMSE was used to select the optimal model using the smallest value.
#The final value used for the model was mtry = 23.

# Random (with feature selection)
set.seed(123)
rfFitRanFS <- train(Volume~.,data=trainSetFS,method="rf",importance=T,trControl=RFfitControl) 
rfFitRanFS
#mtry  RMSE      Rsquared   MAE     
#9    242.6268  0.9031792  163.1948
#16    203.6107  0.9168468  125.2824
#18    192.7652  0.9200915  115.8693

#RMSE was used to select the optimal model using the smallest value.
#The final value used for the model was mtry = 18.

# manual grid (without feature selection)
rfGrid2 <- expand.grid(mtry=c(6)) 
set.seed(123)
rfFitMan6 <- train(Volume~.,data=trainSet,method="rf",importance=T,trControl=fitControl,tuneGrid=rfGrid) 
rfFitMan6
#mtry  RMSE      Rsquared   MAE      
#1    357.7172  0.8932975  287.07063
#3    223.4830  0.9289880  148.56637
#5    192.6026  0.9449934  121.30515
#7    158.3290  0.9619529   97.56700
#9    147.9000  0.9632303   88.04071
#11    131.9378  0.9725871   75.65066
#13    117.7958  0.9790573   67.46027
#15    109.2231  0.9815651   61.85379

#RMSE was used to select the optimal model using the smallest value.
#The final value used for the model was mtry = 15.

# manual grid (with feature selection)
set.seed(123)
rfFitMan6FS <- train(Volume~.,data=trainSetFS,method="rf",importance=T,trControl=fitControl,tuneGrid=rfGrid) 
rfFitMan6FS
#mtry  RMSE      Rsquared   MAE     
#1    463.4091  0.7292317  380.1896
#3    330.9845  0.8626401  250.6270
#5    288.6174  0.8918045  206.6574
#7    254.3569  0.9079133  177.5250
#9    235.9491  0.9077069  159.7716
#11    223.8666  0.9110111  147.6423
#13    211.5629  0.9151475  134.2072
#15    203.8410  0.9173931  125.1126

#RMSE was used to select the optimal model using the smallest value.
#The final value used for the model was mtry = 15.

## ------- SVM ------- ##

# SVM Linear (without feature selection)
set.seed(123)
svmLinear <- train(Volume~.,
                 data = trainSet,
                 method = "svmLinear",
                 trControl = fitControl,
                 tuneLength = 2)
svmLinear
#RMSE      Rsquared   MAE     
#50.33429  0.9952437  42.64277

# SVM Linear (with feature selection)
set.seed(123)
svmLinearFS <- train(Volume~.,
                   data = trainSetFS,
                   method = "svmLinear",
                   trControl = fitControl,
                   tuneLength = 2)
svmLinearFS
#RMSE      Rsquared   MAE    
#1108.364  0.5575805  623.346

# SVM Linear2 (without feature selection)
set.seed(123)
svmLinear2 <- train(Volume~.,
                   data = trainSet,
                   method = "svmLinear2",
                   trControl = fitControl,
                   tuneLength = 2)
svmLinear2
#cost  RMSE      Rsquared   MAE     
#0.25  55.49837  0.9933451  47.12097
#0.50  50.33429  0.9952437  42.64277

# SVM Linear2 (with feature selection)
set.seed(123)
svmLinear2FS <- train(Volume~.,
                    data = trainSetFS,
                    method = "svmLinear2",
                    trControl = fitControl,
                    tuneLength = 2)
svmLinear2FS
#cost  RMSE      Rsquared   MAE     
#0.25  1080.815  0.5821258  601.7909
#0.50  1088.053  0.5649118  609.6753

# SVM Poly (without feature selection
set.seed(123)
svmPoly <- train(Volume~.,
                    data = trainSet,
                    method = "svmPoly",
                    trControl = fitControl,
                    tuneLength = 2)
svmPoly
#degree  scale  C     RMSE      Rsquared   MAE     
#1       0.001  0.25  612.2878  0.8894868  403.0950
#1       0.001  0.50  558.5050  0.9017223  372.3496
#1       0.010  0.25  446.1805  0.9415718  299.4175
#1       0.010  0.50  441.4376  0.9487644  274.5975
#2       0.001  0.25  559.5978  0.9041588  372.5554
#2       0.001  0.50  499.3348  0.9211880  337.6258
#2       0.010  0.25  628.0405  0.9465099  361.5376
#2       0.010  0.50  331.6573  0.9559680  207.5709

# SVM Poly (with feature selection
set.seed(123)
svmPolyFS <- train(Volume~.,
                 data = trainSetFS,
                 method = "svmPoly",
                 trControl = fitControl,
                 tuneLength = 2)
svmPolyFS
#degree  scale  C     RMSE      Rsquared   MAE     
#1       0.001  0.25  662.3126  0.5034512  436.1991
#1       0.001  0.50  645.8943  0.5234925  424.6315
#1       0.010  0.25  557.7457  0.5881761  384.8503
#1       0.010  0.50  571.4220  0.6090008  396.5248
#2       0.001  0.25  652.8995  0.5137003  434.8512
#2       0.001  0.50  622.3894  0.5397438  421.9173
#2       0.010  0.25  671.1439  0.6322686  444.0129
#2       0.010  0.50  877.1230  0.6221439  522.7092

## ------- GB ------- ##

# GB tree (without feature selection)
set.seed(123)
GBTree <- train(Volume~.,
                 data = trainSet,
                 method = "xgbTree",
                 trControl = fitControl,
                 tuneLength = 2)
GBTree

#61 samples
#26 predictors

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 1 times) 
#Summary of sample sizes: 56, 55, 54, 54, 54, 55, ... 
#Resampling results across tuning parameters:
  
#  eta  max_depth  colsample_bytree  subsample  nrounds  RMSE       Rsquared   MAE     
#  0.4  2          0.8               1.0        100       46.77889  0.9978378  30.37493

#Tuning parameter 'gamma' was held constant at a value of 0
#Tuning parameter 'min_child_weight' was held constant at a value of 1
#RMSE was used to select the optimal model using the smallest value.
#The final values used for the model were nrounds = 100, max_depth = 2, eta = 0.4, gamma = 0, colsample_bytree = 0.8, min_child_weight =
#  1 and subsample = 1.


# GB tree (with feature selection)
set.seed(123)
GBTreeFS <- train(Volume~.,
                data = trainSetFS,
                method = "xgbTree",
                trControl = fitControl,
                tuneLength = 2)
GBTreeFS

# GB DART (without feature selection)
set.seed(123)
GBDart <- train(Volume~.,
                data = trainSet,
                method = "xgbDART",
                trControl = fitControl,
                tuneLength = 2)
GBDart

# GB DART (with feature selection)
set.seed(123)
GBDartFS <- train(Volume~.,
                data = trainSetFS,
                method = "xgbDART",
                trControl = fitControl,
                tuneLength = 2)
GBDartFS


##--- Compare metrics ---##

ModelFitResults <- resamples(list(rf=rfFit1,
                                  rfFS=rfFit1FS,
                                  xgbDART=GBDart,
                                  xgbDARTFS=GBDartFS,
                                  xgbTree=GBTree,
                                  xgbTreeFS=GBTreeFS,
                                  rf1=rfFit1,
                                  rf1FS=rfFit1FS,
                                  rf2=rfFit2,
                                  rf2FS=rfFit2FS,
                                  rfManual=rfFitMan,
                                  rfManualFS=rfFitManFS,
                                  rfManual6=rfFitMan6,
                                  rfManual6FS=rfFitMan6FS,
                                  rfRandom=rfFitRan,
                                  rfRandomFS=rfFitRanFS))

# output summary metrics for tuned models 
summary(ModelFitResults)
#Call:
#  summary.resamples(object = ModelFitResults)

#Models: rf, rfFS, xgbDART, xgbDARTFS, xgbTree, xgbTreeFS, rf1, rf1FS, rf2, rf2FS, rfManual, rfManualFS, rfManual6, rfManual6FS, rfRandom, rfRandomFS 
#Number of resamples: 10 

#MAE 
#                   Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#rf           57.551733  78.28373 122.40338 115.59054 146.64451 173.45528    0
#rfFS        116.112209 176.01731 236.40823 230.88641 264.84983 359.73934    0
#xgbDART       5.003470  18.30412  32.22458  33.14525  47.36214  69.77693    0
#xgbDARTFS    56.563183  93.75512 109.70128 120.67411 116.81932 242.94896    0
#xgbTree       9.455012  16.61173  31.92651  30.37493  37.06448  67.25249    0
#xgbTreeFS    48.609835  93.87625  99.74884 115.22152 115.08503 261.95094    0
#rf1          57.551733  78.28373 122.40338 115.59054 146.64451 173.45528    0
#rf1FS       116.112209 176.01731 236.40823 230.88641 264.84983 359.73934    0
#rf2          10.370622  18.77490  56.22342  50.83148  75.48566  97.10046    0
#rf2FS        32.571173  56.05455 102.78990 112.91118 143.55503 289.77509    0
#rfManual     24.008613  34.17308  56.28768  61.85379  85.86764 117.08377    0
#rfManualFS   40.225680  87.08164 115.95274 125.11258 149.14352 284.38278    0
#rfManual6    24.008613  34.17308  56.28768  61.85379  85.86764 117.08377    0
#rfManual6FS  40.225680  87.08164 115.95274 125.11258 149.14352 284.38278    0
#rfRandom     11.591360  21.73018  48.78398  51.86023  80.07668  92.44552    0
#rfRandomFS   32.362627  68.49605 104.60561 115.86925 144.69758 284.03425    0

#RMSE 
#                   Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#rf           88.073330 118.47317 165.53157 181.29876 246.14182 291.97446    0
#rfFS        134.693998 218.84351 304.74910 312.87533 393.37919 489.08101    0
#xgbDART       6.155032  28.20201  47.27817  48.21513  71.11681  94.18990    0
#xgbDARTFS    88.044916 115.48326 149.14549 160.43132 189.22662 311.11818    0
#xgbTree      13.918882  23.85097  53.55245  46.77889  58.59939  93.56052    0
#xgbTreeFS    56.099145 125.72016 140.34644 157.09587 165.02604 333.07216    0
#rf1          88.073330 118.47317 165.53157 181.29876 246.14182 291.97446    0
#rf1FS       134.693998 218.84351 304.74910 312.87533 393.37919 489.08101    0
#rf2          14.455159  32.01279  95.89154  89.90263 145.85282 151.16410    0
#rf2FS        49.923437  82.85052 206.79264 189.29792 241.05455 433.52798    0
#rfManual     41.502237  53.90912 108.93179 109.22314 166.24392 183.10524    0
#rfManualFS   53.731031 122.19277 209.42210 203.84100 272.75894 425.39615    0
#rfManual6    41.502237  53.90912 108.93179 109.22314 166.24392 183.10524    0
#rfManual6FS  53.731031 122.19277 209.42210 203.84100 272.75894 425.39615    0
#rfRandom     18.638854  36.25846  96.76944  90.98784 137.22021 156.31108    0
#rfRandomFS   46.510272  93.61182 209.06742 192.76521 245.97107 432.11046    0

#Rsquared 
#                 Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#rf          0.8738560 0.9132274 0.9607563 0.9505659 0.9851870 0.9986764    0
#rfFS        0.5428116 0.8400262 0.9117253 0.8726878 0.9603249 0.9902797    0
#xgbDART     0.9860487 0.9955352 0.9970492 0.9961856 0.9981968 0.9999639    0
#xgbDARTFS   0.7797567 0.9061074 0.9621924 0.9298012 0.9813980 0.9905006    0
#xgbTree     0.9917073 0.9977240 0.9987324 0.9978378 0.9992284 0.9997112    0
#xgbTreeFS   0.7708830 0.9413761 0.9603249 0.9397848 0.9701857 0.9936550    0
#rf1         0.8738560 0.9132274 0.9607563 0.9505659 0.9851870 0.9986764    0
#rf1FS       0.5428116 0.8400262 0.9117253 0.8726878 0.9603249 0.9902797    0
#rf2         0.9722860 0.9814139 0.9909816 0.9885616 0.9979760 0.9997841    0
#rf2FS       0.5991814 0.8989386 0.9821740 0.9211732 0.9898909 0.9999413    0
#rfManual    0.9394346 0.9719337 0.9936679 0.9815651 0.9958070 0.9997002    0
#rfManualFS  0.6031754 0.8901652 0.9669993 0.9173931 0.9923575 0.9998940    0
#rfManual6   0.9394346 0.9719337 0.9936679 0.9815651 0.9958070 0.9997002    0
#rfManual6FS 0.6031754 0.8901652 0.9669993 0.9173931 0.9923575 0.9998940    0
#rfRandom    0.9593762 0.9836032 0.9913223 0.9878311 0.9982515 0.9996573    0
#rfRandomFS  0.5970549 0.8951467 0.9722853 0.9200915 0.9914988 0.9998443    0



#--- Save/load top performing model ---#

saveRDS(GBTree, "GBTree.rds")  
# load and name model
GBTree <- readRDS("GBTree.rds")


############################
# Predict testSet/validation
############################


# predict with best model
GBTreePred <- predict(GBTree, testSet)

#performace measurment
postResample(GBTreePred, testSet$Volume)

#     RMSE   Rsquared        MAE 
#2388.34142    0.63141  777.60430


#plot predicted verses actual
plot(GBTreePred,testSet$Volume)

# print predictions
GBTreePred


###############################
# Predict new data (Dataset 2)
###############################

GBTreePred2 <- predict(GBTree, newReadyData)

#Add predictions to the new products data set  
output <- newProduct
output$predictions <- GBTreePred2

# Create a csv file and write it to your hard drive
write.csv(output, file="C2.T3output2.csv", row.names = TRUE)
