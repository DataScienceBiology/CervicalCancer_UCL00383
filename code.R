library(purrr)
library(caret)
library(ggplot2)
library(ggcorrplot)
library(dplyr)
library(xgboost)

# read raw data

rawData <- read.csv("risk_factors_cervical_cancer.csv", header = T)
str(rawData)

# correct the variables' format
cols.num <- c(names(rawData[c(2:4,6,7,9,11,13,26,27,28)]))
rawData[cols.num] <- sapply(rawData[cols.num], as.numeric)

cols.num <- c(names(rawData[29:36]))
rawData[,cols.num] <- lapply(rawData[,cols.num], factor)
str(rawData)

rm(cols.num)

# define missing values as NA
idmiss <- rawData == "?"
is.na(rawData) <- idmiss
rm(idmiss)
str(rawData) # there are factor variables with only one level plus NAs. These can be deleted
rawData$STDs.cervical.condylomatosis <- NULL
rawData$STDs.AIDS <- NULL

# factor variables with NA having 3 levels (? as a level), which should be collapsed to two levels
factor_index <- which(sapply(rawData[1:23], is.factor))
name_index <- names(rawData[factor_index])

rawData[name_index] <- lapply(rawData[name_index], factor)
str(rawData)
rm(factor_index, name_index)

# check for missing values


data.frame(map(rawData, ~sum(is.na(.)))) # 14 variables including NAs

# plotting the missing values
library(VIM)
miss_plot <- aggr(rawData, col = c("navyblue", "yellow"),
                  numbers = TRUE, sortVars = TRUE,
                  labels = names(rawData),
                  cex.axis = 0.7, gap = 3,
                  ylab = c("Missing Data", "Pattern"))

# before imputing let's check for suspected identical variables (Dx.Cancer and Dx)
# identical(rawData[["Dx.Cancer"]], rawData[["Dx"]]) # they are not the same

# imputing missing values

library(missForest)

# imput using missForest
library(doParallel)
set.seed(145)

registerDoParallel(cores = 3)
impute_forest <- missForest(data.frame(rawData),maxiter = 10 ,ntree = 100, parallelize = "forests")
stopImplicitCluster()

# check imputation error
impute_forest$OOBerror # the imputation is done for categorical variables with 0.2% error which is well acceptable
imputed_data <- impute_forest$ximp

save(imputed_data, rawData, file = "workingData.RData")


# exploratory analysis
summary(imputed_data) # there are several variables with near to zero variation and could be deleted. We first check the correlation between variables the decide 
# to delete them.


corr <- round(cor(sapply(imputed_data, as.numeric)),1)
ggcorrplot(corr, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("tomato2", "white", "springgreen3"), 
           title="Correlogram of Cervical Cancer Diagnosis", 
           ggtheme=theme_bw)
# Diagnosis of cancer is highly correlated with HPV. There are variables with no correlation to diagnosed cancer patients.

P <- ggplot(imputed_data)
P <- P + geom_boxplot(mapping = aes(x = Dx.Cancer, y = Age, fill = Dx.HPV))
P <- P + labs(title = "Cancer Diagnosis based on Age", subtitle = "From Cervical Cancer Dataset")
P

# let's bring Dx.Cancer to the start of the data frame
imputed_data <- imputed_data %>% select(Dx.Cancer, everything())

# Splitting to training and validation
set.seed(331)

intrain <- createDataPartition(y = imputed_data[,1], p = 0.8)[[1]]
training <- imputed_data[intrain,]
validation <- imputed_data[-intrain,]

rm(intrain)



# xgbTree modeling
registerDoParallel(cores = 3)

set.seed(171)

# 5-fold classification
trctrl <- trainControl(method = "cv", number = 5)

# train elastic net model
gbFit <- train(Dx.Cancer ~., data = imputed_data,
               method = "xgbTree",
               trControl = trctrl,
               # other parameters
               tuneGrid = data.frame(nrounds = 200, eta = c(0.05, 0.1, 0.3),
                                     max_depth = 4, gamma = 0, colsample_bytree = 1, subsample = 0.5,
                                     min_child_weight = 1))

stopImplicitCluster()


# best parameters by cross validation accuracy
gbFit$bestTune

plot(varImp(gbFit), top = 10)

class.res <- predict(gbFit, validation[,-1])
confusionMatrix(validation[,1], class.res)$overall[1]

