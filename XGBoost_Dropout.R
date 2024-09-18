# Libraries
library(tidyverse)
library(caret)
library(car)
library(skimr)
#install.packages("DataExplorer")
library(DataExplorer)
#install.packages("ROSE")
library(ROSE)


# Step 0: EDA
dropout_data <- read_csv("~/Downloads/student dropout.csv")


summaryStats <- skim(dropout_data)
summaryStats

create_report(dropout_data, y = "Dropped_Out")

table(dropout_data$Dropped_Out)


#Step 1 - Partition our Data and Pre-processing

# change response and categorical variables to factor
dropout_data <- dropout_data %>% 
  mutate_at(c("Dropped_Out", "School", "Gender", "Address",
              "Family_Size", "Parental_Status", "Mother_Job", "Father_Job",
              "Reason_for_Choosing_School", "Guardian", "School_Support",
              "Family_Support", "Extra_Paid_Class", "Extra_Curricular_Activities",
              "Attended_Nursery", "Wants_Higher_Education", "Internet_Access",
              "In_Relationship"
  ), 
  as.factor)

#2. rename resonse 
dropout_data$Dropped_Out<-fct_recode(dropout_data$Dropped_Out, "1" = "TRUE", "0" = "FALSE")

#3. relevel response
dropout_data$Dropped_Out<- relevel(dropout_data$Dropped_Out, ref = "1")

#make sure levels are correct
levels(dropout_data$Dropped_Out)

dropout_predictors_dummy <- model.matrix(Dropped_Out~ ., data = dropout_data)#create dummy variables expect for the response
dropout_predictors_dummy<- data.frame(dropout_predictors_dummy[,-1]) #get rid of intercept
dropout_data <- cbind(dropout_predictors_dummy, Dropped_Out = dropout_data$Dropped_Out)

#index
# Train/Test split before SMOTE
set.seed(99) #set random seed
index <- createDataPartition(dropout_data$Dropped_Out, p = .8,list = FALSE)
dropout_train <- dropout_data[index,]
dropout_test <- dropout_data[-index,]

# SMOTE
# Apply SMOTE to balance the classes
dropout_train_smote <- ovun.sample(Dropped_Out ~ ., data = dropout_train, method = "over", p = 0.5, seed = 1234)$data

# Check the class distribution after applying SMOTE
table(dropout_train_smote$Dropped_Out)


# Export data to CSV
#write.csv(OC_data, file = "dropout_data.csv", row.names = FALSE)
#write.csv(OC_train, file = "dropout_train.csv", row.names = FALSE)
#write.csv(OC_test, file = "dropout_test.csv", row.names = FALSE)
#write.csv(OC_train_smote, file = "dropout_train_smote.csv", row.names = FALSE)

#Model having issues solution
dropout_train$Dropped_Out <- make.names(dropout_train$Dropped_Out)


#install.packages("xgboost")
library(xgboost)

#XGBoost Model

set.seed(8)
model_gbm <- train(Dropped_Out~.,
                   data = dropout_train,
                   method = "xgbTree",
                   # provide a grid of parameters
                   tuneGrid = expand.grid(
                     nrounds = c(50,200),
                     eta = c(0.025, 0.05),
                     max_depth = c(2, 3),
                     gamma = 0,
                     colsample_bytree = 1,
                     min_child_weight = 1,
                     subsample = 1),
                   trControl= trainControl(method = "cv",
                                           number = 5,
                                           classProbs = TRUE,
                                           summaryFunction = twoClassSummary),
                   metric = "ROC"
)


#Performance based on various tuning parameters
plot(model_gbm)

#Print out of the best tuning parameters
model_gbm$bestTune

#only print top 10 important variables
plot(varImp(model_gbm), top=10)

#SHAP
#install.packages("SHAPforxgboost")
library(SHAPforxgboost)

# Exclude character columns and Dropped_Out


Xdata<-as.matrix(select(dropout_train,-Dropped_Out)) # change data to matrix for plots

Xdata <- dropout_train %>% 
  select_if(is.numeric) %>% 
  as.matrix()

# Calculate SHAP values
shap <- shap.prep(model_gbm$finalModel, X_train = Xdata)

# SHAP importance plot for top 15 variables
shap.plot.summary.wrap1(model_gbm$finalModel, X = Xdata, top_n = 10)

#example partial dependence plot

p <- shap.plot.dependence(
  shap, 
  x = "Final_Grade", #top val in shapp
  color_feature = "Final_Grade", 
  smooth = FALSE, 
  jitter_width = 0.01, 
  alpha = 0.4
) +
  ggtitle("Final_Grade")
print(p)



