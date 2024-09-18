#Classification Tree
#step 0: exporatory analysis
dropout_data <- read_csv("~/Downloads/student dropout.csv")

library(caret)
library(tidyverse)

library(skimr)
summaryStats <- skim(dropout_data)
summaryStats

#convert categorical predictor variables and the response to factor

dropout_data <- dropout_data %>% 
  mutate_at(c("Dropped_Out", "School", "Gender", "Address",
              "Family_Size", "Parental_Status", "Mother_Job", "Father_Job",
              "Reason_for_Choosing_School", "Guardian", "School_Support",
              "Family_Support", "Extra_Paid_Class", "Extra_Curricular_Activities",
              "Attended_Nursery", "Wants_Higher_Education", "Internet_Access",
              "In_Relationship"
  ), 
  as.factor)

#Step 1: Partition our data
#create dummy variables expect for the response
#you can either keep factors as is or convert to dummy
dummies_model <- dummyVars(Dropped_Out ~ .,
                           data = dropout_data)

#provide only predictors that are now converted to dummy variables
dropout_predictors_dummy<- data.frame(predict(dummies_model,
                                            newdata = dropout_data)) 

#recombine predictors including dummy variables with response
dropout_data <- cbind(Dropped_Out=dropout_data$Dropped_Out,
                    dropout_predictors_dummy)

dropout_data$Dropped_Out<-fct_recode(dropout_data$Dropped_Out,
                             "0" = "FALSE",
                             "1" = "TRUE") 
dropout_data$Dropped_Out<-relevel(dropout_data$Dropped_Out,
                          ref="1")

set.seed(99) #set random seed

index <- createDataPartition(dropout_data$Dropped_Out, p=.8,
                             list=F)

dropout_train <- dropout_data[index,]
dropout_test <- dropout_data[-index,]

#Step 2: train or fit model
# install and load packages for machine learning model
#install.packages("rpart")
library(rpart)

#Model having issues solution
dropout_train$Dropped_Out <- make.names(dropout_train$Dropped_Out)

set.seed(12)
dropout_model <- train(Dropped_Out~.,
                     data = dropout_train,
                     method = "rpart",
                     tuneGrid = expand.grid(cp=seq(0.01,0.2,length=5)),
                     trControl=trainControl(method = "cv",
                                            number = 5,
                                            #number of folds
                                            ## Estimate class probabilities
                                            classProbs = TRUE,
                                            #needed to get ROC
                                            summaryFunction = twoClassSummary),
                     metric="ROC") 

dropout_model #provides information of parameter tuning via cross validation
plot(dropout_model) #provides plot of parameter tuning via cross validation

#plot variable importance
plot(varImp(dropout_model))

#plot the tree
#install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(dropout_model$finalModel, type=5)

#Step 3: get predictions using testing set data

predictions <- predict(dropout_model, dropout_test, type="prob")

#step 4: Get AUC
library(ROCR)

# Get the predicted probabilities for the positive class ("1")
predictions <- predict(dropout_model, dropout_test, type = "prob")

# Ensure the structure of predictions (it should have columns for each class "0" and "1")
str(predictions)
head(predictions)

# Extract probabilities for class "1"
predicted_prob <- predictions[, "X1"]


# Convert the true labels to numeric values (as ROCR expects numeric vectors)
true_labels <- as.numeric(as.character(dropout_test$Dropped_Out))

# Use the ROCR prediction() function
dropout_test_prediction <- prediction(predicted_prob, true_labels)


perf <- performance(dropout_test_prediction,"tpr","fpr")

# Plot the ROC curve
plot(perf, col = "blue", lwd = 2, main = "ROC Curve for Dropout Model")

# Calculate AUC (Area Under the Curve)
auc <- performance(dropout_test_prediction, "auc")
auc_value <- auc@y.values[[1]]

# Print AUC value
print(paste("AUC:", auc_value))

plot(perf, colorize=T)

