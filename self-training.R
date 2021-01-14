#Ioannis Matzakos Chorianopoulos 
#Student Number: 100185943

#PARTIALLY SUPERVISED CLASSIFICATION OF AIR POLLUTION DATA

require(tm)
require(RTextTools)
require(e1071)

load("train_dtm2_df.RData")
load("train_dtm2_df_original.RData")
load("test_dtm2_df.RData")
load("train_dtm2_df2.RData")

###########################################
### PARTIALLY SUPERVISED CLASSIFICATION ###

#remove relevance column from training set
trainlabels2 = train_dtm2_df$Relevance
train_dtm2_df_original = train_dtm2_df
#train_dtm2_df$Relevance = as.factor(train_dtm2_df$Relevance)
#train_dtm2_df$Relevance = NULL

#remove relevance column from test set
testlabels2 = test_dtm2_df$Relevance
test_dtm2_df_original = test_dtm2_df
test_dtm2_df$Relevance = NULL

#trainmat2 = as.matrix(train_dtm2_df)
#testmat2 = as.matrix(test_dtm2_df)

#remove relevance column from unlabeled set
unlabeled_set = train_dtm2_df2
unlabeled_set$Relevance = NULL
tail(names(unlabeled_set))

#remove terms with non ASCII characters from training and test sets
#get terms
terms = names(train_dtm2_df)
length(terms)
nonwords = grep("terms", iconv(terms, "latin1", "ASCII", sub="terms"))
terms = terms[-nonwords]
length(terms)

ncol(train_dtm2_df)
tail(names(train_dtm2_df))
train_dtm2_df = train_dtm2_df[ , (names(train_dtm2_df) %in% terms)]
ncol(train_dtm2_df)
tail(names(train_dtm2_df))

ncol(test_dtm2_df)
tail(names(test_dtm2_df))
test_dtm2_df = test_dtm2_df[ , (names(test_dtm2_df) %in% terms)]
ncol(test_dtm2_df)
tail(names(test_dtm2_df))

ncol(unlabeled_set)
tail(names(unlabeled_set))
unlabeled_set = unlabeled_set[ , (names(unlabeled_set) %in% terms)]
ncol(unlabeled_set)
tail(names(unlabeled_set))

#function used for labeling the unlabeled data in self training approach
label = function(model, test_data){
  labels = predict(model, test_data, type = "class", threshold = 0.03, eps = 0)
  confidence = apply(predict(model, test_data, type = "raw", threshold = 0.03, eps = 0), 1, max)
  results = data.frame(Labels=labels, Prob=confidence)
  return(results)
}

#function that generates performance metrics from a condusion matrix and returnns them in a dataframe
analytics = function(confusion_matrix){
  #number of instances in the test set
  instances = sum(confusion_matrix) 
  instances
  
  #number of labels
  labels = nrow(confusion_matrix) 
  labels
  
  #number of correctly classified instances per class
  diagonal = diag(confusion_matrix) 
  diagonal
  
  #number of incorrectly classified instances per class
  antidiagonal = diag(confusion_matrix[nrow(confusion_matrix):1, ])
  antidiagonal
  
  #number of instances per class
  row_sum = apply(confusion_matrix, 1, sum) 
  row_sum
  
  #number of predictions per class
  column_sum = apply(confusion_matrix, 2, sum) 
  column_sum
  
  #distribution of instances over the actual classes
  actual_dist = row_sum / instances 
  actual_dist
  
  #distribution of instances over the predicted classes
  predicted_dist = column_sum / instances 
  predicted_dist
  
  #overall accuracy
  accuracy = sum(diagonal) / instances 
  #print(accuracy)
  
  #precision of labeling in each class: the percentage of correctly calssified instances
  precision = diagonal/column_sum
  #print(precision)
  
  #recall/sensitivity: the number of correct results divided by the number of results that should have been returned
  recall = diagonal/row_sum
  #print(recall)
  
  #specificity: the portion of tweets that were relevant among those that were not relevant
  specificity = sum(antidiagonal)/(sum(antidiagonal)+confusion_matrix[2,1]) #false positives = confusion_matrix[2,1] 
  #print(specificity)
  #A test with a higher specificity has a lower type I error rate.
  
  #F-score statistic: harmonic mean/weighted average of precision and recall
  f1 = 2*precision*recall/(precision+recall)
  #print(f1)
  
  beta = 2 
  f2 = (1 + beta*beta) * ((precision*recall)/(beta*beta*precision+recall))
  #print(f2)
  
  #overall view of performance metrics
  performance_metrics = data.frame(Accuracy = accuracy, Precision = precision, Recall = recall, F1 = f1, F2 = f2)
  #print(performance_metrics)
  return(performance_metrics)
}

maxsize = 500
maxsize

#the number of times that the unlabeled data are going to be increased by 700 instances
iterations = nrow(unlabeled_set) %/% maxsize
iterations

index = 1

confusion_matrices = ls()
performances = ls()

#Self Training Process
for(i in 1:iterations){
  cat("Iteration No.", i, "\n")
  #train model
  #nb_model = naiveBayes(train_dtm2_df, trainlabels2)
  nb_model = naiveBayes(Relevance~., train_dtm2_df)
  print("Model created")
  
  #test model
  nb_results = predict(nb_model, test_dtm2_df, type = "class", threshold = 0.03, eps = 0)
  print("Model tested, results available")
  
  #confusion matrix
  nb_actual_labels = testlabels2
  nb_predicted_labels = nb_results
  nb_confusion_matrix = as.matrix(table(Actual = nb_actual_labels, Predicted = nb_predicted_labels))
  print(nb_confusion_matrix)
  
  save(list = (confusion_matrices[i]), file = paste(confusion_matrices[i],".RData", sep = ""))
  print("Confusion matrix saved in an .RData file")
  confusion_matrix = as.data.frame(nb_confusion_matrix)
  write.table(confusion_matrix, "confusion_matrices.csv", col.names = TRUE, append = TRUE)
  print("Confusion matrix saved in a .csv file")
  
  #performance metrics
  perf = analytics(nb_confusion_matrix)
  print(perf)
  
  save(list = (performances[i]), file = paste(performances[i],".RData", sep = ""))
  print("Performance saved in an .RData file")
  write.table(perf, "performances.csv", col.names = TRUE, append = TRUE)
  print("Performance saved in a .csv file")
  
  #label unlabeled data
  predicted_labels = label(nb_model, unlabeled_set[index:maxsize, ])
  cat("Number of predictions equals 500: ", nrow(predicted_labels) == 500, "\n")
  print("Predictions for unlabeled data made")
  
  #attaching predictions to the unlabeled set in order to select the instances to add to training
  tempdata = unlabeled_set[index:maxsize, ]
  tempdata$Relevance = predicted_labels$Labels
  tempdata$Probabilities = predicted_labels$Prob
  tail(names(tempdata))
  nrow(tempdata)
  cat("Number of predictions equals the number of temp data: ", nrow(tempdata) == nrow(predicted_labels), "\n")
  print("New labeles attached to unlabeled_set")
  
  #create a dataframe for the new instances to be added in the training set
  data_to_add = subset(tempdata, tempdata$Probabilities>=0.95)
  tail(names(data_to_add))
  nrow(data_to_add)
  #data_to_add$Relevance = NULL
  data_to_add$Probabilities = NULL
  #data_to_add = as.matrix(data_to_add)
  print("Data to add to training set stored")
  
  #create a vector of the new labels to be added in the training set labels
  labels_to_add = subset(tempdata$Relevance, tempdata$Probabilities>=0.95)
  labels_to_add = as.vector(labels_to_add)
  length(labels_to_add)
  print("Labels to add to training set stored")
  
  cat("Number of data to be added equals the number of labels to be added: ", nrow(data_to_add)==length(labels_to_add), "\n")
  
  #combine the new data with the existing training set
  train_dtm2_df = rbind(train_dtm2_df, data_to_add)
  nrow(train_dtm2_df)
  trainlabels2 = c(trainlabels2, labels_to_add)
  length(trainlabels2)
  print("New data combined with old data in the training set")
  
  #new index equals with the old index plus maxsize
  index = index + maxsize
  cat("The new index for unlabeled data is ", index, "\n")
  #maxsize increases by 500
  maxsize = maxsize + 500
  cat("The new maxsize is ", maxsize, "\n")
  
  cat("New training set size is: ", nrow(train_dtm2_df), "\n")
  cat("New training labels size is: ", length(trainlabels2), "\n")
  cat("Training set and labels have equal size: ", nrow(train_dtm2_df)==length(trainlabels2), "\n")
  
  if(maxsize>nrow(unlabeled_set)){
    break
    print("Unlabeled dataset run out of data")
    print("Self training process using Naive Bayes completed!")
  }
  
}

