#Ioannis Matzakos Chorianopoulos 
#Student Number: 100185943

##SUPERVISED CLASSIFICATION OF AIR POLLUTION LABELLED DATA

#install packages
#install.packages("tm", "RTextTools", "e1071", RWeka")

#Load the data from .csv file
data = read.csv(file="air_pollution_labelled_data.csv")
#save the original form of the data in case it is needed again
data_original = data

#Preliminary Analysis of the dataset
names(data)
nrow(data)
ncol(data)
summary(data)
str(data)

###
##Tweet Source Filtering (optional | comment this part of code if you want to ignore it)
#select the most valuable instances based on what source each relevant tweet was posted from

#save relevant tweets separately in a dataframe in order to avoid losing any during the filtering
relevant = subset(data, data$Relevant=="Relevant")
#save irrelevant tweets separately in order to filter the ones created by bots and news websites
irrelevant = subset(data, data$Relevant=="Not Relevant")

#save all the sources of the relevant tweets
reliable_sources = unique(relevant$source, incomparables = FALSE)
length(reliable_sources)
save(reliable_sources, file = "reliable_sources.RData")

#keep in a separate dataframe all the irrelevant tweets that have been created from the sources in the vector reliable_sources
irrelevant_reduced = irrelevant[irrelevant$source %in% reliable_sources, ]
#check how may are th irrelevant tweets at this point
nrow(irrelevant_reduced)

#combine the relevant and the reduced set of the irrelevant tweets
reduced_set = rbind(relevant, irrelevant_reduced)
#order the tweets based on their ID
reduced_set = reduced_set[order(reduced_set$ID),]
#save reduced set
#save(reduced_set, file = "reduced_set.RData")

data = reduced_set

#Preliminary Analysis of the dataset
names(reduced_set)
nrow(reduced_set)
ncol(reduced_set)
summary(reduced_set)
str(reduced_set)
###

##DATA CLEANSING & PREPROCESSING
require(tm)

#creating a corpus
corpus = Corpus(VectorSource(data$text))
#transform all words to lower case
corpus = tm_map(corpus, content_transformer(tolower))

#url removal function
removeURL = function(x) gsub("http[^[:space:]]*", "", x)
#numbers, punctuation and space removal function
removeNumPunct = function(x) gsub("[^[:alpha:][:space:]]*", "", x)

#url removal from corpus 
corpus = tm_map(corpus, content_transformer(removeURL))
#numbers and punctuation removal form corpus
corpus = tm_map(corpus, content_transformer(removeNumPunct))

#read the txt file with the stopwords
john_stopwords = read.table("john_stopwords.txt")
#john_stopwords
#remove common words from stopwords
stop_words = setdiff(stop_words, as.vector(john_stopwords))
#remove stopwords from corpus
corpus = tm_map(corpus, removeWords, stop_words)

#remove white space
corpus = tm_map(corpus, stripWhitespace)
#stemming
corpus = tm_map(corpus, stemDocument)

#create a dtm using term frequency and 3-grams
require(RWeka)
ThreegramTokenizer <- function(x) {NGramTokenizer(x, Weka_control(min=3, max=3))}
dtm = DocumentTermMatrix(corpus, control = list(tokenize=ThreegramTokenizer, weighting = weightTf))
save(dtm, file = "dtm-3grams-tf.RData")


##SPLITTING THE DATASET INTO TRAINING AND TEST SETS

#convert document term matrix into a matrix
dtm_mat = as.matrix(dtm)

#convert document term matrix into a data frame
dtm_df = as.data.frame(dtm_mat)

#save dtm into a csv file
save(dtm_df, file = "dtm_df-3grams-tf.RData")

#save IDs into a data frame
tweet_ids = data.frame(data$ID)

#include IDs and Labels in the document term matrix
dtm_df[,"ID"] = tweet_ids
dtm_df[,"Relevance"] = data$Relevant
#check if the new columns are added at the end of the dataframe
tail(names(dtm_df))

#save dtm dataframe for the semi supervised section
#save.image("dtm_df.RData")

#random sampling based on "ID" column
train_set_IDs = sample(data$ID, nrow(data)*0.6, replace = FALSE)
test_set_IDs = sample(data$ID, nrow(data)*0.4, replace = FALSE)
#ordering the IDs in an ascending order
train_set_IDs = train_set_IDs[order(train_set_IDs)]
test_set_IDs = test_set_IDs[order(test_set_IDs)]
#training set size
length(train_set_IDs)
#test set size
length(test_set_IDs)

#creating a data frame from the document term matrix for the training and test sets respectively based on the randomly chosen IDs
train_dtm_df = dtm_df[dtm_df$ID %in%  train_set_IDs, ]
tail(names(train_dtm_df))
test_dtm_df = dtm_df[dtm_df$ID %in%  test_set_IDs, ]
tail(names(test_dtm_df))

#balancing the training set
#subset containing only the relevant tweets
relevant_tweets = subset(train_dtm_df, train_dtm_df$Relevance=="Relevant")
#summary(relevant_tweets)
#subset containing only the irrelevant tweets
irrelevant_tweets = subset(train_dtm_df, train_dtm_df$Relevance=="Not Relevant")
#summary(irrelevant_tweets)
#bind both classes of tweets together
train_dtm_df = rbind(relevant_tweets, irrelevant_tweets[1:(nrow(relevant_tweets)*2.6),])
#order the tweets by ID
train_dtm_df = train_dtm_df[order(train_dtm_df$ID),]


##PREPARE THE DATA FOR BUILDING THE CLASSIFIERS
require(RTextTools)
require(e1071)

#Defining a Training Set and a Test Set
#store the labels of training and testing set separately
trainlabels = train_dtm_df$Relevance
testlabels= test_dtm_df$Relevance

save(trainlabels, file = "trainlabels-3grams-tf.RData")
save(testlabels, file = "testlabels-3grams-tf.RData")

length(trainlabels)
length(testlabels)

#saving original dtms
train_dtm_df_original = train_dtm_df
test_dtm_df_original = test_dtm_df

#remove Relevance from the dtms
train_dtm_df$Relevance = NULL
test_dtm_df$Relevance = NULL

#convert document term matrix (dtm) into a normal matirx
trainmat = as.matrix(train_dtm_df) #using term frequency
testmat = as.matrix(test_dtm_df)

save(trainmat, file = "trainmat-3grams-tf.RData")
save(testmat, file = "testmat-3grams-tf.RData")

#using TF-IDF
#trainmat = as.matrix(train_dtm_tfidf_df) 
#testmat = as.matrix(test_dtm_tfidf_df)

#trainmat = topic_prob_lda

nrow(trainmat)
nrow(testmat)

#Configure training data
train_container = create_container(trainmat, trainlabels, trainSize=1:nrow(trainmat), testSize=NULL, virgin=FALSE)
#Configure test data
test_container = create_container(testmat, testlabels, trainSize=NULL, testSize=1:nrow(testmat), virgin=FALSE)

save(train_container, file = "train_container-3grams-tf.RData")
save(test_container, file = "test_container-3grams-tf.RData")

save.image("data_after_preprocessing-3grams-tf.RData")


##ANALYTICS & PERFORMANCE METRICS

#training set size
nrow(trainmat)

#test set size
nrow(testmat)

#total number of Relevant tweets in the training set
length(trainlabels[trainlabels=="Relevant"])

#total number of Not Relevant tweets in the training set
length(trainlabels[trainlabels=="Not Relevant"])

#total number of Relevant tweets in the test set
length(testlabels[testlabels=="Relevant"])

#total number of Not Relevant tweets in the test set
length(testlabels[testlabels=="Not Relevant"])

#analytics function | returns a dataframe with all metrics for each class
analytics = function(confusion_matrix){
  
  #number of instances in the test set
  instances = sum(confusion_matrix) 
  
  #number of labels
  labels = nrow(confusion_matrix) 
  
  #number of correctly classified instances per class
  diagonal = diag(confusion_matrix) 
  
  #number of incorrectly classified instances per class
  antidiagonal = diag(confusion_matrix[nrow(confusion_matrix):1, ])
  
  #number of instances per class
  row_sum = apply(confusion_matrix, 1, sum) 
  
  #number of predictions per class
  column_sum = apply(confusion_matrix, 2, sum) 
  
  #distribution of instances over the actual classes
  actual_dist = row_sum / instances 
  
  #distribution of instances over the predicted classes
  predicted_dist = column_sum / instances 
  
  #overall accuracy
  accuracy = sum(diagonal) / instances 
  
  #precision of labeling in each class: the percentage of correctly calssified instances
  precision = diagonal/column_sum
  
  #recall: the number of correct results divided by the number of results that should have been returned
  recall = diagonal/row_sum
  
  #F-score statistic: harmonic mean/weighted average of precision and recall
  f1 = 2*precision*recall/(precision+recall)
  
  beta =2
  
  f2 = (1 + beta*beta) * ((precision*recall)/(beta*beta*precision+recall))
  
  #overall view of performance metrics
  performance_metrics = data.frame(Accuracy = accuracy, Precision = precision, Recall = recall, F1 = f1, F2 = f2)
  #performance_metrics
  return(performance_metrics)
}

##Model Training, Testing and Evaluation

###Naive Bayes
#train a NB Model
nb_model = naiveBayes(trainmat, trainlabels)
summary(nb_model)

nb_results = predict(nb_model, testmat, type = "class", threshold = 0.03, eps = 0)
nb_results2 = predict(nb_model, testmat, type = "raw", threshold = 0.03, eps = 0)
nb_results2 = apply(nb_results2, 1, max)

#confusion matrix
nb_actual_labels = testlabels
nb_predicted_labels = nb_results
nb_confusion_matrix = as.matrix(table(Actual = nb_actual_labels, Predicted = nb_predicted_labels))
nb_confusion_matrix
write.csv(nb_confusion_matrix, file = "nb_confusion_matrix-3grams-tf.csv")

nb_performance = analytics(nb_confusion_matrix)
nb_performance
write.csv(nb_performance, file = "nb_performance-3grams-tf.csv")

###Support Vector Machines
#train a SVM Model
svm_model = train_model(train_container, "SVM", kernel="linear", cost=3, type="C-classification")
summary(svm_model)

#test the model | predict the labels
svm_results = classify_model(test_container, svm_model)

#confusion matrix
svm_actual_labels = testlabels
svm_predicted_labels = as.vector(svm_results$SVM_LABEL)
svm_confusion_matrix = as.matrix(table(Actual = svm_actual_labels, Predicted = svm_predicted_labels))
svm_confusion_matrix
write.csv(svm_confusion_matrix, file = "svm_confusion_matrix-3grams-tf.csv")

svm_performance = analytics(svm_confusion_matrix)
svm_performance
write.csv(svm_performance, file = "svm_performance-3grams-tf.csv")

###Random Forests
#train a RF Model
rf_model = train_model(train_container, "RF")
summary(rf_model)

#test the model | predict the labels
rf_results = classify_model(test_container, rf_model)

#confusion matrix
rf_actual_labels = testlabels
rf_predicted_labels = rf_results$FORESTS_LABEL
rf_confusion_matrix = as.matrix(table(Actual = rf_actual_labels, Predicted = rf_predicted_labels))
rf_confusion_matrix
write.csv(rf_confusion_matrix, file = "rf_confusion_matrix-3grams-tf.csv")

rf_performance = analytics(rf_confusion_matrix)
rf_performance
write.csv(rf_performance, file = "rf_performance-3grams-tf.csv")

###Decision Trees
#train a Decision Tree Model
tree_model = train_model(train_container, "TREE", kernel="linear", cost=1, type="C-classification",  class.weights=c("Relevant"=10, "Not Relevant"=1))
summary(tree_model)

#test the model | predict the labels
tree_results = classify_model(test_container, tree_model)

#confusion matrix
#actual_labels = as.vector(subdata$Relevant[601:1000])
tree_actual_labels = testlabels
tree_predicted_labels = as.vector(tree_results$TREE_LABEL)
tree_confusion_matrix = as.matrix(table(Actual = tree_actual_labels, Predicted = tree_predicted_labels))
tree_confusion_matrix
write.csv(tree_confusion_matrix, file = "tree_confusion_matrix-3grams-tf.csv")

tree_performance = analytics(tree_confusion_matrix)
tree_performance
write.csv(tree_performance, file = "tree_performance-3grams-tf.csv")

###SLDA
#train the model using slda
slda_model = train_model(train_container, "SLDA")
summary(slda_model)

#test the model | predict the labels
slda_results = classify_model(test_container, slda_model)
slda_results

#confusion matrix
slda_actual_labels = testlabels
slda_predicted_labels = as.vector(slda_results$SLDA_LABEL)
slda_confusion_matrix = as.matrix(table(Actual = slda_actual_labels, Predicted = slda_predicted_labels))
slda_confusion_matrix
write.csv(slda_confusion_matrix, file = "slda_confusion-matrix-3grams-tf.csv")

slda_performance = analytics(slda_confusion_matrix)
slda_performance
write.csv(slda_performance, file = "slda_performance-3grams-tf.csv")

###MAXENT
#train the model using maxent algorithm 
maxent_model = train_model(train_container, "MAXENT")
summary(maxent_model)

#test the model | predict the labels
maxent_results = classify_model(test_container, maxent_model)

#confusion matrix
maxent_actual_labels = testlabels
maxent_predicted_labels = as.vector(maxent_results$MAXENTROPY_LABEL)
maxent_confusion_matrix = as.matrix(table(Actual = maxent_actual_labels, Predicted = maxent_predicted_labels))
maxent_confusion_matrix
write.csv(maxent_confusion_matrix, file = "maxent_confusion_matrix-3grams-tf.csv")

maxent_performance = analytics(maxent_confusion_matrix)
maxent_performance
write.csv(maxent_performance, file = "maxent_performance-3grams-tf.csv")

save.image("workspace_sc-3grams-tf.RData")
