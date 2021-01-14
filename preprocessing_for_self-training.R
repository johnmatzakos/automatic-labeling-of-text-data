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

#load unlabeled data
undata = read.csv("airpollution2months6days.csv")
#save the original form of the data in case it is needed again
undata_original = undata

#Preliminary Analysis of the dataset
names(undata)
nrow(undata)
ncol(undata)
summary(undata)
str(undata)

undata$ID = (nrow(data)+1):(nrow(undata)+nrow(data))
undata$Relevant = NA
undata = undata[c("ID", "source", "created_at", "Relevant", "text", "coordinates", "user_id", "time_zone", "place")]
#undata = undata[names(data)]
names(data)
names(undata)
nrow(undata)

#sampling the unlabeled dataset for memory reasons
undata_ids = sample(undata$ID, nrow(undata)*0.007, replace = FALSE)
#ordering the IDs in an ascending order
undata_ids = undata_ids[order(undata_ids)]
#training set size
length(undata_ids)
#creating a data frame from the document term matrix for the training and test sets respectively based on the randomly chosen IDs
undata = undata[undata$ID %in%  undata_ids, ]
names(undata)
nrow(undata)

#combine labeled and unlabeled data in the same dataframe
undata = rbind(data, undata)


#Data Cleansing Part 2 & Prepeocessing for the Unlabled Data

#url removal
removeURL = function(x) gsub("http[^[:space:]]*", "", x)
#removeNumPunct = function(x) gsub("[^[:alpha:][:space:]]*", "", x)
#numbers removal
removeNum = function(x) gsub("[[:digit:]]", "", x)
#punctuation removal
removePunct = function(x) gsub("[[:punct:]]", "", x)

#creating a corpus
corpus2 = Corpus(VectorSource(undata$text))
#transform all words to lower case
corpus2 = tm_map(corpus2, content_transformer(tolower))
#corpus$content

#url removal from corpus
corpus2 = tm_map(corpus2, content_transformer(removeURL))
#remove punctuation
corpus2 = tm_map(corpus2, content_transformer(removePunct))
#remove numbers
corpus2 = tm_map(corpus2, content_transformer(removeNum))

#read the txt file with the stopwords
#other stopword files: "default_stopwords.txt" "longlist_stopwords.txt" and "google_stopwords.txt"
john_stopwords = read.table("john_stopwords.txt")
#john_stopwords
#remove common words from stopwords
stop_words = setdiff(stop_words, as.vector(john_stopwords))
#remove stopwords from corpus
corpus2 = tm_map(corpus2, removeWords, stop_words)

#remove white space
corpus2 = tm_map(corpus2, stripWhitespace)
#stemming
corpus2 = tm_map(corpus2, stemDocument)

#create a dtm using term frequency
dtm2 = DocumentTermMatrix(corpus2, control = list(weighting = weightTf))

save(dtm2, file = "dtm2.RData")

##SPLITTING AND BALANCING

dtm2_mat = as.matrix(dtm2)
dtm2_df = as.data.frame(dtm2_mat)
save(dtm2_mat, file = "dtm2_mat.RData")
#load("dtm2_mat.RData")

#save IDs into a data frame
tweet_ids = data.frame(undata$ID)
nrow(tweet_ids)

labeled= undata[1:2885, ]
nrow(labeled)

unlabeled = undata[2886:nrow(undata), ]
nrow(unlabeled)

unlabeled_ids = subset(undata$ID , is.na.data.frame(undata$Relevant))
length(unlabeled_ids)
unlabeled_ids = as.data.frame(unlabeled_ids)
nrow(unlabeled_ids)

#include IDs and Labels in the document term matrix
dtm2_df[,"ID"] = tweet_ids
dtm2_df[,"Relevance"] = undata$Relevant
#check if the new columns are added at the end of the dataframe
tail(names(dtm2_df))
nrow(dtm2_df)

#random sampling based on "ID" column
train_set_IDs = sample(labeled$ID, nrow(labeled)*0.6, replace = FALSE)
#ordering the IDs in an ascending order
train_set_IDs = train_set_IDs[order(train_set_IDs)]
#training set size
length(train_set_IDs)
#creating a data frame from the document term matrix for the training and test sets respectively based on the randomly chosen IDs
train_dtm2_df = dtm2_df[dtm2_df$ID %in%  train_set_IDs, ]
head(names(train_dtm2_df))
tail(names(train_dtm2_df))
nrow(train_dtm2_df)

test_set_IDs = sample(labeled$ID, nrow(labeled)*0.4, replace = FALSE)
#ordering the IDs in an ascending order
test_set_IDs = test_set_IDs[order(test_set_IDs)]
#test set size
length(test_set_IDs)
#creating a data frame from the document term matrix for the training and test sets respectively based on the randomly chosen IDs
test_dtm2_df = dtm2_df[dtm2_df$ID %in%  test_set_IDs, ]
tail(names(test_dtm2_df))

#balancing the training set
#subset containing only the relevant tweets
relevant_tweets = subset(train_dtm2_df, train_dtm2_df$Relevance=="Relevant")
#subset containing only the irrelevant tweets
irrelevant_tweets = subset(train_dtm2_df, train_dtm2_df$Relevance=="Not Relevant")
#bind both classes of tweets together
train_dtm2_df = rbind(relevant_tweets, irrelevant_tweets[1:(nrow(relevant_tweets)*2.6),])
#order the tweets by ID
train_dtm2_df = train_dtm2_df[order(train_dtm2_df$ID),]
tail(names(train_dtm2_df))
nrow(train_dtm2_df)

#train_dtm2_df2 = dtm2_df[dtm2_df$ID %in%  unlabeled_ids, ]
train_dtm2_df2 = dtm2_df[2886:nrow(dtm2_df), ]
nrow(train_dtm2_df2)
train_dtm2_df_original = train_dtm2_df
nrow(train_dtm2_df_original)
train_dtm2_df = rbind(train_dtm2_df, train_dtm2_df2)
head(names(train_dtm2_df))
tail(names(train_dtm2_df))
nrow(train_dtm2_df)

save(train_dtm2_df_original, file = "train_dtm2_df_original.RData")
save(train_dtm2_df, file = "train_dtm2_df.RData")
save(test_dtm2_df, file = "test_dtm2_df.RData")


