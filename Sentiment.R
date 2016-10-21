#install.packages(e1071,dependencies = TRUE)
#install.packages(RTextTools,dependencies = TRUE)
#install.packages(caret,dependencies = TRUE)
library(RTextTools)
library(e1071)
library(caret)

positiveReviews_train = readLines("./positiveReviews_train.txt")
negativeReviews_train = readLines("./negativeReviews_train.txt")
positiveReviews_test = readLines("./positiveReviews_test.txt")
negativeReviews_test = readLines("./negativeReviews_test.txt")

reviews_train= c(positiveReviews_train, negativeReviews_train)
reviews_test= c(positiveReviews_test, negativeReviews_test)
reviews = c(reviews_train, reviews_test)
sentiment_train = c(rep("positive", length(positiveReviews_train) ), 
              rep("negative", length(negativeReviews_train)))
sentiment_test = c(rep("positive", length(positiveReviews_test) ), 
                   rep("negative", length(negativeReviews_test)))
sentiment_data = as.factor(c(sentiment_train, sentiment_test))



# Here naive Bayes is implemented
naiveBayes_matrix= create_matrix(reviews, language="english", 
                   removeStopwords=FALSE, removeNumbers=TRUE, 
                   stemWords=FALSE, tm::weightTfIdf)

naiveBayes_matrix = as.matrix(naiveBayes_matrix)

naiveBayes_classifier = naiveBayes(naiveBayes_matrix[1:160,], as.factor(sentiment_data[1:160]))
naiveBayes_pred = predict(naiveBayes_classifier, naiveBayes_matrix[161:454,])

table(sentiment_test, naiveBayes_pred)
#Confusionmatrix of naive Bayes to calculate
#Precision ,Recall, Accuracy
confusionMatrix(table(sentiment_test, naiveBayes_pred))

# the other methods
allModels_matrix= create_matrix(reviews, language="english", 
                   removeStopwords=FALSE, removeNumbers=TRUE, 
                   stemWords=FALSE, tm::weightTfIdf)

container = create_container(allModels_matrix, as.numeric(sentiment_data),
                             trainSize=1:160, testSize=161:454,virgin=FALSE) 
#All Supervised Learning Algorithms
allModels = train_models(container, algorithms=c("SVM","GLMNET",
                                              "RF", "TREE" ,
                                              "SLDA","BAGGING","BOOSTING",
                                             "MAXENT" ))

# test the model
results = classify_models(container, allModels)

# Here Random Forest is implemented
table(as.numeric(as.numeric(sentiment_data[161:454])), results[,"FORESTS_LABEL"])
#Confusionmatrix of Random Forest to calculate
#Precision ,Recall, Accuracy
#confusionMatrix(table(as.numeric(as.numeric(sentiment_data[161:454])), results[,"FOREST_LABEL"]))
recall_accuracy(as.numeric(as.factor(sentiment_data[161:454])), results[,"FORESTS_LABEL"])


# Here Maximum Entropy is implemented
table(as.numeric(as.factor(sentiment_data[161:454 ])), results[,"MAXENTROPY_LABEL"])
 #Accuracy
recall_accuracy(as.numeric(as.factor(sentiment_data[161:454])), results[,"MAXENTROPY_LABEL"])

#confusionMatrix(table(as.numeric(as.numeric(sentiment_data[161:454])), results[,"MAXENTROPY_LABEL"]))


# Here Decison Tree is implemented
table(as.numeric(as.factor(sentiment_data[161:454 ])), results[,"TREE_LABEL"])
#Confusionmatrix of Decison Tree to calculate
#Precision ,Recall, Accuracy
#confusionMatrix(table(as.numeric(as.numeric(sentiment_data[161:454])), results[,"TREE_LABEL"]))
recall_accuracy(as.numeric(as.factor(sentiment_data[161:454])), results[,"TREE_LABEL"])


# Here Bagging is implemented
table(as.numeric(as.factor(sentiment_data[161:454 ])), results[,"BAGGING_LABEL"])
#Confusionmatrix of Bagging to calculate
#Precision ,Recall, Accuracy
#confusionMatrix(table(as.numeric(as.numeric(sentiment_data[161:454])), results[,"BAGGING_LABEL"]))
recall_accuracy(as.numeric(as.factor(sentiment_data[161:454])), results[,"BAGGING_LABEL"])



# Here SVM is implemented
table(as.numeric(as.factor(sentiment_data[161:454 ])), results[,"SVM_LABEL"])
#Confusionmatrix of SVM to calculate
#Precision ,Recall, Accuracy
#confusionMatrix(table(as.numeric(as.numeric(sentiment_data[161:454])), results[,"SVM_LABEL"]))
recall_accuracy(as.numeric(as.factor(sentiment_data[161:454])), results[,"SVM_LABEL"])


# Here GLMNET is implemented
table(as.numeric(as.factor(sentiment_data[161:454 ])), results[,"GLMNET_LABEL"])
#Confusionmatrix of GLMNET to calculate
#Precision ,Recall, Accuracy
#confusionMatrix(table(as.numeric(as.numeric(sentiment_data[161:454])), results[,"GLMNET_LABEL"]))
recall_accuracy(as.numeric(as.factor(sentiment_data[161:454])), results[,"GLMNET_LABEL"])



# Here SLDA is implemented
table(as.numeric(as.factor(sentiment_data[161:454 ])), results[,"SLDA_LABEL"])
#Confusionmatrix of SLDA to calculate
#Precision ,Recall, Accuracy
#confusionMatrix(table(as.numeric(as.numeric(sentiment_data[161:454])), results[,"SLDA_LABEL"]))
recall_accuracy(as.numeric(as.factor(sentiment_data[161:454])), results[,"SLDA_LABEL"])


# Here BOOST is implemented
table(as.numeric(as.factor(sentiment_data[161:454 ])), results[,"LOGITBOOST_LABEL"])
#Confusionmatrix of BOOST to calculate
#Precision ,Recall, Accuracy
#confusionMatrix(table(as.numeric(as.numeric(sentiment_data[161:454])), results[,"LOGITBOOST_LABEL"]))
recall_accuracy(as.numeric(as.factor(sentiment_data[161:454])), results[,"LOGITBOOST_LABEL"])



# formal tests
analytics = create_analytics(container, results)
summary(analytics)

head(analytics@algorithm_summary)
head(analytics@label_summary)
head(analytics@document_summary)

N=3
cross_SVM = cross_validate(container,N,"SVM")
cross_GLMNET = cross_validate(container,N,"GLMNET")
cross_MAXENT = cross_validate(container,N,"MAXENT")
cross_TREE = cross_validate(container,N,"TREE")