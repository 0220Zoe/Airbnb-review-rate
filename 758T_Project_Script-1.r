df<-read.csv("dc airbnb_cleaned.csv")
df<-na.omit(df)

#text mining on amenity-------------------------------------------
library(e1071)
library(SparseM)
library(tm)

wordlist_vector<-as.vector(df$amenities)
wordlist_source<-VectorSource(wordlist_vector)
wordlist_corpus<-Corpus(wordlist_source)

wordlist_corpus <- tm_map(wordlist_corpus,content_transformer(stripWhitespace));
wordlist_corpus <- tm_map(wordlist_corpus,content_transformer(tolower));
wordlist_corpus <- tm_map(wordlist_corpus, content_transformer(removeWords),stopwords("english"));
wordlist_corpus <- tm_map(wordlist_corpus,content_transformer(removePunctuation));
wordlist_corpus <- tm_map(wordlist_corpus,content_transformer(removeNumbers));

removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
wordlist_corpus <- tm_map(wordlist_corpus, content_transformer(removeNumPunct))

removeURL <- function(x) gsub("http[[:alnum:]]*", "", x)
wordlist_corpus <- tm_map(wordlist_corpus, content_transformer(removeURL))

tdm1<-TermDocumentMatrix(wordlist_corpus)
tdm1<-removeSparseTerms(tdm1,0.95)

inspect(tdm1)
wordlist_matrix <- t(tdm1)
wordlist <- data.frame(as.matrix(wordlist_matrix))

#
matrix_hclust<-as.matrix(tdm1)
matrix_hclust_distance<-dist(scale(matrix_hclust))
model5<-hclust(matrix_hclust_distance,method='average')
plot(model5, cex=0.9, hang=-1,
     main="Word Cluster Dendrogram")
rect.hclust(model5, k=5)
(groups <- cutree(model5, k=5))
#
library(wordcloud)
library(RColorBrewer)
matrix_cloud<-as.matrix(tdm1)
frequency_word_5<-sort(rowSums(matrix_cloud),decreasing = T)
pal<-brewer.pal(9, "BuGn")
pal <- pal[-(1:4)]
wordcloud(words = names(frequency_word_5), freq = frequency_word_5, min.freq = 1,
          random.order = F,colors = pal)
#
frequency_word<-findFreqTerms(tdm1,lowfreq = 15)
frequency_word_dataframe<-rowSums(as.matrix(tdm1))
frequency_word_dataframe<-subset(frequency_word_dataframe,frequency_word_dataframe>=15)
frequency_word_dataframe<-data.frame(term=names(frequency_word_dataframe),freq=frequency_word_dataframe)
library(ggplot2)
ggplot(frequency_word_dataframe, aes(x = term, y = freq)) + geom_bar(stat = "identity") +
  xlab("Terms") + ylab("Count") + coord_flip()
#we dont want amenities with less frequency, so set the freq to 5000.
frequency_word<-findFreqTerms(tdm1,lowfreq = 5000)
frequency_word_dataframe<-rowSums(as.matrix(tdm1))
frequency_word_dataframe<-subset(frequency_word_dataframe,frequency_word_dataframe>=5000)
frequency_word_dataframe<-data.frame(term=names(frequency_word_dataframe),freq=frequency_word_dataframe)
library(ggplot2)
ggplot(frequency_word_dataframe, aes(x = term, y = freq)) + geom_bar(stat = "identity") + ggtitle('Frequency of top 11 amenities') + 
  xlab("Terms") + ylab("Count") + coord_flip()
#from the result, we can extract some usefull and meaningful amenity
#workspace, wifi, washer,smoke,shampoo, parking, kitchen,heating,dryer, conditioning, coffee, alarm
#so we just use this amenities 

library(tidyverse)
wordlist%>%#if i can use this function to add these column to df
  select(workspace, wifi, washer,smoke,shampoo, parking, kitchen,heating,dryer, conditioning, coffee, alarm)

df$workspace<-wordlist$workspace
df$wifi<-wordlist$wifi
df$washer<-wordlist$washer
df$smoke<-wordlist$smoke
df$shampoo<-wordlist$shampoo
df$parking<-wordlist$parking
df$kitchen<-wordlist$kitchen
df$heating<-wordlist$heating
df$dryer<-wordlist$dryer
df$conditioning<-wordlist$conditioning
df$coffee<-wordlist$coffee
df$alarm<-wordlist$alarm
df$amenities<-NULL

#deal with host_years-------------------------------------------
str(df)
df$host_since <- as.Date(df$host_since, "%m/%d/%Y")
df$host_years <- round((Sys.Date() - df$host_since)/365,0)
df$host_since<-NULL
df$host_years<-as.numeric(df$host_years)

#deal with host_location-------------------------------------------
library(data.table)
df$host_inside_dc <- ifelse(df$host_location %like% "District of Columbia", 1, 0)
df$host_location<-NULL

#deal with room_type-------------------------------------------
df$room_type_binary<-ifelse(df$room_type=='Shared room',0,ifelse(df$room_type=='Hotel room',1,ifelse(df$room_type=='Private room',2,3)))
df$room_type_binary<-factor(df$room_type_binary)
df$room_type<-NULL

#change the value of variable into proper type-------------------------------------------
str(df)
df$host_is_superhost<-factor(df$host_is_superhost)
df$host_identity_verified<-factor(df$host_identity_verified)
df$instant_bookable<-factor(df$instant_bookable)
df$price<-as.numeric(gsub("[\\$,]", "", df$price))
str(df)

#change amenity
df$workspace<-ifelse(df$workspace==0,0,1)
df$wifi<-ifelse(df$wifi==0,0,1)
df$washer<-ifelse(df$washer==0,0,1)
df$smoke<-ifelse(df$smoke==0,0,1)
df$shampoo<-ifelse(df$shampoo==0,0,1)
df$parking<-ifelse(df$parking==0,0,1)
df$kitchen<-ifelse(df$kitchen==0,0,1)
df$heating<-ifelse(df$heating==0,0,1)
df$dryer<-ifelse(df$dryer==0,0,1)
df$conditioning<-ifelse(df$conditioning==0,0,1)
df$coffee<-ifelse(df$coffee==0,0,1)
df$alarm<-ifelse(df$alarm==0,0,1)

#write the file out
write.csv(df,'final data.csv')

#Modeling
str(df)

#define a cutoff for the review rate, use the average number.
summary(df$review_scores_rating)
average_rate<-mean(df$review_scores_rating,na.rm = T)
df$beyond_average<-ifelse(df$review_scores_rating>average_rate,1,0)

#showing distribution of dependent variable beyond_average
barplot(table(df$beyond_average), main = 'Distribution of dependent variable(beyond_average)', ylab = 'Frequency' , ylim = c(0, 4500))

#partition dataset
set.seed(12345)
index <- createDataPartition(df$beyond_average, p=0.8, list=FALSE)
df_train<-df[index,]
df_test<-df[-index,]

######################################################################################################################
#linear classification
model_linear<-lm(review_scores_rating~.,data=df_train[,-24])
pred_linear<-predict(model_linear,newdata = df_test[,-c(7,24)])
pred_linear_class<-ifelse(pred_linear>average_rate,1,0)
CN_linear<-table(pred_linear_class,df_test$beyond_average)
acc_linear<-(CN_linear[1,1]+CN_linear[2,2])/sum(CN_linear)
# ROC Curves of linear regression
par(pty="s")
library(pROC)
roc_rose_L <- plot(roc(df_test$beyond_average, pred_linear), 
                 print.auc = TRUE, col = "blue",legacy.axes=T,main="ROC Curve of Linear Model")
#liftchart  of linear regression
actual <- as.numeric(df_test$beyond_average)-1 

df1_L <- data.frame(pred_linear,actual,df_test$beyond_average)
df1S_L <- df1_L[order(-pred_linear),] ## Sorted by probability (descending)
#
df1S_L$Gains <- cumsum(df1S_L$actual)
plot(df1S_L$Gains,type="n",main="Lift Chart of Linear Model",xlab="Number of Cases",ylab="Cumulative Success")
lines(df1S_L$Gains)
abline(0,sum(df1S_L$actual)/nrow(df1S_L),lty = 2, col="red")
######################################################################################################################

######################################################################################################################
#logistic classification
model_logistic<-glm(beyond_average~.,data=df_train[,-7],family='binomial')
pred_logistic_prob<-predict(model_logistic,newdata = df_test[,-c(7,24)],type='response')
cutoff=0.6
pred_logistic_class<-ifelse(pred_logistic_prob>cutoff,1,0)
CN_logistic<-table(df_test$beyond_average,pred_logistic_class)
acc_logistic<-(CN_logistic[1,1]+CN_logistic[2,2])/sum(CN_logistic)
#ROC Curves of logistic classification
roc_rose_Lm <- plot(roc(df_test$beyond_average, pred_logistic_prob), 
                 print.auc = TRUE, col = "blue",legacy.axes=T,main="ROC Curve of Logistic Model")
#liftchart of logistic classification
actual <- as.numeric(df_test$beyond_average)-1 
df1_LG <- data.frame(pred_logistic_prob,actual,df_test$beyond_average)
df1S_LG <- df1_LG[order(-pred_logistic_prob),] ## Sorted by probability (descending)
#
df1S_LG$Gains <- cumsum(df1S_LG$actual)
plot(df1S_LG$Gains,type="n",main="Lift Chart of Logistic Model",xlab="Number of Cases",ylab="Cumulative Success")
lines(df1S_LG$Gains)
abline(0,sum(df1S_LG$actual)/nrow(df1S_LG),lty = 2, col="red")
######################################################################################################################


######################################################################################################################
#KNN
library(fastDummies)
library(caret)
library(class)
#make sure no factor, all should be dummies variables
df_train_KNN<-df_train
df_validation_KNN<-df_test

df_train_KNN<-fastDummies::dummy_cols(df_train_KNN,select_columns = c('host_is_superhost','host_identity_verified','instant_bookable','room_type_binary'),remove_selected_columns = T)
df_validation_KNN<-fastDummies::dummy_cols(df_validation_KNN,select_columns = c('host_is_superhost','host_identity_verified','instant_bookable','room_type_binary'),remove_selected_columns = T)

fun <- function(x){ 
  a <- mean(x) 
  b <- sd(x) 
  (x - a)/(b) 
} 
df_train_KNN[,-c(5,20)]<-apply(df_train_KNN[,-c(5,20)], 2, fun)
df_validation_KNN[,-c(5,20)]<-apply(df_validation_KNN[,-c(5,20)], 2, fun)

kmax <- 30
acc_KNN_train <- rep(0,kmax)
acc_KNN_validation <- rep(0,kmax)
for (i in 1:kmax){
  prediction <- knn(df_train_KNN[,-c(5,20)], df_train_KNN[,-c(5,20)],df_train_KNN[,20], k=i)
  prediction2 <- knn(df_train_KNN[,-c(5,20)], df_validation_KNN[,-c(5,20)],df_train_KNN[,20], k=i)
  # The confusion matrix for training data is:
  CM1 <- table(prediction, df_train_KNN$beyond_average)
  # The training error rate is:
  acc_KNN_train[i] <- (CM1[1,1]+CM1[2,2])/sum(CM1)
  # The confusion matrix for validation data is: 
  CM2 <- table(prediction2, df_validation_KNN$beyond_average)
  acc_KNN_validation[i] <- (CM2[1,1]+CM2[2,2])/sum(CM2)
}

plot(c(1,kmax),c(0,1),type="n", xlab="k",ylab="Accuracy")
lines(acc_KNN_train,col="red")
lines(acc_KNN_validation,col="blue")

legend(10, 0.4, c("Training","Validation"),lty=c(1,1), col=c("red","blue"))
z <- which.max(acc_KNN_validation)
max(acc_KNN_validation)
cat("Minimum Validation Error k:", z)

#KNN ROC
train_input_KNN <- df_train_KNN[,-c(5,20)]
train_output_KNN <- df_train_KNN[,20]
validate_input_KNN <- df_validation_KNN[,-c(5,20)]
prediction_KR <- knn(train_input_KNN, validate_input_KNN,train_output_KNN, k=25, prob=T)
predicted.probability.knn <- attr(prediction_KR, "prob")
Predicted_class <- knn(train_input_KNN, validate_input_KNN,train_output_KNN, k=25)
predicted.probability.knn <- ifelse(Predicted_class ==1, predicted.probability.knn, 1-predicted.probability.knn)

par(pty="s")
library(pROC)
roc_rose_KNN <- plot(roc(df_validation_KNN$beyond_average, predicted.probability.knn), 
                     print.auc = TRUE, col = "blue",legacy.axes=T,main="ROC Curve of KNN Model")

#KNN lift
prediction_KL <- knn(train_input_KNN, validate_input_KNN,train_output_KNN, k=25, prob=T)
#
predicted.probability.knnL <- attr(prediction_KL, "prob")
# 
# This (unfortunately returns the proportion of votes for the winning class - P(Success))
#
predicted.probability.knnL <- ifelse(prediction_KL ==1, predicted.probability.knnL, 1-predicted.probability.knnL)
#
df_KnnL <- data.frame(prediction_KL, predicted.probability.knnL,df_validation_KNN$beyond_average)
# When prediction is 1, we will use predicted.probability; else use 1-predicted.probability
df_KnnL1 <- df_KnnL[order(-predicted.probability.knnL),]
df_KnnL1$Gains <- cumsum(df_KnnL1$df_validation_KNN.beyond_average)
plot(df_KnnL1$Gains,type="n",main="Lift Chart of KNN Model",xlab="Number of Cases",ylab="Cumulative Success")
lines(df_KnnL1$Gains)
abline(0,sum(df_KnnL1$df_validation_KNN.beyond_average)/nrow(df_KnnL1),lty = 2, col="red")
######################################################################################################################

######################################################################################################################
# NaiveBayes
df_train$beyond_average<-factor(df_train$beyond_average)
df_test$beyond_average<-factor(df_test$beyond_average)
model_NB<-naiveBayes(beyond_average~.,data=df_train[,-7])
pred_NB<-predict(model_NB,newdata=df_test[,-c(7,24)])
CM_NB<-table(df_test$beyond_average,pred_NB)
acc_NB<-(CM_NB[1,1]+CM_NB[2,2])/sum(CM_NB)

#NaiveBayes ROC
predicted.probabilityNB <- predict(model_NB, newdata = df_test[-7], type="raw")
predicted.probability.NB <- predicted.probabilityNB[,2]
roc_rose_NB <- plot(roc(df_test$beyond_average, predicted.probability.NB), print.auc = TRUE, 
                    col = "blue", print.auc.y = .4, legacy.axes=T,main="ROC Curve of NaiveBayes Model")

#NaiveBayes lift
predicted.probability_NBL <- predict(model_NB, newdata = df_test, type="raw")
#
# The first column is for class 0, the second is class 1
score_NB <- as.numeric(df_test$beyond_average)
# To turn into dummy variable to get lift chart
prob_NB <- predicted.probability_NBL[,2] # Predicted probability of success
df_NB <- data.frame(score_NB, prob_NB)
df_NB1 <- df_NB[order(-prob_NB),]
df_NB1$Gains <- cumsum(df_NB1$score_NB)
plot(df_NB1$Gains,type="n",main="Lift Chart of NaiveBayes Model",xlab="Number of Cases",ylab="Cumulative Success")
lines(df_NB1$Gains)
abline(0,sum(df_NB1$score_NB)/nrow(df_NB1),lty = 2, col="red")
######################################################################################################################

######################################################################################################################
#TREE#
library(tree)
model_full_tree<-tree(beyond_average~.,data=df_train[,-7])
pred_full_tree<-predict(model_full_tree,newdata=df_test[,-c(7,24)]);
pred_full_tree_class<-ifelse(pred_full_tree[,2]>0.7,1,0)
CM_full_tree<-table(df_test$beyond_average,pred_full_tree_class)
acc_full_tree<-(CM_full_tree[1,1]+CM_full_tree[2,2])/sum(CM_full_tree)
summary(model_full_tree)

set.seed(12345)
cv_tree<-cv.tree(model_full_tree,FUN=prune.misclass)
plot(cv_tree$size,cv_tree$dev,type='b')
cv_tree

# ROC Curves of TREE
roc_rose_Tree <- plot(roc(df_test$beyond_average, pred_full_tree[,2]), 
                 print.auc = TRUE, col = "blue",legacy.axes=T,main="ROC Curve of Tree Model")
#liftchart of TREE
actual <- as.numeric(df_test$beyond_average)-1 
df1 <- data.frame(pred_full_tree[,2],actual,df_test$beyond_average)
df1S <- df1[order(-pred_full_tree[,2]),] ## Sorted by probability (descending)
df1S$Gains <- cumsum(df1S$actual)
plot(df1S$Gains,type="n",main="Lift Chart of Tree Model",xlab="Number of Cases",ylab="Cumulative Success")
lines(df1S$Gains)
abline(0,sum(df1S$actual)/nrow(df1S),lty = 2, col="red")
######################################################################################################################

######################################################################################################################
#random forest#
set.seed(12345)
library(randomForest)
model_randomForest<-randomForest(beyond_average~.,data=df_train[,-7],mtry=5,importance=TRUE)
pred_randomForest<-predict(model_randomForest,newdata=df_test[,-c(7,24)])
CM_randomForest<-table(df_test$beyond_average,pred_randomForest)
acc_randomForest<-(CM_randomForest[1,1]+CM_randomForest[2,2])/sum(CM_randomForest)
#check the importance of variables
importance_otu.scale <- data.frame(importance(model_randomForest, scale = TRUE), check.names = FALSE)
importance_otu.scale

#Bar chart of random forest variables
#aim to find independent variables influence the dependent variables most
rf_geni <- tibble::rownames_to_column(data.frame(model_randomForest$importance[,4]), "Variable")
colnames(rf_geni)[2] <- "MeanDecreaseGini"
rf_geni
#
#Bar chart of amentities variables
rf_geni[8:19,]%>%
  ggplot(aes(x= Variable,y=MeanDecreaseGini))+geom_col()+ ylab("MeanDecreaseGini") + ggtitle('MeanDecreaseGini of amentities variables') +
  theme_classic()+
  theme(axis.text.x = element_text(face="bold", angle=90))+
  theme(axis.title.x =element_blank())
#
#Bar chart of random forest variables
rf_geni%>%
  ggplot(aes(x= Variable,y=MeanDecreaseGini), axis.title.y = )+geom_col() + ylab("MeanDecreaseGini") + ggtitle('MeanDecreaseGini of variables') +
  theme_classic()+
  theme(axis.text.x = element_text(face="bold", angle=90),
        axis.title.x =element_blank())
#

# ROC Curves of random forest
predicted.probability_rf <- predict(model_randomForest, newdata=df_test[,-c(7,24)],type = "prob") 
roc_rose_randomForest <- plot(roc(df_test$beyond_average, predicted.probability_rf[,2]), 
                 print.auc = TRUE, col = "blue",legacy.axes=T, main="ROC Curve of Random Forest Model")
#liftchart of random forest
actual <- as.numeric(df_test$beyond_average)-1 
df1 <- data.frame(pred_randomForest,actual,df_test$beyond_average)
df1S <- df1[order(-pred_randomForest),] ## Sorted by probability (descending)
#
df1S$Gains <- cumsum(df1S$actual)
plot(df1S$Gains,type="n",main="Lift Chart of Random Forest Model",xlab="Number of Cases",ylab="Cumulative Success")
lines(df1S$Gains)
abline(0,sum(df1S$actual)/nrow(df1S),lty = 2, col="red")
######################################################################################################################


######################################################################################################################
#xgboost
set.seed(12345)
df_train_xgboost<-df_train
df_test_xgboost<-df_test

df_train_xgboost<-fastDummies::dummy_cols(df_train_xgboost,select_columns = c('host_is_superhost','host_identity_verified','instant_bookable','room_type_binary'),remove_selected_columns = T)
df_test_xgboost<-fastDummies::dummy_cols(df_test_xgboost,select_columns = c('host_is_superhost','host_identity_verified','instant_bookable','room_type_binary'),remove_selected_columns = T)
df_train_xgboost$beyond_average<-as.numeric(df_train_xgboost$beyond_average) -1
df_test_xgboost$beyond_average<-as.numeric(df_test_xgboost$beyond_average) -1
str(df_train_xgboost)

library(xgboost)
model_xgboost<-xgboost(as.matrix(df_train_xgboost[,-c(5,20)]),df_train_xgboost$beyond_average,max.depth = 2,eta = 1, nround = 5, objective = "binary:logistic")
pred_xgboost_prob<-predict(model_xgboost,as.matrix(df_test_xgboost[,-c(5,20)]))
pred_xgboost<-ifelse(pred_xgboost_prob>0.5,1,0)
CM_xgboost<-table(df_test_xgboost$beyond_average,pred_xgboost)
(acc_xgboost<-(CM_xgboost[1,1]+CM_xgboost[2,2])/sum(CM_xgboost))

# ROC Curves of xgboost
roc_rose_xgboost <- plot(roc(df_test$beyond_average, pred_xgboost_prob), 
                 print.auc = TRUE, col = "blue", legacy.axes=T,main="Lift Chart of Xgboost Model")
#liftchart of xgboost
actual <- as.numeric(df_test$beyond_average)-1 
df1 <- data.frame(pred_xgboost_prob,actual,df_test$beyond_average)
df1S <- df1[order(-pred_xgboost_prob),] ## Sorted by probability (descending)
#
df1S$Gains <- cumsum(df1S$actual)
plot(df1S$Gains,type="n",main="Lift Chart of Xgboost Model",xlab="Number of Cases",ylab="Cumulative Success")
lines(df1S$Gains)
abline(0,sum(df1S$actual)/nrow(df1S),lty = 2, col="red")
######################################################################################################################


######################################################################################################################
#ROC on one graph
roc_rose_xgboost <- plot(roc(df_test$beyond_average, pred_xgboost_prob), print.auc = TRUE,
                         print.auc.y = .4, col = "blue", legacy.axes=T,main="ROC Curve")
roc_rose_randomForest <- plot(roc(df_test$beyond_average, predicted.probability_rf[,2]), print.auc = TRUE,
                              print.auc.y = .5, col = "red",add = TRUE,legacy.axes=T)
roc_rose_Tree <- plot(roc(df_test$beyond_average, pred_full_tree[,2]), print.auc = TRUE,
                       col = "green",print.auc.y = .1, add = TRUE,legacy.axes=T)
roc_rose_NB <- plot(roc(df_test$beyond_average, predicted.probability.NB), print.auc = TRUE,
                     col = "purple", print.auc.y = 0,add = TRUE,legacy.axes=T)
roc_rose_KNN <- plot(roc(df_validation_KNN$beyond_average, predicted.probability.knn), print.auc = TRUE,
                      col = "orange",print.auc.y = .2,add = TRUE,legacy.axes=T)
roc_rose_Lm <- plot(roc(df_test$beyond_average, pred_logistic_prob), print.auc = TRUE,
                    col = "black",add = TRUE, print.auc.y = .3,legacy.axes=T)

(acc_all <- data.frame(acc_linear, acc_logistic, max(acc_KNN_validation), acc_randomForest, acc_xgboost, acc_NB, acc_full_tree))
