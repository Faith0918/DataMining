#Read data
sms_raw = read.delim("C:/Sources/SMSSpamCollection.txt",quote = "",header = F, encoding = "UTF-8")
glimpse(sms_raw)

#factorize type
sms_raw$V1 = factor(sms_raw$V1)
print((levels(sms_raw$V1)))
summary(sms_raw)

#Data preparation - processing text data for analysis
install.packages("tm")
library(tm)
sms_corpus <- Corpus(VectorSource(sms_raw$V2))
print(sms_corpus)
inspect(sms_corpus[1:3])
corpus_clean <- tm_map(sms_corpus,tolower)
corpus_clean <- tm_map(corpus_clean,removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)
inspect(corpus_clean[1:3])
sms_dtm <- DocumentTermMatrix(corpus_clean)

#Data preparatation - creating traing and test datasets
#split the raw data frame
sms_raw_train <- sms_raw[1:4169, ]
sms_raw_test <- sms_raw[4170:5574, ]

#document-term matrix
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5574, ]

#corpus
sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test <- corpus_clean[4170:5574]

prop.table(table(sms_raw_train$V1))
prop.table(table(sms_raw_test$V1))

#visualize text data - word clouds
install.packages("wordcloud")
library(wordcloud)
wordcloud(sms_corpus_train, min.freq = 40, random.order = FALSE)
spam <- subset(sms_raw_train, V1 == "spam")
ham <- subset(sms_raw_train, V1 == "ham")
wordcloud(spam$V2, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$V2, max.words = 40, scale = c(3, 0.5))

#Data preparation – creating indicator features for frequent words
#find terms appearing at least 5 times in the sms_dtm_train matrix:
findFreqTerms(sms_dtm_train, 5)
Dictionary <- function(x) {
  if( is.character(x) ) {
    return (x)
  }
  stop('x is not a character vector')
}

sms_dict <- Dictionary(findFreqTerms(sms_dtm_train, 5))

sms_train <- DocumentTermMatrix(sms_corpus_train,
                                list(dictionary = sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_test,
                               list(dictionary = sms_dict))

convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}
sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_test, MARGIN = 2, convert_counts)

#Step3 - training a model on the data
install.packages("e1071")
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_raw_train$V1)

#Step4 - evaluating model performance
sms_test_pred <- predict(sms_classifier, sms_test)
install.packages("gmodels")
library(gmodels)
CrossTable(sms_test_pred, sms_raw_test$V1,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))

#Step 5 – improving model performance
sms_classifier2 <- naiveBayes(sms_train, sms_raw_train$V1,
                              laplace = 0.2)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_raw_test$V1,
           prop.chisq = FALSE, prop.t = FALSE, 
           dnn = c('predicted', 'actual'))
