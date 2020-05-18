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
