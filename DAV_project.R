install.packages("jsonlite")
install.packages("readr")
install.packages("tm")
install.packages("SnowballC")
install.packages("caTools")
install.packages("e1071")
install.packages("caret")
install.packages("text2vec")
install.packages("Matrix")

#reading the dataset
library(readr)
column_names <- c("target", "id", "date", "flag", "user", "text")
twitter_data <- read_csv("training.1600000.processed.noemoticon.csv", col_names = column_names, locale = locale(encoding = "Latin1"))

#preprocessing and stemming
library(tm)
library(SnowballC)

clean_text <- function(text) {
  text <- gsub("[^a-zA-Z]", " ", text)
  text <- tolower(text)
  text <- removeWords(text, stopwords("en"))
  text <- wordStem(text, language = "en")
  text <- stripWhitespace(text)
  return(text)
}

twitter_data$stemmed_content <- sapply(twitter_data$text, clean_text)
text


#splitting dataset
library(caTools)
set.seed(2)
split <- sample.split(twitter_data$target, SplitRatio = 0.8)
train_set <- subset(twitter_data, split == TRUE)
test_set <- subset(twitter_data, split == FALSE)


#vectorization
library(text2vec)
tokens <- word_tokenizer(train_set$stemmed_content)
it <- itoken(tokens, progressbar = FALSE)
vectorizer <- vocab_vectorizer(create_vocabulary(it))
dtm_train <- create_dtm(it, vectorizer)

#model training and acuracy
library(caret)

train_set$target <- factor(train_set$target)
model <- train(target ~ ., data = data.frame(as.matrix(dtm_train), target=train_set$target), method = "glm", family = "binomial")

# Evaluate
predictions <- predict(model, newdata = data.frame(as.matrix(dtm_test)))
confusionMatrix(predictions, test_set$target)


#saving and loading model
saveRDS(model, "trained_model.rds")
loaded_model <- readRDS("trained_model.rds")


