## emotion mining

install.packages("syuzhet")
library("syuzhet")
library(readr)
install.packages("tidytext")
install.packages("textdata")
library(tidytext)
library(textdata)
library(dplyr)

nrc_lexicon <- get_sentiments("nrc")

id <- rownames(df2)
df2 <- cbind(id=id, df2)
summary(df2)
glimpse(df2,class)
answer = df2


unnested_6 <- answer %>%
  unnest_tokens(word, answer) %>%  # unnest the words
  left_join(nrc_lexicon) %>%     # join with the lexicon to have sentiments
  left_join(answer)

View(unnested_6)

table_sentiment <- table(unnested_6$id, unnested_6$sentiment)
table_sentiment


table_sentiment <- as.data.frame.matrix(table_sentiment)
sum(is.na(table_sentiment))

table_sentiment_1=transform(table_sentiment, sum=rowSums(table_sentiment))
View(table_sentiment_1)

count(distinct(table_sentiment_1$sum))
table(table_sentiment_1$sum)#it is showing 0 having count 111048

sent_0=subset(table_sentiment_1, sum == 0)
View(sent_0)
sentiment_value <- subset(table_sentiment_1, sum > 0)
View(sentiment_value)
write.csv(sentiment_value, file = "D:\\data_science _data _set\\NLP_project\\resultssentiment_value.csv")


df_sentiment[df_sentiment>1]<-1


