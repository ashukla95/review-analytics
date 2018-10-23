setwd("C:/Users/Samsung/Documents/R")
library(ggplot2)

senti <- read.csv("sentiments.csv",sep = ",", stringsAsFactors = FALSE)

ggplot(data = senti, aes(x = class))+
  geom_bar()

#setwd("C:\Users\Samsung\Desktop\final classification python")
ggsave("sentimentplot.png",path = "C:/Users/Samsung/Desktop/final_classification_python",width=20, height=20,units = "cm")