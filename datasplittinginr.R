setwd("F:\\Lectures\\FPMDatamining\\Data")
data1<-read.csv("train.csv")
summary(data1)
table(data1$target)
library(caTools)
#split <- sample.split(data1$target, SplitRatio=0.8)
#train <- mydata[split==T,]
#test <- mydata[split==F,]
split<-data1$target
train<-data1[split==1,]
train1<-data1[split==0,]
c<-seq(1:699810)
num<-sample(c,35924,replace = FALSE)
trainneg<-train1[num,]
trainnegr<-train1[-num,]
n<-dim(trainnegr)[1]
s1<-seq(1:n)
num2<-sample(s1,10000,replace = FALSE)
testneg <- trainnegr[num2,]
c<-seq(1:45924)
num1<-sample(c,35924,replace = FALSE)
trainpos <- tra[num1,]
testpos<-tra[-num1,]
#total<-rbind(train,train)
#write.csv(total, file = "F:\\Lectures\\FPMDatamining\\Data\\sample.csv", row.names = FALSE)
write.csv(trainpos$question_text, file = "F:\\Lectures\\FPMDatamining\\Data\\pos.csv", row.names = FALSE)
write.csv(trainneg$question_text, file = "F:\\Lectures\\FPMDatamining\\Data\\neg.csv", row.names = FALSE)
write.csv(testpos$question_text, file = "F:\\Lectures\\FPMDatamining\\Data\\testpos.csv", row.names = FALSE)
write.csv(testneg$question_text, file = "F:\\Lectures\\FPMDatamining\\Data\\testneg.csv", row.names = FALSE)
