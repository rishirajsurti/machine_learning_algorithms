#load libraries
library(ISLR)
library(MASS)
#Note: plot3d requires rgl
#if rgl is not installed, uncomment following line
#install.packages("rgl")
library(rgl)
library(ggplot2)

train<-read.csv('train.csv',header=FALSE)
train.labels<-read.csv('train_labels.csv',header=FALSE)
View(train)
View(train.labels)
#names(train)<-c("F1","F2","F3")
##as no names in data..
##assigned names as Feature 1,2,3

test<-read.csv('test.csv',header=FALSE)
test.labels<-read.csv('test_labels.csv',header=FALSE)
View(test.labels)

"
Motive: 
We have training labels.
use Lda to fit to training data.
predict class labels for test data.
compare with actual test labels and report accuracy.
"

#lda.fit = lda(train.labels$V1~train$V1+train$V2+train$V3)
lda.fit = lda(train.labels$V1~V1+V2+V3, data = train)
lda.fit
plot(lda.fit)

lda.pred = predict(lda.fit, test)
#predicting based on test data
summary(lda.pred)
View(lda.pred)
names(lda.pred)
View(lda.pred$class)
table(lda.pred$class, test.labels$V1)
mean(lda.pred$class==test.labels$V1)


plot3d(train$V1, train$V2, train$V3)
