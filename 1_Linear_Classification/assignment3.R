

###FINAL
library(class)
ds<-read.csv("DS1.csv",header=TRUE)

ds.train<-read.csv("DS1_train.csv",header=TRUE)
ds.test<-read.csv("DS1_test.csv",header=TRUE)

"
View(ds); names(ds)
ds1<-ds[which(ds$class==1),]
ds2<-ds[which(ds$class==2),]

View(ds1); View(ds2);
#dim(ds1)[2] # rows of dataset, here we know
#split into train-test data
s1<-sort(sample(c(1:1000),600))

ds1.train<-ds1[s1,]
ds1.test<-ds1[-s1,]
#View(ds1.test);View(ds1.train); dim(ds1.train); names(ds1.train)

s2<-sort(sample(c(1:1000),600))
ds2.train<-ds2[s2,]
ds2.test<-ds2[-s2,]
View(ds2.test)

#
ds.train<-rbind(ds1.train, ds2.train)
View(ds.train); names(ds.train); dim(ds.train)
ds.test<-rbind(ds1.test, ds2.test)
"
ds.train.data=ds.train[,-11]
ds.train.class=ds.train[,11]

ds.test.data= ds.test[,-11];
ds.test.class=ds.test[,11]

knn.pred=knn(ds.train.data, ds.test.data, ds.train.class,k=10)
table(knn.pred, ds.test.class)
mean(knn.pred==ds.test.class)
