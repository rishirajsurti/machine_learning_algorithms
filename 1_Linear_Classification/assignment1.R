?mvrnorm
sigma<-matrix(rep(c(10,5,7),3),c(3,3),byrow=TRUE)
sigma<-matrix(c(1,0,0,1),2,2,byrow=TRUE)

sigma
s<-mvrnorm(100,rep(0,2),sigma)
#s
s1<-sort(s[,1])
s2<-sort(s[,2])

f<-function(x,y){dnorm(x)*dnorm(y)};
s3<-outer(s1,s2,f)
persp3d(s1,s2,s3,col="lightblue")
image(s1,s2,s3)
mean(s1)
mean(s2)
cov(s)
?cov

sd(rnorm(10))
mean(rnorm(10))

sigma<-matrix(rep(dnorm(rnorm(10)),10),10,10)
sigma<-matrix(rep(rnorm(10, mean=2,sd=0.01),10),10,10)

sigma
s<-mvrnorm(10,rep(0,10),sigma)
?mvrnorm

genPositiveDefMat("eigen",dim=5)
genPositiveDefMat("eigen",dim=10)

#clusplot
x <- rbind(cbind(rnorm(10,0,0.5), rnorm(10,0,0.5)),
           cbind(rnorm(15,5,0.5), rnorm(15,5,0.5)))
clusplot(pam(x, 2))
## add noise, and try again :
x4 <- cbind(x, rnorm(25), rnorm(25))
clusplot(pam(x4, 2))

#ploting mvr
s2=s
s2=apply(s2,2,sort)
View(s2)
s2.dnorm=apply(s2,2,dnorm)
View(s2.dnorm)
sdp<-apply(s2.dnorm,1,prod)
View(sdp)
persp3d(seq(1,10,by=0.01),seq(1,10,by=0.01),sdp)
#sweep
A <- array(1:24, dim = 4:2)
A
## no warnings in normal use
sweep(A, 1, 5, FUN="*")
(A.min <- apply(A, 1, min))  # == 1:4
sweep(A, 1, A.min)
sweep(A, 1:2, apply(A, 1:2, median))

## warnings when mismatch
sweep(A, 1, 1:3)  # STATS does not recycle
sweep(A, 1, 6:1)  # STATS is longer

## exact recycling:
sweep(A, 1, 1:2)  # no warning
sweep(A, 1, as.array(1:2))  # warning



###FINAL
require(clusterGeneration)
require(cluster)
#generate a positive definite covariance matrix
sigma<-genPositiveDefMat(dim=10)
#print(sigma)
#sigma is a list: eigen values and covariance matrix
#sigma$Sigma contains cov matrix
#class(sigma$Sigma)

#class1
s1<-mvrnorm(n=1000, mu=rnorm(10, mean=0, sd=1),sigma$Sigma)
#mvrnorm generates samples from multivariate gaussian dist.
#n=no. of samples
#mu=mean of gaussians, 
#here: means are generated from a 
#normal distribution of mean=0, sd=1, hence class centroid in range of (-1,1);
class<-rep(1,1000)
#View(s1)
ds1<-(cbind(s1,class))
#View(ds1)
clusplot(pam(s1,1))

#class2
s2<-mvrnorm(n=1000, mu=rnorm(10, mean=2, sd=1),sigma$Sigma)
#now choosing means from a gaussian mean=3, sd=1
#tune the 'mean' in above expression to vary distance between class centroids


class<-rep(2,1000)
#View(s2)
ds2<-(cbind(s2,class))
#View(ds2)
clusplot(pam(rbind(ds1,ds2),2))

"ds.final=rbind(ds1,ds2)
#View(ds.final)
ds.final=data.frame(ds.final)
write.csv(ds.final, file=""DS1.csv"",row.names=FALSE)
"

samp1<-sort(sample(c(1:1000),600))
View(ds2)
View(samp1)
ds1.train<-ds1[samp1,]
ds1.test<-ds1[-samp1,]
#View(ds1.test);View(ds1.train); dim(ds1.train); names(ds1.train)

samp2<-sort(sample(c(1:1000),600))
View(ds2.test)

ds2.train<-ds2[samp2,]
ds2.test<-ds2[-samp2,]
View(ds2.test)

ds.train=rbind(ds1.train,ds2.train)
View(ds.train)
ds.test=rbind(ds1.test,ds2.test)
View(ds.test)
write.csv(ds.train, file="DS1_train.csv",row.names=FALSE)
write.csv(ds.test, file="DS1_test.csv",row.names=FALSE)
