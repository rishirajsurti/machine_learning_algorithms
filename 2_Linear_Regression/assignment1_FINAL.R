#find which positions have '?'
#find mean of rest of values for a particular attribute
#replace '?' with mean of that attributecommunities <- read.csv("~/cs5011/communities.data", header=FALSE, na.strings="?") #convert all '?' to 'NA'

communities <- read.csv("~/cs5011/communities.data", header=FALSE, na.strings="?") #convert all '?' to 'NA'

cnames <- read.csv("~/cs5011/comm.names", header=FALSE, sep=" ") #convert all '?' to 'NA'
d = communities #better readability
d.na = !is.na(d) # find positions which are NOT 'NA'

f<-function(x){
  if( sum(is.na(x)) != 0 ){
    x = as.numeric(x);
    m <- mean(x,na.rm=TRUE);
    m<-round(m, 2);
    #print(class(m));
    x[which(is.na(x))] <- m;
    x = as.character(x);
  }
  x;
}

d2 <- apply(d,2,f);
d3 <- d2[,c(1:3)]
d3 <- apply(d3,2,as.numeric);
d4 <- d2[,4];
d5 <- d2[, c(5:128)]
d5 <- apply(d5,2, as.numeric);

d6 <- data.frame(d3,d4,d5, stringsAsFactors=FALSE)
#names(d6) = as.character(cnames[,2]);
dnames = as.character(cnames[,2]);
names(d6)<-cnames[,2];
write.csv(dnames, file="communities_names.csv",row.names=FALSE);
write.csv(d6,file="communities_new.csv",row.names=FALSE)
write.csv(d6, file="communities_new_table.table",row.names=FALSE)
"
View(d6)
for(i in 1:3){
  d2[,i] = as.numeric(d2[,i]);
}
#d2[,4]= sapply(d2[,4], as.character);

for( i in 5:128){
  d2[,i] = as.numeric(d2[,i]);
}
View(d2)
"

