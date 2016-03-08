"
ViolentCrimesPerPop: total number of violent crimes per 100K popuation
(numeric - decimal) GOAL attribute (to be predicted) 
"
comm <- read.csv("~/cs5011/2_Linear_Regression/communities_new.csv", header=TRUE);
attach(comm);
#View(comm)
lm.fit = lm(ViolentCrimesPerPop~., data = comm);
sink("regression_fit.txt");
summary(lm.fit);
sink()
