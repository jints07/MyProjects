rm(list=ls());
#Simulate 100 X
x = runif(100,0,10);
X = t(rbind(rep(1,100), x, 1/x));
bhat = matrix(nrow=1000,ncol=3);
beta = matrix(c(10,-1,-25),nrow=3);

for (i in 1:1000){
    y = X%*%beta + rnorm(100,0,5);
    bhat[i,] = t(inv(t(X)%*%X)%*%t(X)%*%y);
}
mean_bhat = colMeans(bhat);

## Make plots
jpeg('hw4_beta_01.jpg');
hist(bhat[,1],breaks=20,xlab = 'beta_01',main = "Frequency of beta_01")
dev.off();
jpeg('hw4_beta_02.jpg');
hist(bhat[,2],breaks=20,xlab = 'beta_02',main = "Frequency of beta_02")
dev.off()
jpeg('hw4_beta_03.jpg');
hist(bhat[,3],breaks=20,xlab = 'beta_03',main = "Frequency of beta_03")
dev.off()
