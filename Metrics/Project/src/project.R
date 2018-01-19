# Read CSV data
raw_asm<-read.csv("ASM_Millions.csv",header=TRUE,stringsAsFactors=FALSE);
raw_load<-read.csv("LoadFactor.csv",header=TRUE,stringsAsFactors=FALSE);
raw_pyield<-read.csv("Passyield_cents.csv",header=TRUE,stringsAsFactors=FALSE);
raw_fuelasm<-read.csv("fuel_asm_usd.csv",header=TRUE,stringsAsFactors=FALSE);
raw_labor<-read.csv("laborasm_cents.csv",header=TRUE,stringsAsFactors=FALSE);
raw_nonlabor<-read.csv("nonlaborasm_cents.csv",header=TRUE,stringsAsFactors=FALSE);


# Store date from tables into matrices
library(stringr)
gregexp <- "[[:digit:].]+"
tabnames=c("raw_asm","raw_load","raw_pyield","raw_fuelasm","raw_labor","raw_nonlabor");
tabs = c("asm","load","pyield","fuelasm","labor","nonlabor");

for (l in 1:length(tabnames)){
    eval(parse(text=paste("tab=matrix(0,length(",tabnames[l],"$X1995),22)",sep    ="")));
for (i in 1:22){
        eval(parse(text=paste("tab[,i]=as.numeric(str_extract(",tabnames[l],"$X",i+1994,        ", gregexp))",sep="")));
}
    eval(parse(text=paste(tabs[l],"=tab;",sep="")));
}


# Separate airline wise and aggregate data from matrices
ind1 = c(1:7,9:13,15:18);
for(l in 1:length(tabnames)){
    eval(parse(text=paste(tabs[l],"1=",tabs[l],"[ind1,]",sep="")));
}

# Calculate HHI for each year (industry concentration)
hhi = colSums((asm1/t(replicate(nrow(asm1),asm[20,])))^2,na.rm=TRUE);

yX = cbind(as.vector(log(pyield1)),as.vector(load1),as.vector(log(fuelasm1)),as.vector(log(labor1)),as.vector(log(nonlabor1)),as.vector(replicate(16,hhi)),as.vector(log(asm1)));

# Vectors for fixed effects
F = matrix(0,nrow(yX),nrow(load1));
F[,1] = rbind(matrix(1,22,1),matrix(0,330,1));
for(i in 2:nrow(load1)){
    F[,i] = rbind(matrix(0,22*(i-1),1),matrix(1,22,1),matrix(0,22*(16-i),1));
}
D = data.frame(lnyield=yX[,1],f1=F[,1],f2=F[,2],f3=F[,3],f4=F[,4],f5=F[,5],f6=F[,6],f7=F[,7],f8=F[,8],f9=F[,9],f10=F[,10],f11=F[,11],f12=F[,12],f13=F[,13],f14=F[,14],f15=F[,15],f16=F[,16],load=yX[,2],lnfuelasm=yX[,3],lnlabor=yX[,4],lnnonlabor=yX[,5],hhi=yX[,6],lnasm=yX[,7]);

fe1<-lm(lnyield~-1+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+load+lnfuelasm+lnlabor+lnnonlabor+hhi+lnasm,data=D);
summary(fe1)

fe2<-lm(lnyield~-1+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+load+lnfuelasm+lnlabor+lnnonlabor+hhi,data=D);
summary(fe2)

re<-lm(lnyield~load+lnfuelasm+lnlabor+lnnonlabor+hhi,data=D);
summary(re)

##GMM estimator
#Get GDP and CPI deflator
raw_gdpcpi<-read.csv("usgdp_inflation.csv",header=TRUE,stringsAsFactors=FALSE);
gdp=as.numeric(str_replace(raw_gdpcpi$gdp_2009_chained_dollar,",",""));
cpi = as.numeric(raw_gdpcpi$CPI,",","");

## Define variables
pyield2 = log(pyield[20,]/cpi);
psm2 = log(asm[20,]*load[20,]);
gdp2 = log(gdp);
fuelasm2= log(fuelasm[20,]/cpi);
labor2 = log(labor[20,]/cpi);

## Get differences
yXiv = cbind(diff(psm2),diff(pyield2),diff(gdp2),hhi[2:length(hhi)],diff(fuelasm2),diff(labor2));

X1iv = cbind(rep(1,21),yXiv[,3:6]);
Z1iv = cbind(rep(1,21),yXiv[,2:3]);
X2iv = X1iv;
Z2iv = cbind(rep(1,21),yXiv[,c(2,4,5)]);
W1 = diag(5);
W2 = diag(5);
    
beta = solve(t(Z1iv)%*%X1iv%*%W1%*%t(X1iv)%*%Z1iv)%*%t(Z1iv)%*%X1iv%*%W1%*%t(X1iv)%*%yXiv[,1];
eb = yXiv[,1]-Z1iv%*%beta;
W1hat = solve(t(X1iv*replicate(ncol(X1iv),as.vector(eb)))%*%(X1iv*replicate(ncol(X1iv),as.vector(eb))));
beta = solve(t(Z1iv)%*%X1iv%*%W1hat%*%t(X1iv)%*%Z1iv)%*%t(Z1iv)%*%X1iv%*%W1hat%*%t(X1iv)%*%yXiv[,1];


gamma = solve(t(Z2iv)%*%X2iv%*%W2%*%t(X2iv)%*%Z2iv)%*%t(Z2iv)%*%X2iv%*%W2%*%t(X2iv)%*%yXiv[,1];
eg = yXiv[,1]-Z2iv%*%gamma;
W2hat = length(yXiv[,1])*solve(t(X2iv*replicate(ncol(X2iv),as.vector(eg)))%*%(X2iv*replicate(ncol(X2iv),as.vector(eg))));
gamma = solve(t(Z2iv)%*%X2iv%*%W2hat%*%t(X2iv)%*%Z2iv)%*%t(Z2iv)%*%X2iv%*%W2hat%*%t(X2iv)%*%yXiv[,1];
Vg = solve(t(Z2iv)%*%X2iv%*%W2hat%*%t(X2iv)%*%Z2iv);

### Simultaneous equations estimator
## Define variables
pyield2 = log(pyield[20,]/cpi);
psm2 = log(asm[20,]*load[20,]);
gdp2 = log(gdp);
fuelasm2= log(fuelasm[20,]/cpi);
labor2 = log(labor[20,]/cpi);

## Get differences
yXiv = cbind(diff(psm2),diff(pyield2),diff(gdp2),hhi[2:length(hhi)],diff(fuelasm2),diff(labor2));

D = data.frame(psm=yXiv[,1],pyield=yXiv[,2],gdp=yXiv[,3],hhi=yXiv[,4],fuel=yXiv[,5],labor=yXiv[,6]);

sim1<-lm(psm~gdp+fuel+labor+hhi,data=D);
summary(sim1)

sim2<-lm(pyield~gdp+fuel+labor+hhi,data=D);
summary(sim2)

D1 = data.frame(pyield=yXiv[,2],z=yXiv[,c(5,6,4)]%*%summary(sim1)$coefficients[3:5,1],gdp=yXiv[,3]);
sim3<-lm(pyield~gdp+z,data=D1);
summary(sim3)

## Get differences
yXiv = cbind(diff(psm2),diff(pyield2),diff(gdp2),hhi[2:length(hhi)],diff(fuelasm2)+diff(labor2));

D = data.frame(psm=yXiv[,1],pyield=yXiv[,2],gdp=yXiv[,3],hhi=yXiv[,4],cost=yXiv[,5]+yXiv[,6]);

sim1<-lm(psm~gdp+cost+hhi,data=D);
summary(sim1)

sim2<-lm(pyield~gdp+cost+hhi,data=D);
summary(sim2)

D1 = data.frame(pyield=yXiv[,2],z=yXiv[,c(5,6,4)]%*%summary(sim1)$coefficients[3:5,1],gdp=yXiv[,3]);
sim3<-lm(pyield~gdp+z,data=D1);
summary(sim3)
