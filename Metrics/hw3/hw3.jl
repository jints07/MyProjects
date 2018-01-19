workspace();
using Plots, Distributions
pyplot()

## SET PARAMETERS
type dgp1
μx   # Mean of X
μϵ   # Mean of ϵ (error term)
vcm  # Variance covariance matrix of (X,ϵ)
β0   # True intercept
β1   # True slope
end

type Sample
X
y
ϵ
end

function draw(w::dgp1,n::Integer)
   P = chol(w.vcm);
   Z = randn(2,n);
   D = (P*Z)';
   X = D[:,1] + w.μx;
   ϵ = D[:,2] + w.μϵ;
   y = w.β0 + w.β1*X + ϵ;
   return(Sample(X,y,ϵ));
end

function OLS(m::Sample)
   X = [ones(length(m.X),1) m.X];
   bhat = inv(X'*X)*X'*m.y;
   res = m.y - X*bhat;
   return bhat,res;
end

function run_mc(d::dgp1, N::Integer, n::Integer)
   obs = Vector{Sample}(N);
   bhat, res, Fstat  = zeros(N,2),zeros(N,n),zeros(N,1);
   R = [0 1];
   r = 0;

   for i in 1:N
      obs[i] = draw(d,n);
      bhat[i,:],res[i,:] = OLS(obs[i]);
      X = [ones(n) obs[i].X];
      s_sq = (obs[i].y-X*bhat[i,:])'*(obs[i].y-X*bhat[i,:])/(n-size(obs[i].X,2)-1);
      Fstat[i] = (R*bhat[i,:]-r)'*inv(s_sq*R*inv(X'X)*R')*(R*bhat[i,:]-r)/length(r);
   end
   return obs, bhat, res, Fstat;
end

function XXinv(m::Sample)
   X = [ones(length(m.X),1) m.X];
   return inv(X'*X);
end

## RUN MONTE CARLO SIMULATION

# QUESTION 1
μx = 4; # Mean of X
μϵ = 0; # Mean of ϵ
β0 = 2; # True intercept
β1 = 0.15; # True slope
vcm = [10 0;0 0.4];
N = 1000; # Number of samples
n = 1000; # Number of data points in a sample
d = dgp1(μx,μϵ,vcm,β0,β1);
obs,bhat,res,Fstat = run_mc(d,N,n);
Vβs = map(XXinv, obs);
Vβ = mean(Vβs)*d.vcm[2,2];
avgβ = mean(bhat,1);

## MAKE PLOTS
plt = histogram(Fstat,bins = 100, normed = true, label="Monte Carlo Distribution")
title = """
        Distribution of F-statistic
        (N = $N Draws, n = $n Obs each) """;
title!(plt,title);
v = FDist(1,n-size(obs[1].X,2)-1);
x = linspace(0,100,100);
y = pdf(v,x);
plot!(plt,x,y,label="F-distribution");
savefig(plt,"hw3_fstat1.jpg")

Tstat = sqrt.(Fstat);
plt = histogram(Tstat,bins = 100, normed = true, label="Monte Carlo Distribution")
title = """
        Distribution of T-statistic
        (N = $N Draws, n = $n Obs each) """;
title!(plt,title);
v = TDist(n-size(obs[1].X,2)-1);
x = linspace(-12,12,100);
y = pdf(v,x);
plot!(plt,x,y,label="T-distribution");
savefig(plt,"hw3_tstat1.jpg");
