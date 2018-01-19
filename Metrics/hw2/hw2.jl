workspace();

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
   bhat, res  = zeros(N,2),zeros(N,n);

   for i in 1:N
      obs[i] = draw(d,n);
      bhat[i,:],res[i,:] = OLS(obs[i]);
   end
   return obs, bhat, res;
end

function XXinv(m::Sample)
   X = [ones(length(m.X),1) m.X];
   return inv(X'*X);
end

## RUN MONTE CARLO SIMULATION

# QUESTION 1
μx = 4; # Mean of X
μϵ = 0; # Mean of ϵ
β0 = 5; # True intercept
β1 = 7; # True slope
vcm = [10 1.5;1.5 4];
N = 100;
n = 100;
d = dgp1(μx,μϵ,vcm,β0,β1);
obs,bhat,res = run_mc(d,N,n);
Vβs = map(XXinv, obs);
Vβ = mean(Vβs)*d.vcm[2,2];
avgβ = mean(bhat,1);

using Plots, Distributions
pyplot()
for (i,b) in enumerate([β0 β1])
   plt = histogram(bhat[:,i],bins = 100, normed = true, label="Monte Carlo Distribution")
   title = """
           Distribution of \$\\hat \\beta_$(i-1)\$
           (N = $N Draws, n = $n Obs each) """;
   title!(plt,title);

   #Plot the true distribution
   βmin, βmax = minimum(bhat[:,i]),maximum(bhat[:,i]);
   v = Normal(b,sqrt(Vβ[i,i]));
   x = linspace(βmin,βmax,100);
   y = pdf(v,x);
   plot!(plt,x,y,label="True Distribution");
   savefig(plt,"Plot6$i.jpg")
end
