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
   ϵ = X.*D[:,2] + w.μϵ;
   y = w.β0 + w.β1*X + ϵ;
   return(Sample(X,y,ϵ));
end

function OLS(m::Sample)
   X = [ones(length(m.X),1) m.X];
   bhat = inv(X'*X)*X'*m.y;
   res = m.y - X*bhat;
   S = X'*(repmat(res.^2,1,2).*X); # mean((ϵ_i^2)*x_i*x_i')
   return bhat,res,S;
end

function run_mc(d::dgp1, N::Integer, n::Integer)
   obs = Vector{Sample}(N);
   bhat, res  = zeros(N,2),zeros(N,n);
   S = 0;
   for i in 1:N
      obs[i] = draw(d,n);
      bhat[i,:],res[i,:],s = OLS(obs[i]);
      S = S+s;
   end
   S=S/(N*n);
   return obs, bhat, res, S;
end

function Vbhat(m::Sample)
    X = [ones(length(m.X),1) m.X];
    ϵ = m.ϵ;
   return inv(X'*X)*X'*ϵ*ϵ'*X*inv(X'*X);
end

function XX(m::Sample)
    X = [ones(length(m.X),1) m.X];
   return (X'*X);
end

## RUN MONTE CARLO SIMULATION

# QUESTION 1
μx = 6; # Mean of X
μϵ = 0; # Mean of ϵ
β0 = 9; # True intercept
β1 = 3; # True slope
vcm = [5 0;0 1];
N = 100;

using Plots, Distributions
pyplot()
n_choices = [10 20 30 50 100 1000 5000];
for (i_n,n) in enumerate(n_choices)
d = dgp1(μx,μϵ,vcm,β0,β1);
obs,bhat,res,S = run_mc(d,N,n);
Vβs = map(Vbhat, obs);
Vβ = mean(Vβs); # Estimate for unconditional variance of bhat
avgβ = mean(bhat,1);
bhat_β = sqrt(n)*(bhat-repmat([β0 β1],N,1));
Σxx = mean(map(XX,obs))./n;
Vβ_asymp = inv(Σxx)*S*inv(Σxx);

cd("D:/Work/MyProjects/Metrics/HW7/");
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
   savefig(plt,"Plothw7_n$n _$i.jpg")

   plt1 = histogram(bhat_β[:,i],bins = 100, normed = true, label="Monte Carlo Distribution")
   title1 = """
           Distribution of sqrt(n)*( \$\\hat \\beta_$(i-1) - \\beta_$(i-1)\$)
           (N = $N Draws, n = $n Obs each, \$\\hat S=$(round(Vβ_asymp[i,i],2))\$) """;
   title!(plt1,title1);
   #Plot the true distribution
   βmin, βmax = minimum(bhat_β[:,i]),maximum(bhat_β[:,i]);
   v = Normal(0,sqrt(Vβ_asymp[i,i]));
   x = linspace(βmin,βmax,100);
   y = pdf(v,x);
   plot!(plt1,x,y,label="Asymptotic Distribution");
   savefig(plt1,"Plothw7_asymp_n$n _$i.jpg")
end
end
