workspace();
using Distributions,Gadfly;

function CheckIsSorted(Valuefn)
    sorted = 1; # Inidicates if Valuefn matrix is increasing with k and z
    for i=1:size(Valuefn,1)
        if !issorted(Valuefn[i,:])
            sorted = 0;
            println("row $i")
            break
        end
    end
    for i=1:size(Valuefn,2)
        if !issorted(Valuefn[:,i])
            sorted = 0;
            println("column $i")
            break
        end
    end
    return sorted;
end

function get_continuation_value(k_next,Continuation_Valuefn,j)
    cont_value = 0;
    if k_next<k_grid[1]
         slope = (Continuation_Valuefn[2,j]-Continuation_Valuefn[1,j])/(k_grid[2]-k_grid[1]);
         cont_value = Continuation_Valuefn[1,j] - slope*(k_grid[1]-k_next);
    elseif k_next>=k_grid[1] && k_next<last(k_grid)
        ind = last(find(x->(x <= k_next),k_grid));
        cont_value = ((k_next-k_grid[ind])/(k_grid[ind+1]-k_grid[ind]))*Continuation_Valuefn[ind+1,j] + ((k_grid[ind+1]-k_next)/(k_grid[ind+1]-k_grid[ind]))*Continuation_Valuefn[ind,j];
    elseif k_next>=last(k_grid)
        cont_value = Continuation_Valuefn[length(k_grid),j];
    end
    return cont_value;
end

function unpack(params)
    return params[1],params[2],params[3],params[4],params[5],params[6];
end

function Compute_Optimal_Decision(Continuation_Valuefn,k,j)
    c = zeros(length(h_grid)); # Current consumption choices for each 'h'
    k_next = zeros(length(h_grid)); # Next period capital for each 'h'
    continuation_value = zeros(length(h_grid)); # Expected continuation value
    total_utility = zeros(length(h_grid)); # Sum of current period utility and continuation value
    for l in 1:length(h_grid)
         c[l] = ((1-θ)*(1-h_grid[l])*(k^θ)*exp(z_array[j]*(1-θ))*h_grid[l]^(-1*θ))/ψ; # consumption from labor-leisure condition

        # Check if c <= production (since investment>0) 
        if c[l] <= (k^θ)*((exp(z_array[j])*h_grid[l])^(1-θ)) 
           k_next[l] = ((k^θ)*((exp(z_array[j])*h_grid[l])^(1-θ)) + (1-δ)*k - c[l])/((1+γ_n)*(1+γ_z)); # next period capital from resource constraint
           continuation_value[l] = get_continuation_value(k_next[l],Continuation_Valuefn,j);
        else
            continuation_value[l] = -Inf;
        end
        total_utility[l] = log(c[l]) + ψ*log(1-h_grid[l]) + continuation_value[l];
    end
    return maximum(total_utility),h_grid[indmax(total_utility)],c[indmax(total_utility)],k_next[indmax(total_utility)];
end

function Get_Continuation_Valuefn(Valuefn)
    Continuation_Valuefn = zeros(size(Valuefn));
    for i in 1:size(Valuefn,1)
        for j in 1:size(Valuefn,2)
            Continuation_Valuefn[i,j] = β*P_z[j,:]'*Valuefn[i,:];
        end
    end
    return Continuation_Valuefn;
end

function Discretize_Tauchen(σ_ϵ,ρ,m)
    if σ_ϵ>0
        
        # Get the vector of values z can take
        len_z_array = 41;
        d = 2*m*σ_ϵ/(sqrt(1-ρ^2)*(len_z_array-1));
        z_array =[-m*σ_ϵ/sqrt(1-ρ^2):d:-m*σ_ϵ/sqrt(1-ρ^2)+d*(len_z_array-1);];
        ϵ = Normal(0,σ_ϵ);

        # Transition probability matrix for z
        P_z = zeros(len_z_array,len_z_array);
        for i in 1:len_z_array
            P_z[i,1] = cdf(ϵ,z_array[1]+d/2-ρ*z_array[i]);
            for j in 2:len_z_array-1
                P_z[i,j] = cdf(ϵ,z_array[j]+d/2-ρ*z_array[i]) - cdf(ϵ,z_array[j]-d/2-ρ*z_array[i]);
            end
            P_z[i,len_z_array] = 1 - cdf(ϵ,z_array[len_z_array]-d/2-ρ*z_array[i]);
        end
    else
        len_z_array = 1;
        z_array = [0];
        P_z = reshape([1],1,1);
    end    
    return z_array,len_z_array,P_z;
end

# Discretize AR-1 process
σ_ϵ = 0.03;  # Std dev of white noise
ρ = 0.95;
m = 3; # z will take values between -m and m std dev

z_array,len_z_array,P_z = Discretize_Tauchen(σ_ϵ,ρ,m);

# Variance of discrete variable (used to compare to that of CTS variable)
c = eig(P_z');
π_stationary = c[2][:,1]/sum(c[2][:,1]);
π_stationary_var = π_stationary'*(z_array.^2) - (π_stationary'*(z_array))^2;

# BELLMAN ITERATION
# Set parameter values
γ_n = 0.03; # Population growth rate
γ_z = 0.05; # Productivity growth rate
β = 0.95*(1+γ_n); # β* - modified discount rate for stationary problem
θ = 1/3; # Capital share
δ = 0.1;
ψ = 1.7;

# Get Steady State Values
# Get non-stochastic steady state assuming z=0 (exp(z)=1) forever
z = 0;
h_k_ratio_ss = ((((1+γ_z)*(1+γ_n)/β)+δ-1)/θ)^(1/(1-θ));
c_k_ratio_ss = h_k_ratio_ss^(1-θ) + 1-δ-(1+γ_n)*(1+γ_z);
k_ss = ((1-θ)*h_k_ratio_ss^(-1*θ))/(ψ*c_k_ratio_ss+(1-θ)*h_k_ratio_ss^(1-θ));
h_ss = k_ss*h_k_ratio_ss;

# Grid for labor (h)
h_grid = zeros(99);
for i in 1:length(h_grid)
h_grid[i] = 0.01*i;
end

# Get maximum possible capital stock (k) for which k'>=k
k_max = exp(last(z_array))*last(h_grid)/(((1+γ_z)*(1+ γ_n)-(1-δ))^(1/(1-θ)));

# Grid for capital stock (k)
k_grid_multiplier = 1.15;
k_grid = zeros(60+fld((log(k_max)-log(3*k_ss))/log(k_grid_multiplier),1));
for i in 1:60
    k_grid[i] = i*k_ss*0.05;
end
for i in 61:length(k_grid)
k_grid[i] = k_grid[i-1]*k_grid_multiplier;
end

# Value function matrix in (k,z)
Valuefn = zeros(length(k_grid),length(z_array));

# BELLMAN ITERATION
ss_diff = 10; # Sum of squares difference between last iteration and current iteration
iter = 1; # Number of iterations
maxiter = 500; # max iterations

c = zeros(length(h_grid)); # Current consumption choices for each 'h'
k_next = zeros(length(h_grid));

while ss_diff > 0.0001 && iter < maxiter
    Valuefn_Next = zeros(size(Valuefn));
# First calculate continuation value βE[V(k',z')|z] for all discrete (k,z)
Continuation_Valuefn = Get_Continuation_Valuefn(Valuefn);
for i in 1:size(Valuefn,1)
    for j in 1:size(Valuefn,2)
        Valuefn_Next[i,j],~,~,~ = Compute_Optimal_Decision(Continuation_Valuefn,k_grid[i],j);
    end
end
    ss_diff = maximum(abs(Valuefn_Next-Valuefn));
    Valuefn_issorted = CheckIsSorted(Valuefn_Next);
    if Valuefn_issorted==0
        println("Valuefn not increasing in k and z")
        break
    end    
    Valuefn = Valuefn_Next;
    iter = iter +1;
    if iter==maxiter
        println("max iterations reached, no convergence")
    end
end

## GET POLICY FUNCTIONS c(k,z), h(k,z) & k′(k,z) FROM VALUE FUNCTION V(k,z)
h_policy = zeros(size(Valuefn));
c_policy = zeros(size(Valuefn));
k_next_policy = zeros(size(Valuefn));

# Continuation Value
Continuation_Valuefn = Get_Continuation_Valuefn(Valuefn);
for i in 1:length(k_grid)
    for j in 1:length(z_array)
        ~,h_policy[i,j],c_policy[i,j],k_next_policy[i,j] = Compute_Optimal_Decision(Continuation_Valuefn,k_grid[i],j);
    end
end

## SIMULATE PATHS FROM STARTING VALUE OF (k,z)
periods = 100;
# all "_hat" variables are stationary. Variables without hat include population and technological growth
k_start = 0.6;
z_start = 30; #contains index of shock in "z_array", not value of shock itself

k_hat_path = zeros(periods+1); # Capital path
c_hat_path = zeros(periods); # Consumption path
y_hat_path = zeros(periods); # Output path
h_path = zeros(periods); # Hours path
z_path = zeros(Int,periods+1); # Shock path - #contains index of shock in "z_array", not value of shock itself
k_hat_path[1] = k_start;
z_path[1] = z_start;
temp = rand(Uniform(0,1),periods);

P_z_cdf = copy(P_z);
for i in 1:size(P_z,1)
    for j in 2:size(P_z,1)
        P_z_cdf[i,j] = P_z_cdf[i,j-1] + P_z[i,j]
    end
end

Continuation_Valuefn = Get_Continuation_Valuefn(Valuefn);
for i in 1:periods
    z_path[i+1] = searchsortedfirst(P_z_cdf[z_path[i],:],temp[i]);
    ~,h_path[i],c_hat_path[i],k_hat_path[i+1] = Compute_Optimal_Decision(Continuation_Valuefn,k_hat_path[i],z_path[i]);
end

plot(x=1:periods,y=k_hat_path[1:periods])
