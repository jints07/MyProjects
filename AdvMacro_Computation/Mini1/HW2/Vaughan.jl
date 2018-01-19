#workspace();
#using ForwardDiff, Distributions, Gadfly, NLsolve;

# BELLMAN ITERATION
# Set parameter values
γn = 0.03; # Population growth rate
γz = 0.05; # Productivity growth rate
β = 0.95*(1+γn); # β* - Modified discount rate for stationary problem
θ = 1/3; # Capital share
δ = 0.1; # Depreciation rate
ψ = 1.7; # Parameter in utility function
Ps = [0.0 0   0   0;
     0   0.0 0   0;
     0   0   0.0 0;
     0   0   0   0.0]
Qs = [0.0 0   0   0;
     0   0.0 0   0;
     0   0   0.0 0;
     0   0   0   0.0]

# Get Steady State Values of k and h
S_ss = [0 0.0 0.0 log(0.0)]; #SS shocks S_ss = [log(z_ss) τx_ss τh_ss log(g_ss)];
r_ss(x_ss) = θ*(x_ss[1]/x_ss[2])^(θ-1)*exp(S_ss[1])^(1-θ); #SS rental rate
w_ss(x_ss) = (1-θ)*(x_ss[1]/x_ss[2])^θ*exp(S_ss[1])^(1-θ); #SS  wage
κ_ss(x_ss) = S_ss[2]*((1+γn)*(1+γz)*x_ss[1] - (1-δ)*x_ss[1]) + S_ss[3]*w_ss(x_ss)*x_ss[2] - exp(S_ss[4]); # SS transfer
c_ss(x_ss) =  r_ss(x_ss)*x_ss[1] + (1-S_ss[3])*w_ss(x_ss)*x_ss[2] + κ_ss(x_ss) - (1+S_ss[2])*((1+γn)*(1+γz)*x_ss[1] - (1-δ)*x_ss[1]); #SS consumption
function f!(x_ss,resvec)
        resvec[1] = ψ*c_ss(x_ss) - (1-S_ss[3])*w_ss(x_ss)*(1-x_ss[2])
        resvec[2] = (1+γn)*(1+γz) - β*(r_ss(x_ss) + (1+S_ss[2])*(1-δ))       
    end
x_start = [.1; 0.4]; #Initial guess for steady state [k_ss h_ss]
x_ss = nlsolve(f!,x_start,autodiff = true)
x_ss = x_ss.zero;

## GET THE MATRICES Q,R,W
# State variable X_t = [1 Y_t] =  [1 k_t log(z_t) τ_xt τ_ht log(g_t/g_ss) K_t H_t K_t+1]
# Control u_t = [k_t+1 h_t]
# Z_t = [X_t u_t] = [1 Y_t u_t]
# X_t = [X_1t X_2t X_3t] where X_1t, X_2t & X_3t represent "little k" variables, shock processes and "big K" variables respectively
# U(Z) - utility function. First we take second order taylor expansion around SS and construct matrix G such that U(Z) = Z'GZ. From this we can back out Q, R and W
n_X = 8; # Number of state variables excluding constant 1. Includes variables with unknown laws of motion as well
n_y = 6; # Number of state variables with known laws of motion
n_u = 2; # Number of control variables
nparam = n_X + n_u; #This is number of true parameters, equals number of state and control variables. Excludes variables like psi etc
param = [x_ss[1],S_ss[1],S_ss[2],S_ss[3],0,x_ss[1],x_ss[2],x_ss[1],x_ss[1],x_ss[2]]; # param = [k,log(z),τ_xt,τ_ht,log(g_t)-log(g_ss),K,H,K',k',h]

# Define single period utility as function of parameters
c(X) = r(X)*X[1] + (1-X[4])*w(X)*X[10] + κ(X) - (1+X[3])*((1+γn)*(1+γz)*X[9] - (1-δ)*X[1]); #Consumption of individual
r(X) = θ*(X[6]/X[7])^(θ-1)*exp(X[2])^(1-θ); # rental rate for capital
w(X) = (1-θ)*(X[6]/X[7])^θ*exp(X[2])^(1-θ); # wage
κ(X) = X[3]*((1+γn)*(1+γz)*X[8] - (1-δ)*X[6]) + X[4]*w(X)*X[7] - exp(S_ss[4])*exp(X[5]); # transfer from govt
Utility(X) = log(c(X)) + ψ*log(1-X[10]);
g = x->ForwardDiff.gradient(Utility,x);
h = x->ForwardDiff.hessian(Utility,x);

# Calculate first and second dervative of utility function
derivative_1 = g(param);
derivative_1 = derivative_1;
derivative_2 = h(param);
derivative_2 = derivative_2;

# Get matrix G
# Define constant of Taylor expansion
Const = Utility(param) - derivative_1'*param[1:nparam] + 0.5*param[1:nparam]'*derivative_2*param[1:nparam];

# Define Vector to be multiplied with Z 
N = derivative_1'-param[1:nparam]'*derivative_2;
G = zeros(nparam+1, nparam+1);
G[1,1] = Const;
G[1,2:size(G,2)] =  N[1:size(N,2)]/2;
G[2:size(G,1),1] = N'/2;
G[2:size(G,1),2:size(G,2)] = derivative_2/2;

# Step 1 - Get Q, R and W from G
Q = G[1:n_X+1,1:n_X+1];
R = G[n_X+2:(nparam+1),(n_X+2):(nparam+1)];
W = G[1:(n_X+1),(n_X+2):(nparam+1)];

# Step 2 - Get Q_tilde,Q_tilde_y,Q_tilde_z
Q_tilde  = Q - W*inv(R)*W';
Qy_tilde = Q_tilde[1:n_y+1,1:n_y+1];
Qz_tilde = Q_tilde[1:n_y+1,n_y+2:n_X+1];


## GET MATRICES Ay,Az,By, Ay_tilde, Az_tilde, By_tilde
Ay = [1.0 0 0 0 0 0 0;
            0 0 0 0 0 0 0;
            0 0 0 0 0 0 0;
            0 0 0 0 0 0 0;
            0 0 0 0 0 0 0;
            0 0 0 0 0 0 0;
            0 0 0 0 0 0 0];
Ay[3:6,3:6] = Ps;
By = [0.0 0;
      1 0;
      0 0;
      0 0;
      0 0;
      0 0;
      0 0];
Cy = zeros(size(Ay,1),size(Ps,1));
Cy[3:6,1:size(Ps,1)] = Qs;
Wy = W[1:n_y+1,1:size(W,2)];
Wz = W[n_y+2:size(W,1),1:size(W,2)];
Az = [0.0 0;
      0 0;
      0 0;
      0 0;
      0 0;
      0 0;
      0 1];
Ay_tilde = sqrt(β)*(Ay-By*inv(R)*Wy');
Az_tilde = sqrt(β)*(Az-By*inv(R)*Wz');
By_tilde = sqrt(β)*By;

# Get Θ, Θ_tilde, ψ_mc,ψ_tilde from market clearing condition
# X_3t = Θ [X_1t; X_2t] + ψ_mc u_t
Θ = zeros(n_X - n_y,n_y+1)
ψ_mc = [0.0 1;
        1 0]; # comes from market clearing X_3t = Θ [X_1t X_2t] + ψ_mc u_t
Θ_tilde = inv(eye(ψ_mc) + ψ_mc*inv(R)*Wz')*(Θ - ψ_mc*inv(R)*Wy');
ψ_tilde = inv(eye(ψ_mc) + ψ_mc*inv(R)*Wz')*ψ_mc;

# Get matrices A_hat,Q_hat,B_hat and A_bar
A_hat = Ay_tilde + Az_tilde*Θ_tilde;
Q_hat = Qy_tilde + Qz_tilde*Θ_tilde;
B_hat = By_tilde + Az_tilde*ψ_tilde;
A_bar = Ay_tilde - By_tilde*inv(R)*ψ_tilde'*Qz_tilde';

# Get H1 and H2 (the hamiltonians)
H1 = [A_hat zeros(A_hat);
      -1*Q_hat eye(Q_hat)];
H2 = [eye(A_hat) B_hat*inv(R)*By_tilde';
      zeros(A_hat) A_bar'];
H = H2,H1;

# Construct the eigenvalues/eigenvectors
    Λ, V = eig(H...)
    
    # Sort the eigenvalues from largest to smallest and then reorder the
    # eigenvectors accordingly
    idx = sortperm(-Λ)
    Λ   = Λ[idx]
    V   = V[:,idx]
    
    # Partition the Eigenvectors and Eigenvalues
    l       = searchsortedfirst(-Λ, -1) - 1
    Λ1, Λ2  = Λ[1:l],Λ[l+1:length(Λ)]
    V11, V12, V21, V22 = V[1:l,1:l],V[1:l,l+1:end],V[l+1:end,1:l],V[l+1:end,l+1:end]
P = V21*inv(V11);    
F = inv(R+By_tilde'*P*B_hat)*By_tilde'*P*A_hat;
F = inv(R + Wz'*ψ_mc)*(R*F + Wy' + Wz'Θ);
A0 = Ay + Az*Θ - Az*ψ_mc*F - By*F;

## GIVEN INITIAL STATE, CALCULATE PATH OF X (State)  AND u (Control)
K0=k0 = 0.5;
S0 = [0.2 0.0 0.0 0.0]; # Initial value for S = [log(z) τ_xt τ_ht g]
path_length = 100;

# Simulate shocks for entire time period
# S = [log(z_t) τ_xt τ_ht log(g_t)-log(g_ss)]. Here g_t = g_ss*exp(S_ss[4]) where S_ss[4] is 4th element of shock that models govt spending. Therefore, S_ss[4] = log(g_t) - log(g_ss)
e_z = Normal(0,1);
ε = rand(e_z,size(Qs,1),path_length-1);

X = zeros(n_y+1,path_length);
X[:,1] = [1 k0 S0 K0]';
U = zeros(size(R,1),path_length);
for i = 2:path_length
    X[:,i] = A0*X[:,i-1] + Cy*ε[:,i-1];
end
U = -F*X;

plot(x=1:path_length,y=U[1,:])
