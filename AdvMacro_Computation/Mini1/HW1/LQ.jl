#workspace();
using ForwardDiff, Distributions, Gadfly;

# BELLMAN ITERATION
# Set parameter values
γn = 0.03; # Population growth rate
γz = 0.05; # Productivity growth rate
β = 0.95*(1+γn); # β* - Modified discount rate for stationary problem
θ = 1/3; # Capital share
δ = 0.1; # Depreciation rate
ψ = 1.7; # Parameter in utility function
ρ = 0.95; # Persistence of technology shock
σz = 0.03; # Standard deviation of technology shock
z=0; # Steady state value of z (used for calculating gradient, hessian)

# Get Steady State Values
# Get non-stochastic steady state assuming z=0 (exp(z)=1) forever
h_k_ratio_ss = ((((1+γz)*(1+γn)/β)+δ-1)/θ)^(1/(1-θ));
c_k_ratio_ss = h_k_ratio_ss^(1-θ) + 1-δ-(1+γn)*(1+γz);
k_ss = ((1-θ)*h_k_ratio_ss^(-1*θ))/(ψ*c_k_ratio_ss+(1-θ)*h_k_ratio_ss^(1-θ));
h_ss = k_ss*h_k_ratio_ss;

## GET THE MATRICES Q,R,W
# Let X=State variable, u = control variable, Z = [1 X u], U(Z) - utility function. First we take second order taylor expansion around SS and construct matrix G such that U(Z) = Z'GZ. From this we can back out Q, R and W
nstate = 2; # Number of state variables
ncontrol = 2; # Number of control variables
nparam = nstate + ncontrol; #This is number of true parameters, equals number of state and control variables. Excludes variables like psi etc
param = [k_ss,z,k_ss,h_ss,ψ]; # param = [k,z,k',h,ψ]

# Define single period utility as function of parameters
Utility(X) = log(X[1]^θ*(exp(X[2])*X[4])^(1-θ) + (1-δ)*X[1] - (1+γz)*(1+γn)*X[3]) + X[5]*log(1-X[4]);
g = x->ForwardDiff.gradient(Utility,x);
h = x->ForwardDiff.hessian(Utility,x);

# Calculate first and second dervative of utility function
derivative_1 = g(param);
derivative_1 = derivative_1[1:nparam];
derivative_2 = h(param);
derivative_2 = derivative_2[1:nparam,1:nparam];

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

# Get Q, R and W from G
Q = G[1:(nstate+1),1:(nstate+1)];
R = G[(nstate+2):(nparam+1),(nstate+2):(nparam+1)];
W = G[1:(nstate+1),(nstate+2):(nparam+1)];

## GET MATRICES A,B AND C, A_tilde, B_tilde, Q_tilde
A = [1 0 0;0 0 0;0 0 ρ];
B = [0 0;1 0;0 0];
C = [0;0;1];
A_tilde = sqrt(β)*(A-B*inv(R)*W');
B_tilde = sqrt(β)*B;
Q_tilde = Q-W*inv(R)*W';

## RICATTI ITERATION TO GET P and F
ricatti_f(P) = Q + β*A'*P*A - (β*A'*P*B+W)*inv(R+β*B'*P*B)*(β*B'*P*A+W');
func_Ftilde(P) = inv(R + β*B'*P*B)*(sqrt(β)*B)'*P*(sqrt(β)*(A-B*inv(R)*W'));

# Set an initial value for P and F
P = -eye(nstate+1);
F = func_Ftilde(P);

# Ricatti Iteration
tolerance = 0.0000001; #tolerance level
iter = 1; #nummber of iterations
diff = 10;
while iter<1000 && diff>tolerance
    Pnew = ricatti_f(P);
    Fnew = func_Ftilde(P);
    diff1 = sum(abs.(Pnew-P));
    diff2 = sum(abs.(Fnew-F));
    diff = max(diff1,diff2);
    P = Pnew;
    F = Fnew;
    iter=iter+1;
end
F = F + inv(R)*W';

## GIVEN INITIAL STATE, CALCULATE PATH OF X (State)  AND u (Control)
k0 = 0.6;
z0 = 0.0;
path_length = 100;
if σz>0
    e_z = Normal(0,σz);
    ε = rand(e_z,path_length-1);
else
    ε = zeros(path_length-1,1);
end
X = zeros(nstate+1,path_length);
X[:,1] = [1 k0 z0]';
U = zeros(size(R,1),path_length);
for i = 2:path_length
    X[:,i] = (A-B*F)*X[:,i-1] + C*ε[i-1];
end
U = -F*X;

plot(x=1:path_length,y=X[2,:])
