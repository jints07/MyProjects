# KALMAN FILTER FOR THE FOLLOWING PROBLEM
"""
α_t = T α_{t-1} + η_t,   E [η_t η_t'] = Q
y_t = Z α_t + δ_t,       E [δ_t δ_t'] = H
"""
abstract type ARMAProcess end

type AR<:ARMAProcess
""" AR(p) process
     x_t = ρ(1) x_{t-1} + ρ(2) x_{t-2} +.....+ρ(p) x_{t-p} + ε_t
     ε_t] = 0 white noise, E [ε_t ε_t'] = σ_ε^2
"""
    ρ # ρ = [ρ(1) ρ(1) ..... ρ(k)] - Vector of coefficients
    σ_ε # Std deviation of innovation
end

type MA<:ARMAProcess
""" MA(q) process
x_t = ε_t + γ(1) ε_{t-1} + ....... + γ(q) ε_{t-q}
Normalization - V[ε_t] = 1
"""
    γ   # γ = [γ(1) γ(2) ......γ(q)] - Coefficients for white noise terms
    σ_ε # std dev of shock process
end

type RW<:ARMAProcess
""" Random Walk
x_t = x_{t-1} + ε_t
"""
    σ_ε # Std deviation of innovation
end

type dgp
"""x_t is some latent ARMA(p,q) process
   y_t = x_t + δ_t, E[δ_t^2] = σ_δ^2, y_t is observed
"""
    x::ARMAProcess
    σ_δ
end

function draw(x::AR,n::Integer)
    ρ = x.ρ;
    rts = roots(Poly(vec([1 -ρ])));
    minimum(abs.(rts))>=1 || error("AR process not stationary")
    ar_coeff = reverse(vec(ρ));
    σ_ε = x.σ_ε;
    ndiscard = 100; # Discard first few simulated values
    ε = σ_ε*randn(n+ndiscard);
    X = zeros(length(ε)+length(ρ));
    for i=length(ρ)+1:length(X)
        X[i] = sum(ar_coeff.*X[i-length(ρ):i-1]) + ε[i-length(ρ)];
    end
    X = X[ndiscard+length(ρ)+1:end];
    return X;
end

function draw(x::MA, n::Integer)
    γ = [1 x.γ];
    ndiscard = 100; # Discard first few simulated values
    ε = σ_ε*randn(n+ndiscard);
    ma_coeff = reverse(vec(γ));
    X = zeros(length(ε));
    for i=length(γ):length(X)
        X[i] = sum(ma_coeff.*ε[i-length(γ)+1:i]);
    end
    X = X[ndiscard+1:end];
    return X;
end

function draw(x::RW, n::Integer)
    σ_ε = x.σ_ε;
    ndiscard = 20; # Discard first few simulated values
    ε = σ_ε*randn(n+ndiscard);
    X = cumsum(ε);
    X = X[ndiscard+1:end];
    return X;
end

function draw(z::dgp,n::Integer)
    δ = z.σ_δ*randn(n);
    X = draw(z.x,n);
    Y = X + δ;
    return X,Y;
end

function Kalman(params,w::dgp)
"""
α_t = T α_{t-1} + η_t,   E [η_t η_t'] = Q
y_t = Z α_t + δ_t,       E [δ_t δ_t'] = H
"""
    T,Q,Z = kalman1(params,w.x);
    if w.σ_δ>0
        H = [params[end]^2];
    else
        H = [0.0];        
    end    
    return T,Z,Q,H;
end

function kalman1(params,x::AR)
    ρ = params[1:end-2];
    σ_η = params[end-1];
    if length(ρ)>1
        T = [ρ'; eye(length(ρ)-1) zeros(length(ρ)-1)];
    else
        T = reshape(ρ,length(ρ),1);
    end
    Z = [1; zeros(length(ρ)-1)]';
    Q = zeros(T);
    Q[1,1] = σ_η^2;
    return T,Q,Z;
end

function kalman1(params,x::MA)
    γ = params[1:end-2];
    T = [zeros(length(γ)+1)';
         eye(length(γ)) zeros(length(γ))];
    Q = zeros(length(γ)+1,length(γ)+1);
     Q[1,1] = params[end-1]^2;
    Z = [1 γ'];
    return T,Q,Z;
end

function kalman1(params,x::RW)
    T=reshape([1.0],1,1);
    Q = reshape([params[1]^2],1,1);
    Z = eye(1);
    return T,Q,Z;
end

function LogLike(params,w::dgp,α0,P0,Y)
    T,Z,Q,H = Kalman(params,w);
    ll = 0;
    αt,Pt =  α0,P0;    
    for i=1:length(Y)
        αtt_1 = T*αt;
        Ptt_1 = T*Pt*T' + Q;
        Ft = Z*Ptt_1*Z' + H;
        vt = Y[i] - Z*αtt_1;
        αt = αtt_1 + Ptt_1*Z'*(Ft\eye(size(Ft,1)))*vt;
        Pt = Ptt_1 - Ptt_1*Z'*(Ft\eye(size(Ft,1)))*Z*Ptt_1;
        ll = ll +(0.5*log((length(Ft)>1? det(Ft): Ft[1]))+0.5*vt'*(Ft\eye(size(Ft,1)))*vt)[1];
    end
    return ll[1];
end

using Polynomials,Optim,Gadfly;

#########################################################
### IMPLEMENT KALMAN FILTER
## AR PROCESS
# Simulate AR process
ρ = [0.3 0.4]; # coefficients of AR(k) process
σ_ε = 2.0; # std deviation of white noise for latent variable
σ_δ = 0.5; # std deviation of white noise for observed variable
n = 10000; # Length of time series
w = dgp(AR(ρ,σ_ε),σ_δ);
X,Y = draw(w,n);

# Initial guesses for parameters
ρhat = 3*rand(length(ρ));
σ_εhat = 1;
if σ_δ>0
    σ_δhat = 1.0;
else
    σ_δhat = 0.0;
end
# Initial guesses for x0 and P0
x0 = zeros(length(ρ));
P0 = eye(length(ρ));
params =  vec([ρhat;σ_εhat;σ_δhat]);

# Estimate parameters using Kalman Filter
f(x) = LogLike(x,w,x0,P0,Y);
asr = optimize(f,params)
asr.minimizer

#####################################################
## MA PROCESS
γ = [0.74]; # coefficients of MA(q) process
σ_δ = 0.0; # std deviation of white noise for observed variable
σ_ε = 2.0; # std dev of white noise for MA(q) process
n = 10000; # Length of time series
w = dgp(MA(γ,σ_ε),σ_δ);
X,Y = draw(w,n);

# Initial guesses for parameters
γhat = 1*rand(length(γ));
σ_εhat = 1.0;
if σ_δ>0
    σ_δhat = 1.0;
else
    σ_δhat = 0.0;
end
# Initial guesses for x0 and P0
x0 = zeros(length(γ)+1);
P0 = eye(length(γ)+1);

params =  vec([γhat;σ_εhat;σ_δhat]);

# Estimate parameters using Kalman Filter
f(x) = LogLike(x,w,x0,P0,Y);
asr = optimize(f,params)
asr.minimizer

###########################################
# Show that MLE has local maxima for MA(1) process with 0.74
x0 = zeros(length(γ)+1);
P0 = eye(length(γ)+1);
LL = zeros(150);
for i=1:150
    γhat = 0.01*i;
    params =  vec([γhat;σ_δhat]);
    LL[i] = min(20000,LogLike(params,w,x0,P0,Y));
end

plot(x=1:150,y=LL)

#################################################
## RANDOM WALK
σ_ε = 1.5;
σ_δ = 0.5; # std deviation of white noise for observed variable
n = 10000; # Length of time series
w = dgp(RW(σ_ε),σ_δ);
X,Y = draw(w,n);

# Initial guesses for parameters
σ_εhat = 0.5+rand(1);
if σ_δ>0
    σ_δhat = 1.0;
else
    σ_δhat = 0.0;
end
# Initial guesses for x0 and P0
x0 = rand(1);
P0 = eye(1);
params =  vec([σ_εhat;σ_δhat]);

# Estimate parameters using Kalman Filter
f(x) = LogLike(x,w,x0,P0,Y);
asr = optimize(f,params)
asr.minimizer


