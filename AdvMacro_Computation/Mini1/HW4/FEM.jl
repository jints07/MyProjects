using FastGaussQuadrature,NLsolve,Gadfly,ForwardDiff;

## Define c(k) = Sum(i=1 to n) θ_i x^i 
ψdownrhs(x,θ,i) = (i>1 ? θ[i-1]*(partitions[i+1]-x)/(partitions[i+1]-partitions[i]): 0);
ψuplhs(x,θ,i) = i>1 ? θ[i-1]*(x-partitions[i-1])/(partitions[i]-partitions[i-1]): 0;
c(x,θ) = x>partitions[end] ? θ[end]: ψdownrhs(x,θ,searchsortedfirst(partitions,x)-1) + ψuplhs(x,θ,searchsortedfirst(partitions,x));

##Residual R(k;θ) = [β*c(k)*[α*A*k'(k)^(α-1) + 1-δ]]/c(k'(k))-1
R(k,θ) = (β*c(k,θ)*(α*A*(A*k^α + (1-δ)*k - c(k,θ))^(α-1) + 1-δ))/c(A*k^α + (1-δ)*k - c(k,θ),θ) - 1;

# Residual
function residual!(θ1, resvec)    
    θ1 = min(θ1,0.999*(A*partitions[2:end].^α + (1-δ)*partitions[2:end])); # Resource constraint (c< resources available)
    
    jmid = searchsortedfirst(nodes,0); #divide interval into left and right side of tent
    for j=2:length(partitions)-1
        knodes = ((nodes+1)*(partitions[j+1]-partitions[j-1])/2 + partitions[j-1]);
        kweights = weights*(partitions[j+1]-partitions[j-1])/2;
        resvec[j-1] = sum((ψuplhs(knodes[1:jmid-1],θ1,j)/θ1[j-1]).*map(x->R(x,        θ1),knodes[1:jmid-1])) + sum((ψdownrhs(knodes[jmid:end],θ1,j)/θ1[j-1])       .*map(x->R(x,θ1),knodes[jmid:end]));
    end
    j=length(partitions);
    knodes = ((nodes+1)*(partitions[j]-partitions[j-1])/2 + partitions[j-1]);
    kweights = weights*(partitions[j]-partitions[j-1])/2;
    resvec[j-1] = sum((ψuplhs(knodes,θ1,j)/θ1[j-1]).*map(x->R(x,θ1),knodes));
end

## Initialize paramters of model
β = 0.96;
δ = .1;
α = 0.25;
A = 1/(α*β);
n = 5;
k_ss = ((1/β + δ - 1)/(α*A))^(1/(α-1)); #steady state capital
c_ss = A*k_ss^α - δ*k_ss;
partitions = [0:0.02*k_ss:0.1*k_ss; 0.2*k_ss:0.1*k_ss:2*k_ss];
partitions = [0:0.005*k_ss:0.02k_ss; 0.04*k_ss:0.02*k_ss:0.1*k_ss; 0.2*k_ss:0.1*k_ss:1.5*k_ss];

## Gauss quadrature nodes and weights for k
nodes, weights = gausslegendre( 25 );

θstart = ones(length(partitions)-1);
θstart[searchsortedfirst(partitions,k_ss)-1:end] = max(θstart[searchsortedfirst(partitions,k_ss)-1:end], A*k_ss^α + -δ*k_ss + 0.0001);
θoptimal = nlsolve(residual!,θstart,autodiff=true)

θoptimal.zero[searchsortedfirst(partitions,k_ss)-1]-c_ss

plot(x=partitions,y=[0 θoptimal.zero'])
res(θoptimal.zero)

function res(θ1)
    resvec = zeros(length(partitions)-1,1);
    jmid = searchsortedfirst(nodes,0);
    for j=2:length(partitions)-1
        knodes = ((nodes+1)*(partitions[j+1]-partitions[j-1])/2 + partitions[j-1]);
        kweights = weights*(partitions[j+1]-partitions[j-1])/2;
        resvec[j-1] = sum((ψuplhs(knodes[1:jmid-1],θ1,j)/θ1[j-1]).*map(x->R(x,        θ1),knodes[1:jmid-1])) + sum((ψdownrhs(knodes[jmid:end],θ1,j)/θ1[j-1])       .*map(x->R(x,θ1),knodes[jmid:end]));
    end
    j=length(partitions);
    knodes = ((nodes+1)*(partitions[j]-partitions[j-1])/2 + partitions[j-1]);
    kweights = weights*(partitions[j]-partitions[j-1])/2;
    resvec[j-1] = sum((ψuplhs(knodes,θ1,j)/θ1[j-1]).*map(x->R(x,θ1),knodes));
    return resvec;
end

