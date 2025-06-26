using SumOfSquares
using MosekTools

include("compoundmatrices.jl")
include("matrix_group.jl")

function make_symmetric_basis(x, wk, symmetry_group, deg)
    raw_basis = kron(monomials(x, 0:deg), monomials(wk, 2:2))
    return invariant_polynomials(symmetry_group, MatrixAction([x; wk]), raw_basis)
end

function try_bound(B, k, degree)
    n = 4
    @polyvar x[1:n]
    @polyvar y[1:binomial(n,k+1)]
    @polyvar w[1:binomial(n,k)]

    α = 1
    β = -1
    γ = 0.3
    δ = 0.2
    ω = 1

    rescale = 1.5
    f = [x[2];
         -δ*x[2]-β*x[1]-α*x[1]^3*rescale*rescale + γ*x[4]/rescale;
         ω*x[4];
         -ω*x[3]];

    Df = differentiate(f,x)

    X = [x; y; w] # augmented state vector
    y2F = [dot(y,y)*f; 
           dot(y,y)*add_comp(Df, k+1)*y; 
           dot(y,y)*add_comp(Df, k)*w + (B/(1-B)) * dot(y, add_comp(Df, k+1)*y)*w] # augmented system rescaled by |y|^2


    symmetry_generators = Vector{Matrix{Int}}([
        [-1 0 0 0;
          0 -1 0 0;
          0 0 -1 0;
          0 0 0 -1
        ]])

    symmetry_group = MatrixGroup(make_2_tangent_symmetries(n, k+1, k, symmetry_generators))

    println("Symmetry group has $(length(symmetry_group)) elements")

    # Define the SOS program
    model = SOSModel(MosekTools.Optimizer)

    V_basis = make_symmetric_basis([x;y], w, symmetry_group, degree)
    rho1_basis = make_symmetric_basis([x;y], w, symmetry_group, degree)

    V = dot(V_basis, @variable(model, [1:length(V_basis)])) # Lyapunov function
    rho1a = dot(rho1_basis, @variable(model, [1:length(rho1_basis)])) # s procedure function
    rho1b = dot(rho1_basis, @variable(model, [1:length(rho1_basis)])) # s procedure function

    y2LV = dot(differentiate(V, X), y2F) # Lie derivative of Lyapunov function

    @constraint(model, V >= dot(w, w) + rho1a*(x[3]^2+x[4]^2-1), sparsity=Sparsity.SignSymmetry())
    @constraint(model, -y2LV >= rho1b*(x[3]^2+x[4]^2-1), sparsity=Sparsity.SignSymmetry())
    optimize!(model)

    display(solution_summary(model))
end

# try_bound(0.8162, 3, 6) reports feasible
# try_bound(0.8161, 3, 6) does not