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
    @polyvar wk[1:binomial(n,k)]

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

    X = [x; wk] # augmented state vector
    F = [f; add_comp(Df, k)*wk - B*wk] # augmented system


    symmetry_generators = Vector{Matrix{Int}}([
        [-1 0 0 0;
          0 -1 0 0;
          0 0 -1 0;
          0 0 0 -1
        ]])

    symmetry_group = MatrixGroup(make_tangent_symmetries(n, k, symmetry_generators))

    println("Symmetry group has $(length(symmetry_group)) elements")

    # Define the SOS program
    model = SOSModel(MosekTools.Optimizer)

    V_basis = make_symmetric_basis(x, wk, symmetry_group, degree)
    rho1_basis = make_symmetric_basis(x, wk, symmetry_group, degree)

    V = dot(V_basis, @variable(model, [1:length(V_basis)])) # Lyapunov function
    rho1a = dot(rho1_basis, @variable(model, [1:length(rho1_basis)])) # s procedure function
    rho1b = dot(rho1_basis, @variable(model, [1:length(rho1_basis)])) # s procedure function

    LV = dot(differentiate(V, X), F) # Lie derivative of Lyapunov function

    @constraint(model, V >= dot(wk, wk) + rho1a*(x[3]^2+x[4]^2-1), sparsity=Sparsity.SignSymmetry())
    @constraint(model, -LV >= rho1b*(x[3]^2+x[4]^2-1), sparsity=Sparsity.SignSymmetry())
    optimize!(model)

    display(solution_summary(model))
end

# at degree=4, the results are messy
# try_bound(0.887898, 1, 6) feasible
# try_bound(0.887897, 1, 6) infeasible
# same for k=2,4 (but slow for k=2)

# try_bound(-0.2, 4, 0) feasible
