using SumOfSquares
using MosekTools

include("compoundmatrices.jl")
include("matrix_group.jl")

function make_symmetric_basis(x, zk, zk1, symmetry_group, degx, degz)
    raw_basis = kron(monomials(x, 0:degx), monomials([zk; zk1], 0:degz))

    raw_basis = filter(p -> degree(p) <= max(degx, degz), raw_basis)

    return invariant_polynomials(symmetry_group, MatrixAction([x; zk; zk1]), raw_basis)
end

function optimise_bound(k, degreex, degreez)
    n = 4
    @polyvar x[1:n]
    @polyvar zk[1:binomial(n,k)]
    @polyvar zk1[1:binomial(n,k+1)]

    α = 1
    β = -1
    γ = 0.3
    δ = 0.2
    ω = 1
    f = [x[2];
         -δ*x[2]-β*x[1]-α*x[1]^3 + γ*x[4];
         ω*x[4];
         -ω*x[3]];

    Df = differentiate(f,x)

    Phik = dot(zk, add_comp(Df, k)*zk)
    Phik1 = dot(zk1, add_comp(Df, k+1)*zk1)

    X = [x; zk; zk1] # augmented state vector
    F = [f; add_comp(Df, k)*zk - Phik*zk; add_comp(Df, k+1)*zk1 - Phik1*zk1] # augmented system


    symmetry_generators = Vector{Matrix{Int}}([
        [-1 0 0 0;
          0 -1 0 0;
          0 0 -1 0;
          0 0 0 -1
        ]])

    symmetry_group = MatrixGroup(make_2_tangent_symmetries(n, k, k+1, symmetry_generators))

    println("Symmetry group has $(length(symmetry_group)) elements")

    # Define the SOS program
    model = SOSModel(MosekTools.Optimizer)

    V_basis = make_symmetric_basis(x, zk, zk1, symmetry_group, degreex, degreez)
    rho1_basis = make_symmetric_basis(x, zk, zk1, symmetry_group, degreex, degreez)
    rho2_basis = make_symmetric_basis(x, zk, zk1, symmetry_group, degreex, degreez)

    V = dot(V_basis, @variable(model, [1:length(V_basis)])) # Lyapunov function
    rho1 = dot(rho1_basis, @variable(model, [1:length(rho1_basis)])) # s procedure function
    rho2 = dot(rho2_basis, @variable(model, [1:length(rho2_basis)])) # s procedure function
    rho3 = dot(rho2_basis, @variable(model, [1:length(rho2_basis)])) # s procedure function

    LV = dot(differentiate(V, X), F) # Lie derivative of Lyapunov function

    @variable(model, B >= 0)
    @constraint(model, B*(Phik-Phik1) - Phik + LV >= rho1*(x[3]^2+x[4]^2-1) + rho2*(1 - dot(zk,zk)) + rho3*(1 - dot(zk1,zk1)), sparsity=Sparsity.SignSymmetry())
    @objective(model, Min, B)
    optimize!(model)

    display(solution_summary(model))
    display(value(B))
end

# optimise_bound(3, 2, 2) 0.9999246874355149
# optimise_bound(3, 4, 2) 0.9991027769820908
# optimise_bound(3, 4, 2) 0.8236075679377035
# optimise_bound(3, 4, 4) 0.8161607655168013
# optimise_bound(3, 6, 2) 0.8161588998086654
# optimise_bound(3, 6, 4) 0.8161589808323491