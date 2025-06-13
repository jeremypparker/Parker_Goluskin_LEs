using SumOfSquares
using Clarabel

include("compoundmatrices.jl")
include("matrix_group.jl")

"""
    make_symmetric_basis(X, symmetry_group, deg)
Constructs a basis of symmetric of degree `deg` in `x` and degree 2 in `wk` that are invariant under the given `symmetries`.
"""
function make_symmetric_basis(x, wk, symmetry_group, deg)
    raw_basis = kron(monomials(x, 0:deg), monomials(wk, 2:2))
    return invariant_polynomials(symmetry_group, MatrixAction([x; wk]), raw_basis)
end


"""
    make_tangent_symmetries(n, k, symmetries)
Given a set of symmetry generators for the basic dynamical system, expands them to the full augmented system.
"""
function make_tangent_symmetries(n, k, symmetries)
    T = eltype(symmetries[1])
    expandedsymmetries = Vector{Matrix{Int}}()

    m = binomial(n,k)
    
    signsymmetry = [I zeros(Int, n,m); zeros(Int, m,n) -I] # additional symmetry for the linear tangent dynamics

    push!(expandedsymmetries, signsymmetry)


    for symmetry in symmetries
        @assert(n == size(symmetry,1))
        @assert(n == size(symmetry,2))

        expandedsymmetry = zeros(T, n + m, n + m)
        expandedsymmetry[1:n,1:n] = symmetry
        expandedsymmetry[n+1:end,n+1:end] = mult_comp(symmetry, k)

        push!(expandedsymmetries, expandedsymmetry)
    end
      
    return expandedsymmetries
end


function try_bound(B, k, degree)
    d=3 # Hamiltonian DoFs

    @polyvar p[1:d] q[1:d]

    x = [q;p]
    n=length(x)

    @polyvar wk[1:binomial(n,k)]

    α = 0
    β = 1
    H(x) = 0.5*(x[4]^2+x[5]^2+x[6]^2+x[1]^2+x[2]^2+x[3]^2) + α*(x[1]^4+x[2]^4+x[3]^4) + β*(x[1]^2*x[2]^2 + x[2]^2*x[3]^2 + x[3]^2*x[1]^2)

    E = 1 # energy level

    J = [zeros(d,d) I; -I zeros(d,d)]
    f = J*differentiate(H(x),x) # standard Hamiltonian system

    Df = differentiate(f,x)

    X = [x; wk] # augmented state vector
    F = [f; add_comp(Df, k)*wk - B*wk] # augmented system


    symmetry_generators = Vector{Matrix{Int}}([
        [[-1 0 0 0 0 0;
          0 1 0 0 0 0;
          0 0 1 0 0 0;
          0 0 0 -1 0 0;
          0 0 0 0 1 0;
          0 0 0 0 0 1]];
        [[0 1 0 0 0 0;
          1 0 0 0 0 0;
          0 0 1 0 0 0;
          0 0 0 0 1 0;
          0 0 0 1 0 0;
          0 0 0 0 0 1]];
        [[0 0 1 0 0 0;
          0 1 0 0 0 0;
          1 0 0 0 0 0;
          0 0 0 0 0 1;
          0 0 0 0 1 0;
          0 0 0 1 0 0]];
        ])

    # Check these are valid Hamiltonian symmetries
    for Γ in symmetry_generators
        @assert J == Γ'*J*Γ
        @assert H(x) == H(Γ*x)
    end

    symmetry_group = MatrixGroup(make_tangent_symmetries(n, k, symmetry_generators))

    println("Symmetry group has $(length(symmetry_group)) elements")

    # Define the SOS program
    model = SOSModel(Clarabel.Optimizer)

    V_basis = make_symmetric_basis(x, wk, symmetry_group, degree)
    rho1_basis = make_symmetric_basis(x, wk, symmetry_group, max(degree-4, 0))
    rho2_basis = make_symmetric_basis(x, wk, symmetry_group, max(degree-2, 0))

    V = dot(V_basis, @variable(model, [1:length(V_basis)])) # Lyapunov function
    rho1a = dot(rho1_basis, @variable(model, [1:length(rho1_basis)])) # s procedure function
    rho2a = dot(rho2_basis, @variable(model, [1:length(rho2_basis)])) # s procedure function
    rho1b = dot(rho1_basis, @variable(model, [1:length(rho1_basis)])) # s procedure function
    rho2b = dot(rho2_basis, @variable(model, [1:length(rho2_basis)])) # s procedure function

    LV = dot(differentiate(V, X), F) # Lie derivative of Lyapunov function

    pattern = Symmetry.Pattern(symmetry_group, MatrixAction(X))
    @constraint(model, V >= dot(wk, wk) + rho1a*(H(x)-E) + rho2a*(2 - dot(x,x)), symmetry=pattern)
    @constraint(model, -LV >= rho1b*(H(x)-E) + rho2b*(2 - dot(x,x)), symmetry=pattern)
    @constraint(model, rho2a>=0, symmetry=pattern)
    @constraint(model, rho2b>=0, symmetry=pattern)

    optimize!(model)

    display(solution_summary(model))
end

# try_bound(0.4291, 1, 4) # infeasible
# try_bound(0.4292, 1, 4) # feasible

