using SumOfSquares
using MosekTools

include("compoundmatrices.jl")
include("matrix_group.jl")


function make_symmetric_basis(x, zk, symmetry_group, degx, degz)
    raw_basis = kron(monomials(x, 0:degx), monomials(zk, 0:degz))

    #raw_basis = filter(p -> degree(p) <= max(degx, degz), raw_basis)

    return invariant_polynomials(symmetry_group, MatrixAction([x; zk]), raw_basis)
end

function filter_basis(x, zk, basis, degx, degz)
    return [
        p for p in basis if maximum(sum(maxdegree(term, xvar) for xvar in x) for term in p) <= degx && 
                            maximum(sum(maxdegree(term, zvar) for zvar in zk) for term in p)  <= degz# &&
        #                    maxdegree(p) <= max(degx, degz)
    ]
end

function optimise_bound(k, degreex, degreez)
    d=3 # Hamiltonian DoFs

    @polyvar p[1:d] q[1:d]

    x = [q;p]
    n=length(x)

    @polyvar zk[1:binomial(n,k)]

    α = 0
    β = 1
    H(x) = 0.5*(x[4]^2+x[5]^2+x[6]^2+x[1]^2+x[2]^2+x[3]^2) + α*(x[1]^4+x[2]^4+x[3]^4) + β*(x[1]^2*x[2]^2 + x[2]^2*x[3]^2 + x[3]^2*x[1]^2)

    E = 1 # energy level

    J = [zeros(d,d) I; -I zeros(d,d)]
    f = J*differentiate(H(x),x) # standard Hamiltonian system

    Df = differentiate(f,x)

    Phik = dot(zk, add_comp(Df, k)*zk)

    X = [x; zk] # augmented state vector
    F = [f; add_comp(Df, k)*zk - Phik*zk] # augmented system


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
    model = SOSModel(MosekTools.Optimizer)

    rho3_basis = make_symmetric_basis(x, zk, symmetry_group, degreex+2, degreez)
    rho2_basis = make_symmetric_basis(x, zk, symmetry_group, degreex, degreez+2)
    rho1_basis = filter_basis(x, zk, rho2_basis, degreex-2, degreez+2)
    V_basis = filter_basis(x, zk, rho3_basis, degreex, degreez)

    V = dot(V_basis, @variable(model, [1:length(V_basis)])) # Lyapunov function
    rho1 = dot(rho1_basis, @variable(model, [1:length(rho1_basis)])) # s procedure function
    rho2 = dot(rho2_basis, @variable(model, [1:length(rho2_basis)])) # nonnegative s procedure function
    rho3 = dot(rho3_basis, @variable(model, [1:length(rho3_basis)])) # s procedure function

    LV = dot(differentiate(V, X), F) # Lie derivative of Lyapunov function

    pattern = Symmetry.Pattern(symmetry_group, MatrixAction(X))
    B = @variable(model)
    @constraint(model, B - Phik + LV >= rho1*(H(x)-E) + rho2*(2 - dot(x,x)) + rho3*(1 - dot(zk,zk)), sparsity=Sparsity.SignSymmetry())
    @constraint(model, rho2>=0, sparsity=Sparsity.SignSymmetry())
    @objective(model, Min, B)

    optimize!(model)

    display(solution_summary(model))
    display(value(B))
end


# optimise_bound(1,2,0) 1.2360681787742616
# optimise_bound(1,2,2) 0.4842745455169288
# optimise_bound(1,4,0) 1.1854435641961132
# optimise_bound(1,4,2) 0.42913564721944875
# optimise_bound(1,4,4) 0.42913028768618855

# optimise_bound(2,2,0) 2.023140372251364
# optimise_bound(2,2,2) 0.9215679421723011
# optimise_bound(2,4,0) 2.0045758968988387

# optimise_bound(3,2,0) 2.1081942641530804
# optimise_bound(3,4,0) 2.061016648162922
