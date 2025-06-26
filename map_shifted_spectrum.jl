using SumOfSquares
using MosekTools

include("compoundmatrices.jl")
include("matrix_group.jl")

function make_symmetric_basis(x, wk, symmetry_group, degx)
    raw_basis = kron(monomials(x, 0:degx), monomials(wk, 2:2))
    return invariant_polynomials(symmetry_group, MatrixAction([x; wk]), raw_basis)
end

function try_bound(B, k, degreex)
    n = 2
    @polyvar x[1:n]
    @polyvar wk[1:binomial(n,k)]

    model = SOSModel(MosekTools.Optimizer)

    rescale = 1
    function g(x)
        a1=0
        a2=-0.73
        a3=0*rescale
        a4=-0.37*rescale
        a5=0.81*rescale
        a6=1.79/rescale

        return [x[2]; a1*x[1]+a2*x[2]+a3*x[1]^2+a4*x[2]^2+a5*x[1]*x[2]+a6]
    end

    Dg = differentiate(g(x), x)

    X = [x; wk] # augmented state vector
    G = [g(x); mult_comp(Dg, k)*wk*exp(-B)] # augmented system

    p(x) = 0.061-(0.14*((x[1]-x[2]+0.025)^2-0.94)^2+0.7*(0.59+x[1]-x[2]+0.035)^2*(x[1]+x[2]-0.48)^2-0.052*(x[1]-x[2]+0.035))

    symmetry_group = MatrixGroup(make_tangent_symmetries(n, k, Vector{Matrix{Int}}()))

    V_basis = make_symmetric_basis(x, wk, symmetry_group, degreex)
    V = dot(V_basis, @variable(model, [1:length(V_basis)])) # Lyapunov function
    LV = V(X=>G) - V

    rho1_basis = make_symmetric_basis(x, wk, symmetry_group, degreex-4)

    rho1a = dot(rho1_basis, @variable(model, [1:length(rho1_basis)]))
    rho1b = dot(rho1_basis, @variable(model, [1:length(rho1_basis)]))

    @constraint(model, V - dot(wk,wk) >= rho1a*p(rescale*x/2), sparsity = Sparsity.SignSymmetry())
    @constraint(model, -LV >= rho1b*p(rescale*x/2), sparsity = Sparsity.SignSymmetry())
    @constraint(model, rho1a >= 0, sparsity = Sparsity.SignSymmetry())
    @constraint(model, rho1b >= 0, sparsity = Sparsity.SignSymmetry())

    optimize!(model)

    display(solution_summary(model))
end

# I ought to rerun these at higher precision:

# try_bound(0.59, 1, 4) reports feasible
# try_bound(0.58, 1, 4) does not

# try_bound(0.40, 1, 6) reports feasible
# try_bound(0.39, 1, 6) does not

# try_bound(0.280, 1, 8) # reports feasible
# try_bound(0.279, 1, 8) # does not

# try_bound(-0.03, 2, 4) # reports feasible
# try_bound(-0.04, 2, 4) # does not

# try_bound(-0.25, 2, 6) # reports feasible
# try_bound(-0.26, 2, 6) # does not

# try_bound(-0.26422, 2, 8) # reports feasible
# try_bound(-0.26423, 2, 8) # does not