using SumOfSquares
using MosekTools

include("compoundmatrices.jl")
include("matrix_group.jl")

function make_symmetric_basis(x, zk, symmetry_group, degx, degz)
    raw_basis = kron(monomials(x, 0:degx), monomials(zk, 0:degz))

    raw_basis = filter(p -> degree(p) <= max(degx, degz), raw_basis)

    return invariant_polynomials(symmetry_group, MatrixAction([x; zk]), raw_basis)
end

function try_bound(B, k, degreex, degreez)
    n = 2
    @polyvar x[1:n]
    @polyvar zk[1:binomial(n,k)]

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

    # these will get replaced later
    @polyvar Dgz2
    @polyvar invsqrtDgz2

    Dg = differentiate(g(x), x)

    X = [x; zk] # augmented state vector
    G = [g(x); mult_comp(Dg, k)*zk * invsqrtDgz2] # augmented system

    p(x) = 0.061-(0.14*((x[1]-x[2]+0.025)^2-0.94)^2+0.7*(0.59+x[1]-x[2]+0.035)^2*(x[1]+x[2]-0.48)^2-0.052*(x[1]-x[2]+0.035))

    symmetry_group = MatrixGroup(make_tangent_symmetries(n, k, Vector{Matrix{Int}}()))

    W_basis = make_symmetric_basis(x, zk, symmetry_group, degreex, degreez)
    W = dot(W_basis, @variable(model, [1:length(W_basis)])) # Lyapunov function
    WG = W(X=>G)

    P = (exp(2*B)*W - Dgz2*WG) * Dgz2^(degreez÷2-1) 

    # we now need to replace Dgz2 * invsqrtDgz2^2 with 1
    newmonos = Vector{eltype(monomials(P))}()
    for mono in monomials(P) # we do this termwise
        deg = degree(mono, invsqrtDgz2)

        @assert(mod(deg,2)==0)
        @assert(degree(mono, Dgz2) >= deg÷2)

        if deg>0
            newpowers = copy(mono.z)

            newpowers[findall(x->x==Dgz2,mono.vars)[1]] -= deg÷2
            newpowers[findall(x->x==invsqrtDgz2,mono.vars)[1]] -= deg

            newmono = prod(mono.vars.^newpowers)
        else
            newmono = copy(mono)
        end

        push!(newmonos, newmono)
    end
    P = dot(newmonos, P.a)
    @assert(maxdegree(P, invsqrtDgz2) == 0) # we should have removed all invsqrtDgz2

    # then clean up any remaining Dgz2
    P = subs(P, Dgz2=>dot(mult_comp(Dg, k)*zk, mult_comp(Dg, k)*zk))


    degx = maximum(maxdegree(P, var) for var in x)
    degz = maximum(maxdegree(P, var) for var in zk)
    rho1_basis = make_symmetric_basis(x, zk, symmetry_group, degx-4, degz)
    rho0_basis = make_symmetric_basis(x, zk, symmetry_group, degx, degz-2)

    rho1 = dot(rho1_basis, @variable(model, [1:length(rho1_basis)]))
    rho0 = dot(rho0_basis, @variable(model, [1:length(rho0_basis)]))

    display(P)
    @constraint(model, P >= rho0*(1-dot(zk,zk)) + rho1*p(rescale*x/2), sparsity = Sparsity.SignSymmetry())
    @constraint(model, W >= 1, sparsity = Sparsity.SignSymmetry())
    @constraint(model, rho1 >= 0, sparsity = Sparsity.SignSymmetry())

    optimize!(model)

    display(solution_summary(model))
end

# try_bound(0.279, 1, 6, 2) reports feasible
# try_bound(0.278, 1, 6, 2) does not

# try_bound(0.27454, 1, 8, 2) reports feasible
# try_bound(0.27453, 1, 8, 2) does not

# try_bound(-0.22, 2, 4, 2) reports feasible
# try_bound(-0.23, 2, 4, 2) does not

# try_bound(-0.26421, 2, 6, 2) reports feasible
# try_bound(-0.26422, 2, 6, 2) does not