using GroupsCore
using SumOfSquares
using LinearAlgebra
using SymbolicWedderburn
import StarAlgebras
using DynamicPolynomials

# This is all messy but essentially it's a layer for using matrix groups as symmetry groups in the SymbolicWedderburn.jl package.

struct MatGrpElement{T} <: GroupElement
    matrix::Matrix{T}
    G
end
Base.:(==)(a::MatGrpElement, b::MatGrpElement) = isapprox(a.matrix, b.matrix, atol=1e-4, rtol=1e-4)
Base.inv(el::MatGrpElement{T}) where T = MatGrpElement{T}(Matrix{T}(inv(el.matrix)), el.G)
Base.parent(el::MatGrpElement) = el.G
Base.:*(a::MatGrpElement, b::MatGrpElement) = MatGrpElement(a.matrix*b.matrix, a.G)
Base.:^(el::MatGrpElement, k::Integer) = MatGrpElement(el.matrix^k, el.G)
Base.conj(a::MatGrpElement, b::MatGrpElement) = inv(b) * a * b
Base.:^(a::MatGrpElement, b::MatGrpElement) = conj(a, b)
Base.deepcopy_internal(el::MatGrpElement, ::IdDict) = MatGrpElement(deepcopy(el.matrix), el.G)

struct MatrixGroup{T} <: Group
    generators::Vector{MatGrpElement{T}}
    elements::Vector{MatGrpElement{T}}
end
Base.eltype(::MatrixGroup{T}) where T = MatGrpElement{T}
Base.one(G::MatrixGroup) = MatGrpElement(one(G.generators[1].matrix), G)
Base.one(c::MatGrpElement) = MatGrpElement(one(c.matrix), c.G)
Base.isfinite(::MatrixGroup) = true
GroupsCore.gens(c::MatrixGroup) = c.generators
GroupsCore.order(::Type{T}, c::MatrixGroup) where {T} = convert(T, length(c.elements))
Base.iterate(c::MatrixGroup, state) = Base.iterate(c.elements, state)
Base.iterate(c::MatrixGroup) = Base.iterate(c.elements)

struct MatrixAction{V} <: Symmetry.OnMonomials
    variables::Vector{V}
end
Symmetry.SymbolicWedderburn.coeff_type(::MatrixAction) = Float64

function Symmetry.SymbolicWedderburn.action(a::MatrixAction, el::MatGrpElement, mono::AbstractMonomial)
    return subs(mono, a.variables=>el.matrix*a.variables)
end

function MatrixGroup(generators::Vector{Matrix{T}}) where T
    symmetry_group = [[one(generators[1])]; generators]
    symmetry_group_old = []
    while symmetry_group != symmetry_group_old
        symmetry_group_old = copy(symmetry_group)
        symmetry_group = [
            symm1*symm2 for symm1 in symmetry_group for symm2 in symmetry_group
        ]
        unique!(symmetry_group)
    end
    G = MatrixGroup{T}([], [])
    append!(G.generators, [MatGrpElement{T}(symm, G) for symm in generators])
    append!(G.elements, [MatGrpElement{T}(symm, G) for symm in symmetry_group])
    return G
end

function invariant_polynomials(G::MatrixGroup{T}, action::MatrixAction, raw_polynomials) where T
    tbl = SymbolicWedderburn.CharacterTable(Rational{Int}, G)
    invariants = invariant_vectors(tbl, action, StarAlgebras.Basis{UInt32}(raw_polynomials))
    return [dot(v, raw_polynomials) for v in invariants]
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
