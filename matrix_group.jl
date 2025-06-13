using GroupsCore
import PermutationGroups
using SumOfSquares
using LinearAlgebra
using SymbolicWedderburn
import StarAlgebras

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

function PermutationGroups.order(el::MatGrpElement)
    for i=1:100
        if el^i == one(el) # stupid but it works
            return i
        end
    end
    throw("Could not compute order")
end

struct MatrixGroup{T} <: Group
    generators::Vector{MatGrpElement{T}}
    elements::Vector{MatGrpElement{T}}
end
Base.eltype(::MatrixGroup{T}) where T = MatGrpElement{T}
Base.one(G::MatrixGroup) = MatGrpElement(one(G.generators[1].matrix), G)
Base.one(c::MatGrpElement) = MatGrpElement(one(c.matrix), c.G)
Base.isfinite(::MatrixGroup) = true
PermutationGroups.gens(c::MatrixGroup) = c.generators
PermutationGroups.order(::Type{T}, c::MatrixGroup) where {T} = convert(T, length(c.elements))
Base.iterate(c::MatrixGroup, state) = Base.iterate(c.elements, state)
Base.iterate(c::MatrixGroup) = Base.iterate(c.elements)

import MultivariatePolynomials as MP
struct MatrixAction{V<:MP.AbstractVariable} <: Symmetry.OnMonomials
    variables::Vector{V}
end
Symmetry.SymbolicWedderburn.coeff_type(::MatrixAction) = Float64

function Symmetry.SymbolicWedderburn.action(a::MatrixAction, el::MatGrpElement, mono::MP.AbstractMonomial)
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



# See https://github.com/jump-dev/SumOfSquares.jl/issues/387
import SumOfSquares.Symmetry: block_diag, isblockdim, ordered_block_check, ordered_block_diag
function Symmetry.ordered_block_diag(As, d)
    function clean!(M, tol=1e-15)
        M[abs.(M) .< tol] .= 0
    end

    for A in As
        clean!(A)
    end

    U = block_diag(As, d)
    isnothing(U) && return nothing
    iU = U'
    @assert iU ≈ inv(U)
    Bs = [iU * A * U for A in As]
    @assert all(Bs) do B
        return isblockdim(B, d)
    end
    refs = [B[1:d, 1:d] for B in Bs]
    offset = d
    for offset in d:d:(size(U, 1)-d)
        I = offset .+ (1:d)
        Cs = [B[I, I] for B in Bs]
        λ = rand(length(Bs))
        # We want to find a transformation such that
        # the blocks `Cs` are equal to the blocks `refs`
        # With probability one, making a random combination match
        # should work, this trick is similar to [CGT97].
        #
        # [CGT97] Corless, R. M.; Gianni, P. M. & Trager, B. M.
        # A reordered Schur factorization method for zero-dimensional polynomial systems with multiple roots
        # Proceedings of the 1997 international symposium on Symbolic and algebraic computation,
        # 1997, 133-140
        R = sum(λ .* refs)
        C = sum(λ .* Cs)
        V = orthogonal_transformation_to(R, C)
        @assert R ≈ V' * C * V
        for i in eachindex(refs)
            @assert refs[i] ≈ V' * Cs[i] * V
        end
        U[:, I] = U[:, I] * V
        offset += d
    end
    @assert ordered_block_check(U, As, d)
    return U
end