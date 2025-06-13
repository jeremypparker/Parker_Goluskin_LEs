using LinearAlgebra
using Combinatorics
using DynamicPolynomials

"""
    mult_comp(A, k)
    Computes the `k` th multiplicative compound matrix of a square matrix `A`.
"""
function mult_comp(A, k)
    n = size(A,1)
    @assert(size(A,2)==n)

    m = binomial(n, k)

    C = zeros(eltype(A), m, m)

    indices = sort!(collect(powerset(1:n, k, k))) # lexicographically ordered indices for subdeterminants
    @assert(length(indices) == m)

    for j = 1:m
        for i = 1:m
            C[i,j] = det(A[indices[i],indices[j]])
        end
    end

    return C
end

"""
    add_comp(A, k)
    Computes the `k` th additive compound matrix of a square matrix `A`,
    via symbolic differentiation of the multiplicative compound matrix.
"""
function add_comp(A, k)
    @polyvar δ
    differentiable = mult_comp(I+δ*A, k)
    derivative = differentiate.(differentiable, δ)
    return subs(derivative, δ=>0)
end