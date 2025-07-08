module contractions
export contract, updateLeft
using LinearAlgebra

"""
    contract(A, Aindices, B, Bindices, indexpermutation=[])

Contract tensors `A` and `B`.
The legs to be contracted are given by `Aindices` and `Bindices`.

Input:
- `A`, `B` (numeric arrays): the tensors to be contracted.
- `Aindices`, `Bindices` (ordered collection of integers):
    Indices for the legs of `A` and `B` to be contracted. The `Aindices[n]`-th leg of `A`
    will be contracted with the `Bindices[n]`-th leg of `B`. `Aindices` and `Bindices`
    should have the same size. If they are both empty, the result will be the direct product
    of `A` and `B`.
- `indexpermutation` (ordered collection of integers):
    Permutation of of the output tensor's indices to be performed after contraction.

Output:
- Contraction of `A` and `B`. All non-contracted legs are in the following order: first, all
    legs previously attached to `A`, in order, then all legs previously attached to `B`, in
    order.
"""
function contract(A, Aindices, B, Bindices, indexpermutation=[])
    if length(Aindices) != length(Bindices)
        error("Different number of legs to contract for tensors A and B.")
    end
    if any(Aindices .< 1) || any(Aindices .> ndims(A))
        error("Got indices out of range for tensor A in contract().")
    end
    if any(Bindices .< 1) || any(Bindices .> ndims(B))
        error("Got indices out of range for tensor B in contract().")
    end

    # indices of legs *not* to be contracted
    Arestinds = setdiff(1:ndims(A), Aindices)
    Brestinds = setdiff(1:ndims(B), Bindices)

    # reshape tensors into matrices with "thick" legs
    A2 = reshape(
        permutedims(A, tuple(cat(dims=1, Arestinds, Aindices))...),
        (prod(size(A)[Arestinds]), prod(size(A)[Aindices]))
    )
    B2 = reshape(
        permutedims(B, tuple(cat(dims=1, Bindices, Brestinds))...),
        (prod(size(B)[Bindices]), prod(size(B)[Brestinds]))
    )
    C2 = A2 * B2 # matrix multiplication

    # size of C
    Cdim = (size(A)[Arestinds]..., size(B)[Brestinds]...)
    # reshape matrix to tensor
    C = reshape(C2, Cdim)

    if !isempty(indexpermutation) # if permutation option is given
        C = permutedims(C, indexpermutation)
    end

    return C
end

"""
    updateLeft(C, B, X, A)

Obtain the operator Cleft that acts on the Hilbert space to the left of an MPS site through
contraction, in the following pattern,
```
    .-A-- 3
    | |
    C-X-- 2
    | |
    '-B*- 1
```
where B* is the complex conjugate of B. Note the leg ordering (bottom-to-top).

Input:
- `C` (3-leg tensor): environment to the left of the current site.
- `B`, `A` (3-leg tensors): ket tensors on the current site.
- `X` (4-leg tensor): local operator. # ordering: left bottom right top

Output:
- Tensor corresponding to the fully contracted tensor network shown above.
"""
function updateLeft(C, B, X, A)
    # Checking dimensionality errors
    if ndims(C) != 3
        error("In updateLeft, got parameter C with $(ndims(C)) dimensions. C must have 3 dimensions.")
    end
    if ndims(B) != 3
        error("In updateLeft, got parameter B with $(ndims(B)) dimensions. B must have 3 dimensions.")
    end
    if ndims(X) != 4
        error("In updateLeft, got parameter X with $(ndims(X)) dimensions. X must have 4 dimensions.")
    end
    if ndims(A) != 3
        error("In updateLeft, got parameter A with $(ndims(A)) dimensions. A must have 3 dimensions.")
    end

    CA = contract(C, [3], A, [1])
    CAX = contract(CA, [2, 3], X, [1, 4])
    return contract(conj(B), [1, 2], CAX, [1, 3], [1, 3, 2])
end

end