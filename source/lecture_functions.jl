module lecture_functions
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
# < Description >
#
# Cleft = updateLeft(Cleft,rankC,B,X,rankX,A)
#
# Contract the operator Cleft that act on the Hilbert space of the left
# part of the MPS [i.e., left of a given site] with the tensors B, X, &
# A; acting on the given site.
#
# < Input >
# Cleft : [tensor] Rank-2 | 3 tensor from the left part of the system. If
#       given as empty [], then Cleft is considered as the identity tensor
#       of rank 2 [for rank(X) .< 4] | rank 3 [for rank(X) .== 4].
# rankC : [integer] Rank of Cleft.
# B, A : [tensors] Ket tensors, whose legs are ordered as left - bottom
#       (local physical) - right. In the contraction, the Hermitian
#       conjugate [i.e., bra form] of B is used, while A is contracted as
#       it is. This convention of inputting B as a ket tensor reduces extra
#       computational cost of taking the Hermitian conjugate of B.
# X : [tensor] Local operator with rank 2 | 3. If given as empty [], then
#       X is considered as the identity.
# rankX : [integer] Rank of X.
#
# < Output >
# Cleft : [tensor] Contracted tensor. The tensor network diagrams
#       describing the contraction are as follows.
#       * When Cleft is rank-3 & X is rank-2:
#                    1     3
#          /--------->- A ->--            /---->-- 3
#          |            | 2               |
#        3 ^            ^                 |
#          |    2       | 2               |      
#        Cleft---       X         =>    Cleft ---- 2
#          |            | 1               |
#        1 ^            ^                 |
#          |            | 2               |
#          \---------<- B'-<--            \----<-- 1
#                    3     1
#       * When Cleft is rank-2 & X is rank-3:
#                    1     3
#          /--------->- A ->--            /---->-- 3
#          |            | 2               |
#        2 ^            ^                 |
#          |          3 |   2             |      
#        Cleft          X ----    =>    Cleft ---- 2
#          |          1 |                 |
#        1 ^            ^                 |
#          |            | 2               |
#          \---------<- B'-<--            \----<-- 1
#                    3     1
#       * When both Cleft & X are rank-3:
#                    1     3
#          /--------->- A ->--            /---->-- 2
#          |            | 2               |
#        3 ^            ^                 |
#          |   2     2  | 3               |      
#        Cleft--------- X         =>    Cleft
#          |            | 1               |
#        1 ^            ^                 |
#          |            | 2               |
#          \---------<- B'-<--            \----<-- 1
#                    3     1
#       * When Cleft is rank3 & X is rank-4:
#                    1     3
#          /--------->- A ->--            /---->-- 3
#          |            | 2               |
#        3 ^            ^                 |
#          |   2    1   | 4               |      
#        Cleft--------- X ---- 3   =>   Cleft ---- 2
#          |            | 2               |
#        1 ^            ^                 |
#          |            | 2               |
#          \---------<- B'-<--            \----<-- 1
#                    3     1
#       Here B' denotes the Hermitian conjugate [i.e., complex conjugate
#       & permute legs by [3 2 1]] of B.
#
# Written by H.Tu [May 3,2017]; edited by S.Lee [May 19,2017]
# Rewritten by S.Lee [May 5,2019]
# Updated by S.Lee [May 27,2019]: Case of rank-3 Cleft & rank-4 X is()
#       added.
# Updated by S.Lee [Jul.28,2020]: Minor fix for the case when Cleft .== []
#       & rank(X) .== 4.
# Transformed by Changkai Zhang into Julia [May 4, 2022]

    # error checking
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
