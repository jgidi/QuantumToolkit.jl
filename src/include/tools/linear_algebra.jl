export commutator, commutator!
export trn, issemiposdef, nearest_posdef, nearest_density, ptrace


"""
    commutator!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)

Computes the commutator `[A, B] = AB - BA` of the operators `A` and `B`,
and save it in place on `C`.

For an allocating version of this function see [`commutator`](@ref).
"""
function commutator!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    mul!(C, A, B)
    mul!(C, B, A, -1, 1)

    return C
end

"""
    commutator(A::AbstractMatrix, B::AbstractMatrix)

Returns the commutator `[A, B] = AB - BA` of the operators `A` and `B`.

For an in-place version of this function see [`commutator!`](@ref).
"""
function commutator(A::AbstractMatrix, B::AbstractMatrix)
    T = promote_type(typeof(A), typeof(B))
    C = similar(T, size(A))
    commutator!(C, A, B)

    return C
end


"""
    trn(A::AbstractMatrx)

Returns the trace of `A` normalized to the interval `[0, 1]`.

Notes
=====
* `trn(A) = (d*tr(A) - 1)/(d - 1)` where `d = size(A, 1)`.
"""
function trn(A::AbstractMatrix)
    d = size(A, 1)

    return (d*tr(A) - 1)/(d - 1)
end


"""
    issquared(M::AbstractMatrix)

Checks if the matrix `M` is squared.
"""
function issquared(M::AbstractMatrix)
    N, M = size(M)

    return N == M
end


issemiposdef(ρ::AbstractMatrix, tol=eps()) = isposdef(ρ + tol*I(size(ρ, 1)))


"""
    nearest_posdef(A::AbstractMatrix; tol=eps())

Computes the nearest Hermitian positive semidefinite matrix
to `A` in the Frobenius norm, according to [1].
The value of `tol` will be the smallest eigenvalue of the returned matrix.

References
==========
[1] : "Computing a nearest symmetric positive semidefinite matrix" - N. J. Higham (1988)
      https://doi.org/10.1016/0024-3795(88)90223-6
"""
function nearest_posdef(A::AbstractMatrix; tol=eps())
    # Take Hermitian part of B
    B = 0.5(A + A')

    vals, vecs = eigen(Hermitian(B))

    # Patch negative eigenvalues
    vals = max.(vals, tol)

    # Re-compose patched matrix
    B = vecs * Diagonal(vals) * vecs'

    # Positive definite is a subset of Hermitian matrices
    return Hermitian(B)
end

"""
    nearest_density(A::AbstractMatrix; tol=eps())

Computes the nearest density matrix to `A` in the Frobenius norm
by first computing the nearest positive definite matrix with [`nearest_posdef`](@ref)
and then trace-normalizing the result.
"""
function nearest_density(A::AbstractMatrix; tol=eps())
    B = nearest_posdef(A; tol=tol)

    return B / tr(B)
end


"""
    ptrace(M::AbstractMatrix, subsystem_sizes, trace_over)

Returns the partial trace of the matrix `M` over the subsystem(s)
`trace_over`, where the size of all of the subsystems is specified
by `subsystem_sizes`.

Examples
=======

```julia-repl
julia> using LinearAlgebra: kron # Kronecker product

julia> state1 = random_density(3); #  3x3 density matrix

julia> state2 = random_density(4); #  4x4 density matrix

julia> full_state = kron(state1, state2);

julia> reduced1 = ptrace(full_state, (3, 4), 2); # Trace second subspace

julia> reduced2 = ptrace(full_state, (3, 4), 1); # Trace first subspace

julia> isapprox(reduced1, state1)
true

julia> isapprox(reduced2, state2)
true
```

```julia-repl
julia> A = reshape(1:16, 4, 4)[:, :]
4×4 Matrix{Int64}:
 1  5   9  13
 2  6  10  14
 3  7  11  15
 4  8  12  16

julia> ptrace(A, (2,2), 1) # Trace over the first subsystem
2×2 Matrix{Int64}:
 12  20
 14  22

julia> ptrace(A, (2,2), 2) # Trace over the second subsystem
2×2 Matrix{Int64}:
  7  23
 11  27

julia> ptrace(A, (2,2), (1,2)) # Trace over both subsystems
1×1 Matrix{Int64}:
 34
```
"""
function ptrace(M::AbstractMatrix, subsystem_sizes, trace_over)
    issquared(M) || throw("Matrix is not squared")
    prod(subsystem_sizes)==size(M, 1) || throw("Matrix size does not comply with the subsystem sizes")

    # Separate the subsystems of M by axes
    rev_subsystem_sizes = reverse(subsystem_sizes)
    T = reshape(M, rev_subsystem_sizes..., rev_subsystem_sizes...)

    # Iterate tracing over each subsystem
    Nsys = length(subsystem_sizes)
    for subsystem in trace_over
        dim = Nsys - subsystem + 1
        T = mapslices(tr, T, dims=(dim, Nsys + dim))
    end

    # Compute size of the traced matrix
    non_traced_systems = filter(!in(trace_over), 1:Nsys)
    Ntraced = prod( subsystem_sizes[ non_traced_systems ] )

    return reshape(T, Ntraced, Ntraced)
end
