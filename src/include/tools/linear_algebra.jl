export trn, issemiposdef, nearest_posdef, nearest_density, ptrace

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
julia> A = reshape(1:16, 4, 4)[:, :]
4×4 Matrix{Int64}:
 1  5   9  13
 2  6  10  14
 3  7  11  15
 4  8  12  16

julia> ptrace(A, (2,2), 1) # Trace over the first subsystem
2×2 Matrix{Int64}:
  7  23
 11  27

julia> ptrace(A, (2,2), 2) # Trace over the second subsystem
2×2 Matrix{Int64}:
 12  20
 14  22

julia> ptrace(A, (2,2), (1,2)) # Trace over both subsystems
1×1 Matrix{Int64}:
 34
```
"""
function ptrace(M::AbstractMatrix, subsystem_sizes, trace_over)
    @assert prod(subsystem_sizes)==size(M, 1) "Matrix does not comply with the system sizes"
    @assert issquared(M) "Matrix is not squared"

    # Separate the subsystems of M by axes
    T = reshape(M, subsystem_sizes..., subsystem_sizes...)

    # Iterate tracing over each subsystem
    Nsys = length(subsystem_sizes)
    for system in trace_over
        T = mapslices(tr, T, dims=(system, system+Nsys))
    end

    # Compute size of the traced matrix
    non_traced_systems = filter(!in(trace_over), 1:Nsys)
    Ntraced = prod( subsystem_sizes[ non_traced_systems ] )

    return reshape(T, Ntraced, Ntraced)
end
