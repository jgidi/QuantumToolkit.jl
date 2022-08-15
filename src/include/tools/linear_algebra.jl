export issemiposdef, ptrace

"""
    issquared(M::Matrix)

Checks if the matrix `M` is squared.
"""
function issquared(M::Matrix)
    N, M = size(M)

    return N == M
end


issemiposdef(ρ::Matrix, tol=eps()) = isposdef(ρ + tol*I(size(ρ, 1)))

"""
    ptrace(M::Matrix, subsystem_sizes, trace_over)

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