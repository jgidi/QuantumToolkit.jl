export squeeze, logrange, bound
export bloch_vector, bloch_angles, qubit_from_angles

"""
    squeeze(A::AbstractArray)

Remove singleton (length 1) dimensions of `A` by reshaping.

Notes
=====

* This function is not type-stable and may harm performance.
  Don't use it within performance-critical parts of code.
* The resulting array points to the same memory as `A`.

Examples
========

```julia-repl
julia> A = rand(1, 2, 1, 2, 1)
1×2×1×2×1 Array{Float64, 5}:
[:, :, 1, 1, 1] =
 0.608853  0.240589

[:, :, 1, 2, 1] =
 0.906431  0.109752

julia> B = squeeze(A) # Remove singleton dimensions
2×2 Matrix{Float64}:
 0.608853  0.906431
 0.240589  0.109752

julia> B[1, 1] = 0.0 # Changing B also changes A
0.0

julia> A
1×2×1×2×1 Array{Float64, 5}:
[:, :, 1, 1, 1] =
 0.0  0.240589

[:, :, 1, 2, 1] =
 0.906431  0.109752
```

"""
squeeze(A::AbstractArray) = reshape(A, filter(!=(1), size(A)))

"""
    logrange(start, stop, N)
Returns an array of `N` logarithmically spaced points from `start` to `stop`.
"""
logrange(start, stop, N) = exp.(range(log(start), log(stop), N))

"""
    bound(value, limits = (zero(value), one(value)))

Returns the number `value` bounded within a pair of limits, such that
`limits[1] <= value <= limits[2]`.
"""
function bound(value, limits = (zero(value), one(value)))
  mini, maxi = minmax(limits...)

  return max(min(value, maxi), mini)
end


"""
    bloch_angles(ket::AbstractVector)

Returns the angles `theta` and `phi` defining the position of the pure state
`ket` on the Bloch sphere.
"""
function bloch_angles(ket::AbstractVector)
    isstate(ket)   || throw("Input is not a valid physical state")
    length(ket)==2 || throw("Bloch sphere is only defined for d=2")

    a, b = ket

    theta = 2acos(abs(a))
    phi = angle(b/a)

    return theta, phi
end


"""
    bloch_vector(ket::AbstractVector)

Returns the Bloch vector defining the position of the pure state
`ket` on the Bloch sphere.
"""
bloch_vector(ket::AbstractVector) = bloch_vector(ket*ket')

"""
    bloch_vector(A::AbstractMatrix)

Returns the Bloch vector defining the position of the density state
`A` on the Bloch ball.
"""
function bloch_vector(A::AbstractMatrix)
    size(A, 1)==2 || throw("Bloch sphere is only defined for d=2")
    # isstate(A)    || throw("Input is not a valid physical state.")
    mx, my = reim(2A[1, 2])
    mz = real(A[1, 1] - A[2,2])

    return (mx, my, mz)
end


"""
    qubit_from_angles(theta::Real, phi::Real, global_phase::Real=0.0)

Returns the 2-level ket state defined by the two angles, `theta` and `phi`
on the Bloch sphere.
"""
function qubit_from_angles(theta::Real, phi::Real, global_phase::Real=0.0)
    a = cos(0.5theta)
    b = cis(phi)*sqrt(1 - a^2)

    qubit = [a, b]
    if !iszero(global_phase)
        qubit .*= cis(global_phase)
    end

    return qubit
end
