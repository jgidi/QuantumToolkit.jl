export squeeze, logrange, bound, simulate_experiment

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

function simulate_experiment(value::Number, Nshots=Inf)

    if isinf(Nshots)
        simulated = value
    else
        distribution = Binomial(Nshots, value)
        simulated = rand(distribution) / Nshots
    end

    return simulated
end

function simulate_experiment(values::Vector, Nshots=Inf)

    if isinf(Nshots)
        simulated = values
    else
        distribution = Multinomial(Nshots, values)
        simulated = rand(distribution) / Nshots
    end

    return simulated
end
