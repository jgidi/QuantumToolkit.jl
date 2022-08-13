export logrange, bound, simulate_experiment

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
