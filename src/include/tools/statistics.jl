export simulate_experiment

"""
    simulate_experiment(value::Number, Nshots=Inf)

Returns a random sample from a
[`binomial distribution`](https://en.wikipedia.org/wiki/Binomial_distribution)
with success probability `value` and number of trials `Nshots`.
"""
function simulate_experiment(value::Number, Nshots=Inf)

    isinf(Nshots) && return value

    return simulate_experiment(value, Int(Nshots))
end

function simulate_experiment(value::Number, Nshots::Integer)
    distribution = Binomial(Nshots, value)
    simulated = rand(distribution) / Nshots

    return simulated
end

"""
    simulate_experiment(value::AbstractVector, Nshots=Inf)

Returns a random sample from a
[`multinomial distribution`](https://en.wikipedia.org/wiki/Multinomial_distribution)
with success probabilities `values` and number of trials `Nshots`.
"""
function simulate_experiment(values::AbstractVector, Nshots=Inf)

    isinf(Nshots) && return values

    return simulate_experiment(values, Int(Nshots))
end

function simulate_experiment(values::AbstractVector, Nshots::Integer)
    distribution = Multinomial(Nshots, values)
    simulated = rand(distribution) / Nshots

    return simulated
end
