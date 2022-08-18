export simulate_experiment

"""
    simulate_experiment(value::Number, Nshots=Inf)

Returns a random sample from a
[`binomial distribution`](https://en.wikipedia.org/wiki/Binomial_distribution)
with success probability `value` and number of trials `Nshots`.
"""
function simulate_experiment(value::Number, Nshots=Inf)

    if isinf(Nshots)
        simulated = value
    else
        distribution = Binomial(Nshots, value)
        simulated = rand(distribution) / Nshots
    end

    return simulated
end

"""
    simulate_experiment(value::AbstractVector, Nshots=Inf)

Returns a random sample from a
[`multinomial distribution`](https://en.wikipedia.org/wiki/Multinomial_distribution)
with success probabilities `values` and number of trials `Nshots`.
"""
function simulate_experiment(values::AbstractVector, Nshots=Inf)

    if isinf(Nshots)
        simulated = values
    else
        distribution = Multinomial(Nshots, values)
        simulated = rand(distribution) / Nshots
    end

    return simulated
end
