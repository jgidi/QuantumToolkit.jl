export simulate_experiment

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
