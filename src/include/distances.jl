export fidelity
export cross_entropy, cross_entropy_experimental

"""
    fidelity(s1::Vector, s2::Vector)

Returns the fidelity for two states `s1` and `s2`.
"""
function fidelity(s1::Vector, s2::Vector)
    prod2  = abs2( s1's2 )
    norms2 =  sum(abs2, s1) * sum(abs2, s2)

    return prod2/norms2
end

"""
    fidelity(ρ::Matrix, σ::Matrix)

Returns the fidelity for two matrices `ρ` and `σ`.
"""
function fidelity(ρ::Matrix, σ::Matrix)
    sqrtρ = sqrt(ρ)

    return abs2( tr(sqrt(sqrtρ * σ * sqrtρ)) )
end


"""
    cross_entropy(ρ::Matrix, σ::Matrix)

Returns the cross entropy `tr(ρ log(σ))` from two matrices `ρ` and `σ`.
"""
cross_entropy(ρ::Matrix, σ::Matrix) = -tr(ρ * log(σ))

function cross_entropy_experimental(ρ::Matrix, σ::Matrix;
                                    Nshots=Inf, basis_perturbation=0.0)
    vals, vecs = eigen(σ)

    # perturb sigma eigenvector basis
    vecs = perturb_basis(vecs, basis_perturbation)

    # <λ|ρ|λ> for |λ> in eigvecs(σ)
    expectvals = expectation_values(ρ, vecs)

    # Add statistical noise
    expectvals = simulate_experiment(expectvals, Nshots)

    # Collect entropy from projection values
    entropy = -sum(@. log(abs(vals)) * expectvals)

    return entropy
end
