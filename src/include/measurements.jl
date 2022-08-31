export expectation_values

function expectation_values(ρ::AbstractMatrix, base::AbstractMatrix)
    return sum(conj(base) .* (ρ * base), dims=1)[:]
end
