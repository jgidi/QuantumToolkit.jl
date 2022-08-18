export expectation_values

function expectation_values(ρ::AbstractMatrix, base::AbstractMatrix)
    return [ real(vec' * ρ * vec) for vec in eachcol(base) ]
end
