export expectation_values

function expectation_values(ρ::AbstractMatrix, base::AbstractMatrix)
    return sum(conj(base) .* (ρ * base), dims=1)[:]
end

# Hermitian matrices should yield real expectation values
function expectation_values(ρ::Hermitian, base::AbstractMatrix)
    return sum(real, conj(base) .* (ρ * base), dims=1)[:]
end
