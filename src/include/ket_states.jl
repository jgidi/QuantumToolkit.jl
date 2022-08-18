export random_ketstate, isstate

"""
    random_ketstate(d::Integer)

Returns a Haar distributed random pure ket state,
\$|\\Psi\\rangle\$, with `d` elements.
"""
function random_ketstate(d::Integer)
    ψ = randn(Complex{Float64}, d)

    return ψ / norm(ψ)
end

"""
    isstate(ket::AbstractVector)

Checks if `ket` is normalized to probability `1`.
"""
isstate(ket::AbstractVector) = isapprox(norm(ket), 1)
