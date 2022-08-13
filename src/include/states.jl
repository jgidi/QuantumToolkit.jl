export random_state, isstate

"""
    random_state(d::Integer)

Returns a Haar distributed random pure ket state,
\$|\\Psi\\rangle\$, with `d` elements.
"""
function random_state(d::Integer)
    ψ = randn(Complex{Float64}, d)

    return ψ / norm(ψ)
end

"""
    isstate(psi::Vector)

Checks if `psi` is normalized to probability `1`.
"""
isstate(psi::Vector) = isapprox(norm(psi), 1)
