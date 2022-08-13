export random_unitary, random_density
export projector, purity
export issemiposdef, isdensity, ispure

function random_unitary(d::Integer)
    M = randn(Complex{Float64}, d, d)
    Q, R = qr( M )

    return Q * sign.(Diagonal(R))
end

"""
    random_density(d::Integer)

Returns a random density matrix with side size `d` from the Hilbert-Schmidt metric.
"""
function random_density(d::Integer)
    M = randn(Complex{Float64}, d, d)
    M = M*M'

    return M/tr(M)
end

"""
    projector(ket::Vector)

Returns the ket-bra formed from the input state `ket`.
"""
projector(ket::Vector) = ket * ket'

"""
    purity(ρ::Matrix) = tr(ρ*ρ)
Returns de purity, `tr(ρ^2)`, of the density matrix `ρ`.
"""
purity(ρ::Matrix) = tr(ρ*ρ)

issemiposdef(ρ::Matrix, tol=eps()) = isposdef(ρ + tol*I(size(ρ, 1)))

isdensity(ρ::Matrix) = issemiposdef(ρ) && isapprox(tr(ρ), 1)

ispure(ρ::Matrix) = isapprox(purity(ρ), 1)
