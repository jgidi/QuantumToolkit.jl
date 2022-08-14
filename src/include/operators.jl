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


# TODO Compare distriution with standard implementation
#
# According to Alsing et. al. (2022)
# "The distribution of density matrices at fixed purity for arbitrary dimensions"
function _random_density(d::Integer)
    # Two Haar-distributed unitary matrices
    U = random_unitary(d)
    V = random_unitary(d)

    # Sample the eigenvalues from a column of V and build the
    # diagonalized form of the density matrix
    D = Diagonal( abs2.(V[:, 1]) )

    # Use U to form ρ as a similarity transformation of D
    return U * D * U'
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
purity(ρ::Matrix) = real(tr(ρ*ρ))

issemiposdef(ρ::Matrix, tol=eps()) = isposdef(ρ + tol*I(size(ρ, 1)))

isdensity(ρ::Matrix) = issemiposdef(ρ) && isapprox(tr(ρ), 1)

ispure(ρ::Matrix) = isapprox(purity(ρ), 1)
