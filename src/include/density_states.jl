export random_density
export projector, purity
export issemiposdef, isstate, ispure


"""
    random_density(d::Integer)

Returns a random density matrix with side size `d` from the Hilbert-Schmidt metric.
"""
function random_density(d::Integer)
    M = randn(Complex{Float64}, d, d)
    M = M*M'

    return M/tr(M)
end
# counts(purity) ~ sqrt(purity) (w/randn)
# counts(purity) ~ purity^3 (w/rand)


# TODO Compare distriution with standard implementation
# counts(purity) ~ 1 / sqrt(purity)
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

"""
    isstate(ρ::Matrix)

Checks if `ρ` is a density matrix.
That is, if `ρ` is semi-positive definite and normalized to trace 1.
"""
isstate(ρ::Matrix) = isapprox(tr(ρ), 1) && issemiposdef(ρ)

ispure(ρ::Matrix) = isapprox(purity(ρ), 1)
