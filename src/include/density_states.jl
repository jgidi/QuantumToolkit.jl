export random_density
export projector, purity
export isstate, ispure


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


"""
    random_density(d::Integer, purity)

Returns a random density matrix with side size `d` and fixed purity, `purity`.

You can also pass `purity` as a tuple `(mini, maxi)` for the generation
of a mixed state with purity uniformly sampled from the interval `[mini, maxi]`.

Notes
=====
* The purity must fulfill `1/d <= purity <= 1`.
* The purity for a matrix `ρ` is `LinearAlgebra.tr(ρ^2)`.
"""
function random_density(d::Integer, purity)

    all(@. 1 <= d*purity <= d) ||
        throw("The purity must fulfill `1/d <= purity <= 1`.")

    if length(purity)>1
        mini, maxi = minmax(purity...)
        p = (maxi-mini)*rand() + mini
    else
        p = purity
    end

    ket = random_ketstate(d)

    s = sqrt((d*p - 1)/(d - 1))

    return s * ket*ket' + (1-s) * I(d) / d
end

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
    projector(ket::AbstractVector)

Returns the ket-bra formed from the input state `ket`.
"""
projector(ket::AbstractVector) = ket * ket'

"""
    purity(ρ::AbstractMatrix) = tr(ρ*ρ)

Returns de purity, `tr(ρ^2)`, of the density matrix `ρ`.
"""
purity(ρ::AbstractMatrix) = real(tr(ρ*ρ))

"""
    isstate(ρ::AbstractMatrix)

Checks if `ρ` is a density matrix.
That is, if `ρ` is semi-positive definite and normalized to trace 1.
"""
isstate(ρ::AbstractMatrix) = isapprox(tr(ρ), 1) && issemiposdef(ρ)

"""
    ispure(ρ::AbstractMatrix)

Checks if `ρ` is pure. That is, if `tr(ρ^2)` is approximately 1.
"""
ispure(ρ::AbstractMatrix) = isapprox(purity(ρ), 1)
