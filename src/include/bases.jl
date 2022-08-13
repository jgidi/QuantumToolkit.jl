export orthonormalize, perturb_basis


"""
    orthonormalize(M::Matrix)

Perform orthonormalization via QR decomposition to a vector base,
where each column of the input matrix `M` is a vector of the base.
"""
orthonormalize(M::Matrix) = Matrix(qr(M).Q)

function perturb_basis(basis, amplitude)
    N = size(basis, 1)
    perturbed = similar(basis)
    for (i, vec) in enumerate(eachcol(basis))
        perturbed[:, i] = vec + amplitude * random_state(N)
    end

    return orthonormalize(perturbed)
end
