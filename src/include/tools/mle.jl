using GellMannMatrices: gellmann

___Optim_loaded = false
isloaded_Optim() = ___Optim_loaded

"""
Load package `Optim` on runtime to diminish package startup time.
"""
function _load_Optim()
    global ___Optim_loaded

    if !___Optim_loaded
        @eval using Optim: Optim
        ___Optim_loaded = true
    end

    return nothing
end

# TODO
# Rewrite using functions from this package.
# You don't really need this function.
# You only need get_probs(matrix, basis_eigvals) (with experimental opts)
# and MLE(ρ, probs, basis || basis_eigvecs)
function standard_tomography(ρ::AbstractMatrix;
                             Nshots = Inf, err_amp = 0.0, method = :linear,
                             kwargs...,
                             )
    d = size(ρ, 1)
    GM = gellmann(d)
    Nb = length(GM)             # Length of the basis

    gmvecs = eigvecs.(GM)
    gmvals = eigvals.(GM)

    # B = [ tr(gmi * gmj) for gmi in GM, gmj in GM ] # 2*I(length(GM))

    p = Array{Float64}(undef, Nb)
    prob = Array{Float64}(undef, d, Nb)
    eigenvals = Array{Float64}(undef, d, Nb)
    for i in eachindex(GM)
        vecs = gmvecs[i]
        vals = gmvals[i]

        # Add noise to the bases (Preparation error)
        perturbed_basis = perturb_basis(vecs, err_amp)

        # Expectation values
        expectvals = expectation_values(ρ, perturbed_basis)

        # Simulate experiment (Statistical sampling error)
        expectvals = simulate_experiment(expectvals, Nshots)

        prob[:, i] = expectvals
        eigenvals[:, i] = vals

    end

    p = sum(eigenvals .* prob, dims=1) |> x->reshape(x, :)

    # Coefficients by pseudo inversion
    # s = pinv(B) * p # Generic basis
    s = 0.5p # Special case for Gell-Mann matrices

    # Reconstruct state in the Gell-Mann basis
    ρnew = I(d)/d + sum(ss*gg for (ss, gg) in zip(s, GM))

    # Make solution positive definite
    ρnew = nearest_posdef(ρnew)

    if method == :MLE
        Anew = cholesky(ρnew).U
        Anew = MLE(Anew, prob, gmvecs)

        ρnew = Anew'Anew
        ρnew /= tr(ρnew)
    end

    return ρnew
end


# TODO
# Add to QuantumToolkit the MLE assoc. functions.
# * Separate to_real_vec(::AbstractMatrix{ComplexF64}),
#   and to_complex_mat(::AbstractVector{Float64}, N, M)
# * Also add makeposdef(). Regularize, maybe?? Check algos.
# * Replace sum(f) by length(eigenvecs) (d^2 - 1)
#
# E. g.
# function to_real_vec(cx_mat::AbstractMatrix{ComplexF64})
#     cx_vec = reshape(cx_mat, :)
#     re_vec = reinterpret(Float64, cx_vec)

#     return Vector(re_vec)
# end

function make_fg!(A, f, eigenvecs)

    N, M = size(A)

    function to_real_vec(cx_mat)
        cx_vec = reshape(cx_mat, :)
        re_vec = reinterpret(Float64, cx_vec)

        return Vector(re_vec)
    end

    function to_complex_mat(re_vec)
        cx_vec = reinterpret(ComplexF64, re_vec)
        cx_mat = reshape(cx_vec, N, M)

        return Matrix(cx_mat)
    end

    function fg!(F, G, A_real_vec)
        # Should F, G or both computed?
        compute_F = !isnothing(F)
        compute_G = !isnothing(G)

        # Transform real vector input to complex matrix A
        A = to_complex_mat(A_real_vec)

        # Probability distribution assoc. to A'A
        # Common computations for both f and g!
        p = similar(f)
        trA2 = sum(abs2, A)
        if compute_G
            grad = sum(f) * A
        end
        for (i, gm) in enumerate(eigenvecs)
            for (j, vec) in enumerate(eachcol(gm))
                Avec = A*vec
                p[j, i] = sum(abs2, Avec) / trA2

                if compute_G
                    # Conjugate gradient with respect to complex matrix A
                    grad -= (f[j, i]/p[j, i]) * Avec*vec'
                end
            end
        end

        if compute_G
            # Return gradient as real vec
            G .= to_real_vec(grad / trA2^2)
        end
        if compute_F
            # Value does not need conversion
            return -sum(@. f * log(p))
        end

        return nothing
    end

    return fg!
end

# TODO: Cholesky decomp matrix A should be used on the inside,
# but the input as well as the output of the function should
# be the whole density matrix.
# If the input is not hermitian, use nearest_posdef()
# before applying chol.
#
# Also, it should accept either basis (Vector{Matrix})
# or eigenvecs (with the same shape as probs).
function MLE(A::AbstractMatrix, probs, eigenvecs)

    # Load optimization library if not already loaded
    _load_Optim()

    # Cast guess as real vector
    guess = Vector(reinterpret(Float64, A)[:])

    # Function to compute objective and gradient
    fg! = make_fg!(A, probs, eigenvecs)

    # Disable stopping on gradient norm
    options = Optim.Options(g_tol = -Inf)

    # Perform optimization
    opt = Optim.optimize(Optim.only_fg!(fg!), guess,
                         Optim.LBFGS(),
                         options,
                         )

    # Extract value and return to matrix form
    Anew = opt.minimizer
    Anew = reshape(reinterpret(ComplexF64, Anew), size(A)...)

    return Matrix(Anew)
end
