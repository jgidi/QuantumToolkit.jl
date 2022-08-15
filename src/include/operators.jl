export random_unitary

function random_unitary(d::Integer)
    M = randn(Complex{Float64}, d, d)
    Q, R = qr( M )

    return Q * sign.(Diagonal(R))
end
