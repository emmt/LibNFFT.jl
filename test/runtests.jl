module TestLibNFFT

using LibNFFT, Test
import LibNFFT: get_x, get_x!, set_x!, get_M, get_d, isready

@testset "Basic Transforms" begin
    for dims in ((10,), (8,12), (4,6,8))
        d = length(dims)
        M = 11
        x = rand(d, M) .- 0.5
        T = Cdouble
        F = NFFT(M, T, dims)
        @test !isready(F)
        set_x!(F, x)
        @test isready(F)
        C = NFCT(M, T, dims)
        @test !isready(C)
        set_x!(C, x)
        @test isready(C)
        S = NFST(M, T, dims)
        @test !isready(S)
        set_x!(S, x)
        @test isready(S)
        GC.gc()
    end
end

end # module
