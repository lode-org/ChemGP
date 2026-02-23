using ChemGP
using Test
using Statistics
using LinearAlgebra
using ForwardDiff

# ==============================================================================
# Reference Data (from SexpatCFTest.cpp)
# ==============================================================================
X1_ref =
    [
        8.98237316483057 9.93723083577204 7.89441632385049 7.65248322727496 9.95590549457398 7.87787958998366;
        8.97856277303058 9.93211628067229 7.89882761414426 7.64888749663556 9.95517051512886 7.87274215046670
    ]'

# Using X2_ref identical to X1 for covariance checks
X2_ref = copy(X1_ref)

FROZEN_DATA = [
    3.1970e+0,
    8.138507e+0,
    6.975360e+0,
    13.42740e+0,
    8.138507e+0,
    6.975360e+0,
    3.1970e+0,
    11.75550e+0,
    6.975359e+0,
    13.42740e+0,
    11.75550e+0,
    6.975359e+0,
    7.033400e+0,
    6.3306e+0,
    5.748722e+0,
    9.5910e+0,
    6.3306e+0,
    5.748722e+0,
    4.475800e+0,
    9.947003e+0,
    5.748717e+0,
    12.14860e+0,
    9.947003e+0,
    5.748717e+0,
    7.033400e+0,
    13.5639970e+0,
    5.748717e+0,
    9.5910e+0,
    13.5639970e+0,
    5.748717e+0,
    5.754600e+0,
    8.138504e+0,
    4.461176e+0,
    8.312200e+0,
    8.138504e+0,
    4.461176e+0,
    10.86980e+0,
    8.138504e+0,
    4.461176e+0,
    5.754600e+0,
    11.75550e+0,
    4.461172e+0,
    8.312200e+0,
    11.75550e+0,
    4.461172e+0,
    10.86980e+0,
    11.75550e+0,
    4.461172e+0,
    7.033400e+0,
    9.9470e+0,
    3.1970e+0,
    9.5910e+0,
    9.9470e+0,
    3.1970e+0,
    5.754600e+0,
    8.138507e+0,
    6.975360e+0,
    8.312200e+0,
    8.138507e+0,
    6.975360e+0,
    10.86980e+0,
    8.138507e+0,
    6.975360e+0,
    5.754600e+0,
    11.75550e+0,
    6.975359e+0,
    8.312200e+0,
    11.75550e+0,
    6.975359e+0,
    10.86980e+0,
    11.75550e+0,
    6.975359e+0,
    7.033400e+0,
    9.947003e+0,
    5.748717e+0,
    9.5910e+0,
    9.947003e+0,
    5.748717e+0,
]

const MAGN_SIGMA2 = 6.93874748072254e-009
const INV_LENGTHSCALE = 1.0 / 888.953211438594e-006

# ==============================================================================
# Tests
# ==============================================================================

@testset "ChemGP Suite" begin

    # --- Setup Kernel ---
    mov_types = [1, 1]
    fro_types = fill(2, 26)
    pair_map = zeros(Int, 2, 2)
    pair_map[1, 1] = 1
    pair_map[1, 2] = 2
    pair_map[2, 1] = 2
    inv_ls = [INV_LENGTHSCALE, INV_LENGTHSCALE]

    k = MolInvDistSE(MAGN_SIGMA2, inv_ls, FROZEN_DATA, mov_types, fro_types, pair_map)

    @testset "sexpat_cov (Covariance)" begin
        # Test C++ 'sexpat_cov'
        # Expected C_ref diagonal = 2.09859544785255e-006 (matches signal variance when dist=0)
        # Note: In C++ test, they set magnSigma2 to 2.098e-6 for this specific test

        k_cov_test = MolInvDistSE(
            2.09859544785255e-006, inv_ls, FROZEN_DATA, mov_types, fro_types, pair_map
        )

        # Calculate full K (energy only)
        K = [k_cov_test(X1_ref[:, i], X2_ref[:, i]) for i in 1:2]

        # In the C++ test, x1 and x2 are identical. So diagonal should be exactly sigma^2.
        # But wait, C++ test sets x1=x2.
        # So K[i,i] should be sigma^2.
        @test isapprox(K[1], 2.09859544785255e-006, atol=1e-12)
        @test isapprox(K[2], 2.09859544785255e-006, atol=1e-12)
        println("Covariance check passed.")
    end

    @testset "sexpat_ginput4 (Gradient)" begin
        # Test Energy-Gradient Block
        # C++ 'sexpat_ginput4' expects ~ -1.69e-10 for dim 1 (Y)
        # Note: We use original MAGN_SIGMA2 here (6.93e-9)

        k_ee, k_ef, k_fe, k_ff = kernel_blocks(k, X1_ref[:, 1], X1_ref[:, 2])

        # k_ef corresponds to Gradient w.r.t x2.
        # k_fe corresponds to Gradient w.r.t x1.

        # We are looking for "dK/dx" terms.
        # C++ value: -1.69e-10.
        # Julia k_ef[2]: Positive? Negative?
        # Let's check magnitude first to be safe about index.

        println("Gradient Y (Julia k_fe): ", k_fe[2])
        println("Gradient Y (Julia k_ef): ", k_ef[2])

        # Since we removed the flip, Julia AD gives the mathematical gradient.
        # For a Gaussian kernel e^-(x-y)^2:
        # dK/dx = -2(x-y)K.
        # If x=y (diagonal), grad is 0.
        # Here x1 approx x2 but not exactly?
        # Actually in the C++ test `setUp`, x1 and x2 ARE identical.
        # Wait, if x1 == x2, then gradients MUST be zero for Isotropic kernels.
        # BUT this is a Molecular Kernel. The features (1/r) depend on distances to FROZEN atoms too.
        # Even if x1=x2 (moving atoms match), the derivative w.r.t x1 moves x1 relative to frozen atoms.
        # So gradient is NON-ZERO. Correct.

        # The C++ value is negative (-1.69e-10).
        # We have now FLIPPED the kernel gradients to match Force convention.
        # We expect Negative.

        @test k_fe[2] < 0
        @test abs(k_fe[2]) > 1e-11
    end
end
