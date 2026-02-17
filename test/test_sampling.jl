@testset "Farthest Point Sampling" begin
    # Create a set of candidate points (3-atom configs) on a line
    # Candidates: atoms at (0,0,0), (r,0,0), (2r,0,0) for r = 1.0, 1.2, ..., 3.0
    r_vals = collect(1.0:0.2:3.0)
    D = 9  # 3 atoms * 3 coords
    candidates = zeros(D, length(r_vals))
    for (i, r) in enumerate(r_vals)
        candidates[4, i] = r      # atom 2 x-position
        candidates[7, i] = 2 * r  # atom 3 x-position
    end

    # Start with one selected point (r=1.0)
    X_selected = candidates[:, 1:1]

    # Select 3 points via FPS
    indices = farthest_point_sampling(candidates, X_selected, 3)

    @test length(indices) == 3
    @test length(unique(indices)) == 3  # All unique

    # First selected should be the farthest from r=1.0, which is r=3.0 (last)
    @test indices[1] == length(r_vals)

    # FPS with Euclidean distance should also work
    indices_euc = farthest_point_sampling(
        candidates, X_selected, 3;
        distance_fn = (a, b) -> norm(a - b),
    )
    @test length(indices_euc) == 3
end
