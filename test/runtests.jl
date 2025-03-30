using ContinuumWorld
using POMDPs
using POMDPModelTools
using POMDPModels
using POMDPTools
using Random
using Test
using Plots
using Plotly

@testset "ContinuumWorld.jl Tests" begin
    w = CWorld()

    @testset "CWorld structure" begin
        @test w.xlim == (0.0, 10.0)
        @test length(w.reward_regions) == length(w.rewards)
        @test all(r isa CircularRegion for r in w.reward_regions)
    end

    @testset "Vec2Distribution sampling" begin
        d = Vec2Distribution((0.0, 10.0), (0.0, 10.0))
        sample = rand(MersenneTwister(42), d)
        @test sample isa SVector{2, Float64}
        @test 0.0 <= sample[1] <= 10.0
        @test 0.0 <= sample[2] <= 10.0
    end

    @testset "Reward and terminal" begin
        for (r, val) in zip(w.reward_regions, w.rewards)
            @test reward(w, r.center, Vec2(0.0, 0.0), r.center) ≥ val
            @test isterminal(w, r.center) == true
        end
        @test isterminal(w, Vec2(0.0, 0.0)) == false
    end

    @testset "Transition" begin
        a = Vec2(1.0, 0.0)
        s = Vec2(3.0, 2.0)
        rng = MersenneTwister(19)
        t_dist = transition(w, s, a)
        sp = rand(rng, t_dist)
        @test sp isa Vec2
    end

    @testset "Solver and Policy" begin
        sol = CWorldSolver(rng=MersenneTwister(7))
        pol = solve(sol, w)

        s = Vec2(5.0, 5.0)
        a = action(pol, s)
        @test a in w.actions

        v = value(pol, s)
        @test v isa Float64
    end

    @testset "Simulation" begin
        sim = HistoryRecorder(rng=MersenneTwister(5), max_steps=30)
        hist = simulate(sim, w, solve(CWorldSolver(), w))
        @test length(state_hist(hist)) ≤ 30
    end

    @testset "Visualization" begin
        vis = CWorldVis(w, f=s -> value(solve(CWorldSolver(rng=MersenneTwister(1)), w), s), g=CWorldSolver().grid)
        plt = Plots.plot(vis)
        @test plt isa Plots.Plot
    end
end
