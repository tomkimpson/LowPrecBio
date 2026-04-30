#!/usr/bin/env julia

"""
One-command reproducibility entrypoint for reviewers.

Usage:
  julia --project=. scripts/reproduce.jl
  julia --project=. scripts/reproduce.jl --skip-benchmarks
  julia --project=. scripts/reproduce.jl --skip-schlogl
  julia --project=. scripts/reproduce.jl --run-tests=false
"""

using Printf
using Dates

include("cli_args.jl")

function run_checked(cmd::Cmd, name::String; dry_run::Bool=false)
    println("-"^72)
    println(name)
    println("-"^72)
    println("CMD: ", cmd)
    t0 = time()
    ok = true
    if !dry_run
        proc = run(ignorestatus(cmd))
        ok = success(proc)
    end
    dt = time() - t0
    @printf("%s: %s (%.1fs)\n\n", name, ok ? "OK" : "FAILED", dt)
    return ok
end

function main()
    opts = parse_cli_args(ARGS)
    dry_run = arg_bool(opts, "dry-run", false)
    run_tests = arg_bool(opts, "run-tests", true)
    skip_benchmarks = arg_bool(opts, "skip-benchmarks", false)
    skip_schlogl = arg_bool(opts, "skip-schlogl", true)

    scripts_dir = @__DIR__
    project_dir = dirname(scripts_dir)
    run_all_script = joinpath(scripts_dir, "run_all.jl")

    println("="^72)
    println("LowPrecBio Reproducibility Runner")
    println("="^72)
    println("Started: ", Dates.now())
    println("Run tests: ", run_tests)
    println("Skip benchmarks: ", skip_benchmarks)
    println("Skip Schlogl: ", skip_schlogl)
    println("Dry run: ", dry_run)
    println()

    ok = true

    if run_tests
        test_cmd = `$(Base.julia_cmd()) --project=$project_dir -e "using Pkg; Pkg.test()"`
        ok &= run_checked(test_cmd, "Step 1/2: Test Suite"; dry_run=dry_run)
    end

    args = String[]
    if skip_benchmarks
        push!(args, "--skip-benchmarks")
    end
    if skip_schlogl
        push!(args, "--skip-schlogl")
    end
    run_all_cmd = `$(Base.julia_cmd()) --project=$project_dir $run_all_script $args`
    ok &= run_checked(run_all_cmd, run_tests ? "Step 2/2: Full Experiment Run" : "Step 1/1: Full Experiment Run"; dry_run=dry_run)

    if ok
        println("Reproduction run completed successfully.")
    else
        println("Reproduction run failed.")
        exit(1)
    end
end

main()
