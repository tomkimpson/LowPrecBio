#!/usr/bin/env julia
#
# Master script: run all experiment scripts end-to-end via subprocesses.
#
# Usage:
#   julia --project=. scripts/run_all.jl                 # run everything
#   julia --project=. scripts/run_all.jl --skip-benchmarks  # skip benchmarks
#   julia --project=. scripts/run_all.jl --skip-schlogl     # skip Schlogl model
#

const EXPERIMENT_SCRIPTS = [
    "run_birth_death.jl",
    "run_schlogl.jl",
    "run_telegraph.jl",
    "run_dimer.jl",
    "run_repressilator.jl",
]

const BENCHMARK_SCRIPT = "run_benchmarks.jl"

function main()
    skip_benchmarks = "--skip-benchmarks" in ARGS
    skip_schlogl = "--skip-schlogl" in ARGS

    scripts_dir = @__DIR__
    project_dir = dirname(scripts_dir)

    scripts = copy(EXPERIMENT_SCRIPTS)
    if skip_schlogl
        scripts = filter(!=("run_schlogl.jl"), scripts)
    end
    if !skip_benchmarks
        push!(scripts, BENCHMARK_SCRIPT)
    end

    # Results: (name, passed, elapsed_seconds)
    results = Tuple{String,Bool,Float64}[]

    println("=" ^ 60)
    println("LowPrecBio: running all experiments")
    println("  Scripts: ", join(scripts, ", "))
    println("  Skip benchmarks: ", skip_benchmarks)
    println("  Skip Schlogl: ", skip_schlogl)
    println("=" ^ 60)
    println()

    total_t0 = time()

    for script in scripts
        script_path = joinpath(scripts_dir, script)

        println("-" ^ 60)
        println("Running: $script")
        println("-" ^ 60)

        t0 = time()
        cmd = `$(Base.julia_cmd()) --project=$project_dir $script_path`
        proc = run(ignorestatus(cmd))
        elapsed = time() - t0
        passed = success(proc)

        push!(results, (script, passed, elapsed))

        status_str = passed ? "OK" : "FAILED"
        println()
        println("  $script  =>  $status_str  ($(round(elapsed; digits=1))s)")
        println()
    end

    total_elapsed = time() - total_t0

    # Summary table
    println()
    println("=" ^ 60)
    println("Summary")
    println("=" ^ 60)
    println()
    println(rpad("Script", 30), rpad("Status", 10), "Time (s)")
    println("-" ^ 50)
    for (name, passed, elapsed) in results
        status_str = passed ? "OK" : "FAILED"
        println(rpad(name, 30), rpad(status_str, 10), round(elapsed; digits=1))
    end
    println("-" ^ 50)
    println(rpad("Total", 30), rpad("", 10), round(total_elapsed; digits=1))
    println()

    n_failed = count(r -> !r[2], results)
    if n_failed > 0
        println("$n_failed script(s) FAILED.")
        exit(1)
    else
        println("All scripts passed.")
    end
end

main()
