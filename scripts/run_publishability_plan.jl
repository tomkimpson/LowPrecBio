#!/usr/bin/env julia

"""
Run the publishability evidence plan (R1-R5) defined in submission_checklist.md.

Usage:
  julia --project=. scripts/run_publishability_plan.jl
  julia --project=. scripts/run_publishability_plan.jl --quick
  julia --project=. scripts/run_publishability_plan.jl --only=R1,R2
  julia --project=. scripts/run_publishability_plan.jl --dry-run
"""

using Printf
using Dates

include("cli_args.jl")

struct PlanTask
    id::String
    name::String
    cmd::Cmd
end

function parse_float_list(s::String)
    vals = Float64[]
    for part in split(s, ',')
        x = tryparse(Float64, strip(part))
        x === nothing && error("Invalid float list item: $part")
        push!(vals, x)
    end
    isempty(vals) && error("Expected at least one float value")
    return vals
end

function parse_int_list(s::String)
    vals = Int[]
    for part in split(s, ',')
        x = tryparse(Int, strip(part))
        x === nothing && error("Invalid integer list item: $part")
        push!(vals, x)
    end
    isempty(vals) && error("Expected at least one integer value")
    return vals
end

function normalize_only_list(s::String)
    return Set(uppercase(strip(x)) for x in split(s, ',') if !isempty(strip(x)))
end

function fmt_for_tag(x::Real)
    s = @sprintf("%.5g", Float64(x))
    s = replace(s, "." => "p")
    s = replace(s, "-" => "m")
    s = replace(s, "+" => "")
    return sanitize_tag(s)
end

function push_task!(tasks::Vector{PlanTask}, id::String, name::String, script_path::String, args::Vector{String}, project_dir::String)
    cmd = `$(Base.julia_cmd()) --project=$project_dir $script_path $args`
    push!(tasks, PlanTask(id, name, cmd))
end

function run_tasks(tasks::Vector{PlanTask}; dry_run::Bool=false)
    println()
    println("="^88)
    println("Publishability Plan Runner")
    println("="^88)
    println("Generated at: ", Dates.now())
    println("Tasks: ", length(tasks))
    println("Dry run: ", dry_run)
    println()

    results = Tuple{String, String, Bool, Float64}[]
    t0_all = time()

    for (i, task) in enumerate(tasks)
        println("-"^88)
        @printf("[%d/%d] %s (%s)\n", i, length(tasks), task.name, task.id)
        println("-"^88)
        println("CMD: ", task.cmd)

        t0 = time()
        ok = true
        if !dry_run
            proc = run(ignorestatus(task.cmd))
            ok = success(proc)
        end
        dt = time() - t0
        push!(results, (task.id, task.name, ok, dt))
        @printf("Status: %s (%.1fs)\n\n", ok ? "OK" : "FAILED", dt)
    end

    total_dt = time() - t0_all

    println("="^88)
    println("Plan Summary")
    println("="^88)
    @printf("%-5s  %-42s  %-8s  %8s\n", "ID", "Name", "Status", "Time (s)")
    println("-"^88)
    for (id, name, ok, dt) in results
        @printf("%-5s  %-42s  %-8s  %8.1f\n", id, name, ok ? "OK" : "FAILED", dt)
    end
    println("-"^88)
    @printf("%-5s  %-42s  %-8s  %8.1f\n", "", "Total", "", total_dt)

    failed = count(!r[3] for r in results)
    if failed > 0
        println()
        println("$failed task(s) failed.")
        exit(1)
    end
end

function main()
    opts = parse_cli_args(ARGS)

    quick = arg_bool(opts, "quick", false)
    dry_run = arg_bool(opts, "dry-run", false)
    only = normalize_only_list(arg_string(opts, "only", "R1,R2,R3,R4,R5"))

    output_root = arg_string(opts, "output-root", "results/publishability")
    seed_count = arg_int(opts, "seed-count", 5)
    seed_ssa_base = arg_int(opts, "seed-ssa-base", 7000)
    seed_sr_base = arg_int(opts, "seed-sr-base", 9000)

    telegraph_n = arg_int(opts, "telegraph-n", quick ? 5_000 : 50_000)
    repressilator_n = arg_int(opts, "repressilator-n", quick ? 300 : 1_000)
    telegraph_sweep_n = arg_int(opts, "telegraph-sweep-n", quick ? 5_000 : 20_000)
    repressilator_sweep_n = arg_int(opts, "repressilator-sweep-n", quick ? 300 : 1_000)
    schlogl_n = arg_int(opts, "schlogl-n", quick ? 300 : 1_000)

    telegraph_kons = parse_float_list(arg_string(opts, "telegraph-kons", "0.001,0.005,0.01,0.05,0.1"))
    repressilator_horizons = parse_int_list(arg_string(opts, "repressilator-horizons", "100,200,400,800"))

    scripts_dir = @__DIR__
    project_dir = dirname(scripts_dir)

    telegraph_script = joinpath(scripts_dir, "run_telegraph.jl")
    repressilator_script = joinpath(scripts_dir, "run_repressilator.jl")
    schlogl_script = joinpath(scripts_dir, "run_schlogl.jl")

    tasks = PlanTask[]

    # R1: Telegraph strict robustness across seeds
    if "R1" in only
        out_dir = joinpath(output_root, "robustness", "telegraph")
        for i in 1:seed_count
            ssa_seed = seed_ssa_base + i - 1
            sr_seed = seed_sr_base + i - 1
            tag = "r1_seed$(i)_ssa$(ssa_seed)_sr$(sr_seed)"
            args = [
                "--seed-ssa=$(ssa_seed)",
                "--seed-sr=$(sr_seed)",
                "--n-replicas=$(telegraph_n)",
                "--t-end=500.0",
                "--output-dir=$(out_dir)",
                "--tag=$(tag)",
                "--skip-plots",
            ]
            push_task!(tasks, "R1", "Telegraph strict robustness seed $i", telegraph_script, args, project_dir)
        end
    end

    # R2: Repressilator strict robustness across seeds
    if "R2" in only
        out_dir = joinpath(output_root, "robustness", "repressilator")
        for i in 1:seed_count
            ssa_seed = seed_ssa_base + i - 1
            sr_seed = seed_sr_base + i - 1
            tag = "r2_seed$(i)_ssa$(ssa_seed)_sr$(sr_seed)"
            args = [
                "--seed-ssa=$(ssa_seed)",
                "--seed-sr=$(sr_seed)",
                "--n-replicas=$(repressilator_n)",
                "--t-end=200.0",
                "--output-dir=$(out_dir)",
                "--tag=$(tag)",
                "--skip-plots",
            ]
            push_task!(tasks, "R2", "Repressilator strict robustness seed $i", repressilator_script, args, project_dir)
        end
    end

    # R3: Telegraph parameter sensitivity over k_on
    if "R3" in only
        out_dir = joinpath(output_root, "sweeps", "telegraph_rates")
        for kon in telegraph_kons
            tag = "r3_kon_$(fmt_for_tag(kon))"
            args = [
                "--seed-ssa=$(seed_ssa_base)",
                "--seed-sr=$(seed_sr_base)",
                "--n-replicas=$(telegraph_sweep_n)",
                "--t-end=500.0",
                "--k-on=$(kon)",
                "--output-dir=$(out_dir)",
                "--tag=$(tag)",
                "--skip-plots",
            ]
            push_task!(tasks, "R3", "Telegraph sensitivity k_on=$(kon)", telegraph_script, args, project_dir)
        end
    end

    # R4: Repressilator horizon sweep
    if "R4" in only
        out_dir = joinpath(output_root, "sweeps", "repressilator_horizon")
        for horizon in repressilator_horizons
            tag = "r4_tend_$(horizon)"
            args = [
                "--seed-ssa=$(seed_ssa_base)",
                "--seed-sr=$(seed_sr_base)",
                "--n-replicas=$(repressilator_sweep_n)",
                "--t-end=$(horizon)",
                "--output-dir=$(out_dir)",
                "--tag=$(tag)",
                "--skip-plots",
            ]
            push_task!(tasks, "R4", "Repressilator horizon t_end=$(horizon)", repressilator_script, args, project_dir)
        end
    end

    # R5: Schlogl framing runs (baseline + simple rescaled pilot)
    if "R5" in only
        out_dir = joinpath(output_root, "sweeps", "schlogl_rescaled")

        baseline_args = [
            "--seed-ssa=$(seed_ssa_base)",
            "--seed-sr=$(seed_sr_base)",
            "--n-replicas=$(schlogl_n)",
            "--t-end=5.0",
            "--mixed-set=safe",
            "--output-dir=$(out_dir)",
            "--tag=r5_baseline",
            "--skip-strict",
        ]
        push_task!(tasks, "R5", "Schlogl framing baseline", schlogl_script, baseline_args, project_dir)

        # Rescaled pilot: reduce reservoir concentrations to shrink propensity magnitudes.
        rescaled_args = [
            "--seed-ssa=$(seed_ssa_base)",
            "--seed-sr=$(seed_sr_base)",
            "--n-replicas=$(schlogl_n)",
            "--t-end=5.0",
            "--mixed-set=safe",
            "--A=20000.0",
            "--B=20000.0",
            "--output-dir=$(out_dir)",
            "--tag=r5_rescaled_pilot",
            "--strict-rtn=false",
        ]
        push_task!(tasks, "R5", "Schlogl framing rescaled pilot", schlogl_script, rescaled_args, project_dir)
    end

    isempty(tasks) && error("No tasks selected. Check --only option.")

    run_tasks(tasks; dry_run=dry_run)
end

main()
