"""
Lightweight CLI argument helpers for experiment scripts.

Supported forms:
  --key=value
  --key value
  --flag            (stored as "true")
"""

"""
    parse_cli_args(args=ARGS) -> Dict{String,String}
"""
function parse_cli_args(args=ARGS)
    opts = Dict{String, String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        startswith(arg, "--") || error("Unexpected positional argument: $arg")
        body = arg[3:end]
        isempty(body) && error("Invalid argument: --")

        if occursin('=', body)
            key, val = split(body, '='; limit=2)
            isempty(key) && error("Invalid option: $arg")
            opts[key] = val
        else
            key = body
            if i < length(args) && !startswith(args[i + 1], "--")
                opts[key] = args[i + 1]
                i += 1
            else
                opts[key] = "true"
            end
        end
        i += 1
    end
    return opts
end

"""
    arg_string(opts, key, default) -> String
"""
arg_string(opts::Dict{String, String}, key::String, default::String) = get(opts, key, default)

"""
    arg_int(opts, key, default) -> Int
"""
function arg_int(opts::Dict{String, String}, key::String, default::Int)
    val = get(opts, key, nothing)
    val === nothing && return default
    parsed = tryparse(Int, val)
    parsed === nothing && error("Invalid integer for --$key: $val")
    return parsed
end

"""
    arg_float(opts, key, default) -> Float64
"""
function arg_float(opts::Dict{String, String}, key::String, default::Float64)
    val = get(opts, key, nothing)
    val === nothing && return default
    parsed = tryparse(Float64, val)
    parsed === nothing && error("Invalid float for --$key: $val")
    return parsed
end

"""
    arg_bool(opts, key, default) -> Bool

Truthy: true, 1, yes, y, on
Falsy:  false, 0, no, n, off
"""
function arg_bool(opts::Dict{String, String}, key::String, default::Bool)
    val = get(opts, key, nothing)
    val === nothing && return default
    s = lowercase(strip(val))
    if s in ("true", "1", "yes", "y", "on")
        return true
    elseif s in ("false", "0", "no", "n", "off")
        return false
    end
    error("Invalid boolean for --$key: $val")
end

"""
    sanitize_tag(tag) -> String

Keep alphanumeric, '.', '-', '_' and replace everything else with '_'.
"""
function sanitize_tag(tag::String)
    cleaned = replace(strip(tag), r"[^A-Za-z0-9._-]+" => "_")
    return strip(cleaned, '_')
end

"""
    model_name_with_tag(base, tag) -> String

If `tag` is empty, returns `base`. Else returns `base_tag`.
"""
function model_name_with_tag(base::String, tag::String)
    t = sanitize_tag(tag)
    return isempty(t) ? base : "$(base)_$(t)"
end

"""
    current_commit_hash() -> String

Best-effort git commit hash; returns "unknown" outside a git repo.
"""
function current_commit_hash()
    try
        return chomp(read(`git rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end
