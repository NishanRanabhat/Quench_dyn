# IO/serialization.jl
#
# JLD2 save/load for MPS snapshots, mirroring the KagomeDSF / U1 interface
# (save_mps(path, mps; meta) / load_mps(path) -> (mps, meta)) so the run
# scripts follow the same checkpoint-then-postprocess workflow:
#
#   evolution script:  saves one .jld2 per measurement time into a
#                      parameter-tagged run directory under `run_root()`
#   post-processing:   loads each snapshot and measures whatever it wants
#                      (FCS, correlators, entropies, ...) — observables are
#                      DECOUPLED from the expensive evolution.
#
# On disk each file holds "tensors" (the plain Vector{Array{T,3}} — no
# custom struct, so files load in any Julia session with JLD2 alone) and
# "meta" (a Dict{String,Any} of parameters: t, dt, Δi, Δf, χ, energy, ...).
#
# Run root: `run_root()` is ./runs next to the repository by default and is
# overridden by ENV["QUENCHDYN_RUN_ROOT"] — on Rivanna the SLURM script
# exports QUENCHDYN_RUN_ROOT=/scratch/$USER/QuenchDyn/runs so bulky MPS
# snapshots land on scratch, never in the repo or home directory. See
# docs/mps_checkpointing.md.

"""
    save_mps(path, mps::MPS; meta=Dict{String,Any}())

Serialize an MPS and a metadata dict to a JLD2 file. Stores the raw tensor
vector (plain arrays), so no package types are baked into the file. `meta`
is copied to `Dict{String,Any}` so any AbstractDict can be passed.
"""
function save_mps(path::AbstractString, mps::MPS;
                  meta::AbstractDict = Dict{String,Any}())
    jldopen(path, "w") do file
        file["tensors"] = mps.tensors
        file["meta"]    = Dict{String,Any}(meta)
    end
    return nothing
end

"""
    load_mps(path) -> (mps::MPS, meta::Dict{String,Any})

Inverse of [`save_mps`](@ref).
"""
function load_mps(path::AbstractString)
    tensors, meta = jldopen(path, "r") do file
        return file["tensors"], file["meta"]
    end
    return MPS{eltype(tensors[1])}(tensors), meta
end

"""
    run_root() -> String

Root directory for run outputs: `ENV["QUENCHDYN_RUN_ROOT"]` if set (on the
cluster: a scratch path), else `<repo>/runs`. Created on first use.
"""
function run_root()
    root = get(ENV, "QUENCHDYN_RUN_ROOT",
               normpath(joinpath(@__DIR__, "..", "..", "runs")))
    mkpath(root)
    return root
end

"""
    snapshot_name(t) -> String

Canonical, lexicographically sortable snapshot filename for time `t`:
`mps_t0012.500.jld2`. Post-processing scripts glob `mps_t*.jld2` and rely on
this zero-padded format for time ordering.
"""
snapshot_name(t::Real) = @sprintf("mps_t%08.3f.jld2", t)

"""
    snapshot_time(filename) -> Float64

Parse the time back out of a [`snapshot_name`](@ref)-style filename.
"""
snapshot_time(f::AbstractString) =
    parse(Float64, match(r"mps_t([0-9.]+)\.jld2$", basename(f)).captures[1])
