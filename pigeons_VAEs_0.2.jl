# p786_project_VAEs_pigeons.jl
#
# VAE applied to homing-pigeon GPS trajectories (PHYS486/786).
#
# Adapted from p786_project_VAEs.jl. The VAE core / loss / training loop are
# unchanged; the image datasets are replaced by pigeon trajectories extracted
# exactly as in plot_data.jl. Each trajectory (a sequence of lat/long points)
# is padded to a common length and fed to a VAE with a 2-D latent space and a
# 4-hidden-layer encoder/decoder, to check whether the runs cluster in latent
# space (e.g. by release site).
#
# Stack: Flux.jl (model/training), CSV.jl + DataFrames.jl (data), Plots.jl.
#
# Setup (run once):
#   using Pkg
#   Pkg.add(["Flux", "CSV", "DataFrames", "Plots", "MLUtils"])
#   # Optional GPU:  Pkg.add("CUDA")
#
# Differences from the image VAE, and why:
#   * Reconstruction loss is MSE, not binary cross-entropy: GPS coordinates are
#     continuous, not Bernoulli pixels. (BCE version kept below, commented.)
#   * Inputs are normalized by a SINGLE GLOBAL function shared across all
#     trajectories (constants computed once over the whole dataset), switchable
#     between :minmax (-> [0,1], sigmoid output) and :zscore (mean 0/std 1,
#     identity output). The output activation follows the choice automatically.
#   * "4 hidden dimensions" is read as a 4-hidden-layer encoder/decoder,
#     hidden_dims = [1024, 256, 64, 16] (paralleling the earlier VAE3/VAE5).

using Flux
using Flux: DataLoader, withgradient, setup, update!
using CSV, DataFrames
using Random
using Statistics
using Plots

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
const HAS_CUDA = try
    using CUDA
    using cuDNN
    CUDA.functional()
catch
    false
end
device(x) = HAS_CUDA ? gpu(x) : x
@info "Compute device" gpu = HAS_CUDA

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
# Latent space dimension, used everywhere as the default. Change it here once
# (e.g. Z_DIM = 3 for a 3-D latent space + 3-D plots). Every runner/plot still
# accepts a `z_dim` keyword to override per-call if needed.
Z_DIM = 3

# ---------------------------------------------------------------------------
# Depth-parametric VAE (unchanged from p786_project_VAEs.jl)
# ---------------------------------------------------------------------------
struct VAE{E, M, L, D}
    encoder::E
    mean_layer::M
    logvar_layer::L
    decoder::D
end

Flux.@layer VAE

# `out` is the decoder's output activation: sigmoid for [0,1] (min-max) targets,
# identity for unbounded (z-score) targets.
function VAE(z_dim::Int; input_dim::Int = 784, hidden_dims::Vector{Int} = [400, 200],
            out = sigmoid)
    enc_dims = vcat(input_dim, hidden_dims)
    encoder  = Chain([Dense(enc_dims[i] => enc_dims[i+1], relu) for i in 1:length(enc_dims)-1]...)

    mean_layer   = Dense(hidden_dims[end] => z_dim)
    logvar_layer = Dense(hidden_dims[end] => z_dim)

    dec_dims = vcat(z_dim, reverse(hidden_dims))
    dec_hidden = [Dense(dec_dims[i] => dec_dims[i+1], relu) for i in 1:length(dec_dims)-1]
    decoder  = Chain(dec_hidden..., Dense(hidden_dims[1] => input_dim, out))

    return VAE(encoder, mean_layer, logvar_layer, decoder)
end

encode(m::VAE, x) = (h = m.encoder(x); (m.mean_layer(h), m.logvar_layer(h)))
decode(m::VAE, z) = m.decoder(z)

function reparameterize(μ, logσ²)
    ε = randn!(similar(logσ²))
    return μ .+ exp.(0.5f0 .* logσ²) .* ε
end

function (m::VAE)(x)
    μ, logσ² = encode(m, x)
    z  = reparameterize(μ, logσ²)
    x̂ = decode(m, z)
    return x̂, μ, logσ²
end

# ---------------------------------------------------------------------------
# Loss functions — MSE reconstruction for continuous coordinates
# ---------------------------------------------------------------------------
recon_loss(x, x̂) = Flux.Losses.mse(x̂, x; agg = sum)
# Bernoulli/pixel alternative, if you prefer the original BCE:
# recon_loss(x, x̂) = Flux.Losses.binarycrossentropy(x̂, x; agg = sum)

kld_loss(μ, logσ²) = -0.5f0 * sum(1f0 .+ logσ² .- μ .^ 2 .- exp.(logσ²))

loss_function(x, x̂, μ, logσ²)       = recon_loss(x, x̂) + kld_loss(μ, logσ²)
loss_function_recon(x, x̂, μ, logσ²) = recon_loss(x, x̂)
loss_function_kld(x, x̂, μ, logσ²)   = kld_loss(μ, logσ²)

# ---------------------------------------------------------------------------
# Training loop (unchanged)
# ---------------------------------------------------------------------------
function train!(model, opt_state, train_loader, epochs::Int;
                lossfn = loss_function, log_every::Int = 10)
    final_avg = 0.0
    overall   = 0.0
    for epoch in 1:epochs
        overall = 0.0
        nseen   = 0
        for (x, _) in train_loader
            x = device(x)
            loss, grads = withgradient(model) do m
                x̂, μ, logσ² = m(x)
                lossfn(x, x̂, μ, logσ²)
            end
            update!(opt_state, model, grads[1])
            overall += loss
            nseen   += size(x, 2)
        end
        final_avg = overall / nseen
        if (epoch - 1) % log_every == 0
            println("\t Epoch ", epoch, "\t Average Loss: ", final_avg)
        end
    end
    return overall, final_avg
end

# Masked reconstruction: padded positions (mask == 0) contribute zero error and
# therefore zero gradient, so the padding choice can't influence the latent code.
# Same sum-of-squared-errors convention as recon_loss above.
masked_recon_loss(x, x̂, m) = sum(m .* abs2.(x̂ .- x))

# Mask-aware training loop: the loader yields (x, mask) batches.
function train_masked!(model, opt_state, loader, epochs::Int; log_every::Int = 10)
    final_avg = 0.0
    overall   = 0.0
    for epoch in 1:epochs
        overall = 0.0
        nseen   = 0
        for (x, m) in loader
            x = device(x); m = device(m)
            loss, grads = withgradient(model) do mdl
                x̂, μ, logσ² = mdl(x)
                masked_recon_loss(x, x̂, m) + kld_loss(μ, logσ²)
            end
            update!(opt_state, model, grads[1])
            overall += loss
            nseen   += size(x, 2)
        end
        final_avg = overall / nseen
        if (epoch - 1) % log_every == 0
            println("\t Epoch ", epoch, "\t Average Loss: ", final_avg)
        end
    end
    return overall, final_avg
end

# ===========================================================================
# Pigeon trajectory data — extracted exactly as in plot_data.jl
# ===========================================================================
const PIGEON_LIST = ["A_A35", "A_A38", "A_A45", "A_D06", "A_D09", "A_D16",
                     "A_D83", "A_D94", "A_D96", "B_A31", "B_A34", "B_D04",
                     "B_D12", "B_D86", "B_D90", "B_D98", "C_A39", "C_A43",
                     "C_A47", "C_D00", "C_D01", "C_D08", "C_D10", "C_D15",
                     "C_D88", "C_D93"]

const N_SITE   = 3   # release sites R1..R3
const N_PIGEON = 26  # pigeons 1..26
const N_RUN    = 6   # homing runs 01..06

# Reads every CSV into the 3x26x6 lat/long arrays (the all_trajs structure).
# Each lats[i,j,k] / longs[i,j,k] is the vector of points for one homing run.
function load_pigeon_trajectories(; data_root::AbstractString = "data")
    file_paths = ["$(data_root)/R$(i)/$(PIGEON_LIST[j])/$(PIGEON_LIST[j])_R$(i)_0$(k).csv"
                  for i in 1:N_SITE, j in 1:N_PIGEON, k in 1:N_RUN]
    raw_data = [DataFrame(CSV.File(file_paths[i, j, k]))
                for i in 1:N_SITE, j in 1:N_PIGEON, k in 1:N_RUN]
    # Column names carry a leading space, exactly as in plot_data.jl.
    lats  = [raw_data[i, j, k][!, " Latitude"]  for i in 1:N_SITE, j in 1:N_PIGEON, k in 1:N_RUN]
    longs = [raw_data[i, j, k][!, " Longitude"] for i in 1:N_SITE, j in 1:N_PIGEON, k in 1:N_RUN]
    return lats, longs
end

# Number of padding points to place before / after a length-T trajectory in a
# length-L window. Shared by pad_series and the loss mask so they always agree.
function pad_counts(T::Int, L::Int, align::Symbol)
    total = L - T
    total <= 0 && return (0, 0)
    return align === :center ? (total ÷ 2, total - total ÷ 2) :
           align === :pre    ? (total, 0) :
           align === :post   ? (0, total) :
           error("align must be :center, :pre, or :post")
end

# Pad one trajectory to length L. The total padding L-T is split before/after;
# pre-padding repeats the START point, post-padding repeats the END point.
#   align = :center  -> half before, half after (default; honors both rules)
#   align = :pre     -> all padding before  (trajectory right-aligned)
#   align = :post    -> all padding after   (trajectory left-aligned)
function pad_series(v::AbstractVector, L::Int; align::Symbol = :center)
    pre, post = pad_counts(length(v), L, align)
    (pre, post) == (0, 0) && return collect(v)
    return vcat(fill(v[1], pre), collect(v), fill(v[end], post))
end

# Build the (input_dim, N) feature matrix plus label vectors.
# A trajectory becomes [normalized padded lats; normalized padded longs],
# so input_dim = 2L.
#
# NORMALIZATION IS A SINGLE GLOBAL FUNCTION SHARED BY ALL TRAJECTORIES: the
# constants (lat_a, lat_b, lon_a, lon_b) are computed ONCE over all 468
# trajectories and the identical affine map  x -> (x - a) / b  is applied to
# every one. Two choices, both global:
#   normalization = :minmax  ->  a = min,  b = max - min   (targets in [0,1]; use sigmoid output)
#   normalization = :zscore  ->  a = mean, b = std         (mean 0 / std 1; use identity output)
# De-normalize anywhere with  x = xnorm * b + a.
function build_dataset(lats, longs; align::Symbol = :center, normalization::Symbol = :minmax,
                       sites = 1:N_SITE)
    site_ids = collect(sites)
    triples  = [(i, j, k) for i in site_ids for j in 1:N_PIGEON for k in 1:N_RUN]  # k fastest
    L = maximum(length(lats[i, j, k]) for (i, j, k) in triples)      # max trajectory length

    # Flatten every included coordinate once, to compute the shared constants.
    all_lat = reduce(vcat, (collect(lats[i, j, k])  for (i, j, k) in triples))
    all_lon = reduce(vcat, (collect(longs[i, j, k]) for (i, j, k) in triples))

    if normalization === :minmax
        lat_a, lat_b = minimum(all_lat), maximum(all_lat) - minimum(all_lat)
        lon_a, lon_b = minimum(all_lon), maximum(all_lon) - minimum(all_lon)
    elseif normalization === :zscore
        lat_a, lat_b = mean(all_lat), std(all_lat)
        lon_a, lon_b = mean(all_lon), std(all_lon)
    else
        error("normalization must be :minmax or :zscore")
    end
    lat_b = lat_b == 0 ? 1.0 : lat_b
    lon_b = lon_b == 0 ? 1.0 : lon_b
    @info "Shared global normalization (same function for every trajectory in this run)" sites = site_ids normalization lat_a lat_b lon_a lon_b

    N = length(triples)
    X    = Array{Float32}(undef, 2L, N)
    mask = Array{Float32}(undef, 2L, N)                              # 1 = real point, 0 = padding
    site = Vector{Int}(undef, N)
    pid  = Vector{Int}(undef, N)
    run  = Vector{Int}(undef, N)

    for (col, (i, j, k)) in enumerate(triples)
        T = length(lats[i, j, k])
        pre, post = pad_counts(T, L, align)
        plat = pad_series(lats[i, j, k],  L; align = align)
        plon = pad_series(longs[i, j, k], L; align = align)
        latn = (plat .- lat_a) ./ lat_b                             # identical map for all trajectories
        lonn = (plon .- lon_a) ./ lon_b
        X[:, col]  = Float32.(vcat(latn, lonn))
        rm = vcat(zeros(Float32, pre), ones(Float32, T), zeros(Float32, post))  # real-point mask
        mask[:, col] = vcat(rm, rm)                                 # same pattern for lat & lon blocks
        site[col]  = i
        pid[col]   = j
        run[col]   = k
    end

    meta = (; L, normalization, lat_a, lat_b, lon_a, lon_b, mask, sites = site_ids)
    return X, site, pid, run, meta
end

# ===========================================================================
# Visualization
# ===========================================================================
# Latent means for every trajectory -> (z_dim, N).
function latent_coords(model, X)
    μ, logσ² = encode(model, device(X))
    return cpu(reparameterize(μ, logσ²))
end

# Draw a latent scatter that is 3-D when z_dim >= 3, otherwise 2-D. All color/
# group/legend options are forwarded through `kwargs...`. We branch on the
# actual number of latent rows too, so a z_dim of 3 never indexes a missing row.
function latent_scatter(z, z_dim, title; kwargs...)
    if z_dim >= 3 && size(z, 1) >= 3
        scatter(z[1, :], z[2, :], z[3, :]; markersize = 3, markerstrokewidth = 0,
                xlabel = "z[1]", ylabel = "z[2]", zlabel = "z[3]", title = title, kwargs...)
    else
        scatter(z[1, :], z[2, :]; markersize = 3, markerstrokewidth = 0,
                xlabel = "z[1]", ylabel = "z[2]", title = title, kwargs...)
    end
end

# Scatter the latent space, colored by release site (the clustering test).
function plot_latent_by_site(model, X, site; z_dim::Int = Z_DIM)
    z = latent_coords(model, X)
    latent_scatter(z, z_dim, "Pigeon trajectories in VAE latent space (by release site)";
                   group = ["R$s" for s in site], palette = :tab10, legend = :outertopright)
end

# Same latent space, colored by pigeon id (continuous colorbar, 26 pigeons).
function plot_latent_by_pigeon(model, X, pid; z_dim::Int = Z_DIM)
    z = latent_coords(model, X)
    latent_scatter(z, z_dim, "Pigeon trajectories in VAE latent space (by pigeon)";
                   zcolor = pid, c = :viridis, colorbar_title = "pigeon id", legend = false)
end

# Same latent space, colored by homing-run number (1..6) — the within-site
# "learning dynamics" variable: do later runs converge/separate from early ones?
function plot_latent_by_run(model, X, run; z_dim::Int = Z_DIM)
    z = latent_coords(model, X)
    runpal = palette(:viridis, N_RUN)     # 6 DISTINCT colors sampled from viridis
    latent_scatter(z, z_dim, "Pigeon trajectories in VAE latent space (by run)";
                   group = ["run $r" for r in run], palette = runpal, legend = :outertopright)
end

# Sanity check: overlay one original trajectory and its VAE reconstruction
# (both de-normalized back to lat/long).
function plot_reconstruction(model, X, meta, col::Int)
    L = meta.L
    x = device(X[:, col])
    μ, _ = encode(model, x)               # encode to the latent space first...
    x̂ = cpu(decode(model, μ))             # ...then decode the latent MEAN (no sampling noise)
    denlat(v) = v .* meta.lat_b .+ meta.lat_a
    denlon(v) = v .* meta.lon_b .+ meta.lon_a

    olat, olon = denlat(X[1:L, col]),       denlon(X[L+1:2L, col])
    rlat, rlon = denlat(x̂[1:L]),            denlon(x̂[L+1:2L])

    plt = plot(olon, olat; label = "original", lw = 2, xlabel = "longitude",
               ylabel = "latitude", title = "Trajectory $col: original vs. reconstruction")
    plot!(plt, rlon, rlat; label = "reconstruction", lw = 2, ls = :dash)
    return plt
end

# ===========================================================================
# Main experiment
# ===========================================================================
function run_pigeons(; data_root::AbstractString = "data",
                       z_dim::Int = Z_DIM,
                       align::Symbol = :center,
                       normalization::Symbol = :minmax,
                       hidden_dims::Vector{Int} = [1024, 256, 64, 16],  # 4 hidden layers, first = 1024
                       epochs::Int = 200, batchsize::Int = 64)
    lats, longs = load_pigeon_trajectories(; data_root = data_root)
    X, site, pid, run, meta = build_dataset(lats, longs; align = align, normalization = normalization)
    input_dim = size(X, 1)
    @info "Dataset" trajectories = size(X, 2) max_len = meta.L input_dim = input_dim z_dim = z_dim

    out = normalization === :minmax ? sigmoid : identity   # match the target range
    loader = DataLoader((X, site); batchsize = batchsize, shuffle = true)
    model  = device(VAE(z_dim; input_dim = input_dim, hidden_dims = hidden_dims, out = out))
    opt    = setup(Adam(1f-3), model)
    train!(model, opt, loader, epochs)

    display(plot_latent_by_site(model, X, site; z_dim = z_dim))
    display(plot_latent_by_pigeon(model, X, pid; z_dim = z_dim))
    display(plot_reconstruction(model, X, meta, 1))

    return model, (; X, site, pid, run, meta)
end

# Fit a VAE on a single release site and return (model, dataset) so the latent
# space reflects only within-site variation (between pigeons / across runs),
# not the trivial between-site separation.
function run_one_site(site_id::Int; data_root::AbstractString = "data",
                      z_dim::Int = Z_DIM,
                      align::Symbol = :center, normalization::Symbol = :minmax,
                      hidden_dims::Vector{Int} = [1024, 256, 64, 16],
                      epochs::Int = 200, batchsize::Int = 64,
                      masked::Bool = false, seed::Int = 0,
                      preloaded = nothing)
    lats, longs = preloaded === nothing ?
        load_pigeon_trajectories(; data_root = data_root) : preloaded
    X, site, pid, run, meta = build_dataset(lats, longs; align = align,
                                            normalization = normalization, sites = [site_id])
    @info "Site R$site_id" trajectories = size(X, 2) max_len = meta.L input_dim = size(X, 1) z_dim = z_dim

    Random.seed!(seed)
    out   = normalization === :minmax ? sigmoid : identity
    model = device(VAE(z_dim; input_dim = size(X, 1), hidden_dims = hidden_dims, out = out))
    opt   = setup(Adam(1f-3), model)
    if masked
        train_masked!(model, opt, DataLoader((X, meta.mask); batchsize = batchsize, shuffle = true), epochs)
    else
        train!(model, opt, DataLoader((X, pid); batchsize = batchsize, shuffle = true), epochs)
    end
    return model, (; X, site, pid, run, meta)
end

# Run R1, R2, R3 each on their OWN VAE (one model per site, trained only on that
# site's 156 trajectories). Shows a 3x2 grid: rows = release site, columns =
# latent space colored by pigeon id and by run number. Each site is trained
# independently with the same seed so the panels are comparable.
function run_sites_separately(; data_root::AbstractString = "data",
                                z_dim::Int = Z_DIM,
                                align::Symbol = :center, normalization::Symbol = :minmax,
                                hidden_dims::Vector{Int} = [1024, 256, 64, 16],
                                epochs::Int = 200, batchsize::Int = 64,
                                masked::Bool = false, seed::Int = 0)
    preloaded = load_pigeon_trajectories(; data_root = data_root)
    runpal = palette(:viridis, N_RUN)     # 6 distinct colors, one per run
    panels = Any[]
    for s in 1:N_SITE
        println("=== Release site R$s ===")
        model, d = run_one_site(s; z_dim = z_dim, align = align, normalization = normalization,
                                hidden_dims = hidden_dims, epochs = epochs,
                                batchsize = batchsize, masked = masked, seed = seed,
                                preloaded = preloaded)
        z = latent_coords(model, d.X)
        push!(panels, latent_scatter(z, z_dim, "R$s — by pigeon";
                                     zcolor = d.pid, c = :viridis,
                                     colorbar_title = "pigeon", legend = false))
        push!(panels, latent_scatter(z, z_dim, "R$s — by run";
                                     group = ["run $r" for r in d.run],
                                     palette = runpal, legend = :outertopright))
    end
    plt = plot(panels...; layout = (N_SITE, 2), size = (1200, 1400),
               plot_title = "Within-site latent clustering (each site its own VAE)")
    display(plt)
    return plt
end
# identical settings, plus a fourth panel that masks padded positions out of the
# reconstruction loss. The masked panel is alignment-invariant (padding has zero
# gradient), so it serves as the control: if the three unmasked panels disagree
# but the masked one differs from them, the apparent clusters were padding-driven.
function run_alignment_comparison(; data_root::AbstractString = "data",
                                    z_dim::Int = Z_DIM,
                                    normalization::Symbol = :minmax,
                                    hidden_dims::Vector{Int} = [1024, 256, 64, 16],
                                    epochs::Int = 200, batchsize::Int = 64,
                                    seed::Int = 0)
    lats, longs = load_pigeon_trajectories(; data_root = data_root)
    out = normalization === :minmax ? sigmoid : identity

    site_panel(z, site, ttl) =
        latent_scatter(z, z_dim, ttl; group = ["R$s" for s in site],
                       palette = :tab10, legend = :outertopright)

    # Panels 1-3: standard (unmasked) loss under each alignment.
    unmasked = map((:pre, :center, :post)) do align
        Random.seed!(seed)   # same init/shuffle across panels -> fair comparison
        X, site, _, _, _ = build_dataset(lats, longs; align = align, normalization = normalization)
        model = device(VAE(z_dim; input_dim = size(X, 1), hidden_dims = hidden_dims, out = out))
        opt   = setup(Adam(1f-3), model)
        println("=== align = :$align (unmasked) ===")
        train!(model, opt, DataLoader((X, site); batchsize = batchsize, shuffle = true), epochs)
        site_panel(latent_coords(model, X), site, "align = :$align")
    end

    # Panel 4: masked loss. Alignment is irrelevant here, so :center is used.
    Random.seed!(seed)
    Xm, sitem, _, _, metam = build_dataset(lats, longs; align = :center, normalization = normalization)
    modelm = device(VAE(z_dim; input_dim = size(Xm, 1), hidden_dims = hidden_dims, out = out))
    optm   = setup(Adam(1f-3), modelm)
    println("=== masked (alignment-invariant) ===")
    train_masked!(modelm, optm, DataLoader((Xm, metam.mask); batchsize = batchsize, shuffle = true), epochs)
    masked_panel = site_panel(latent_coords(modelm, Xm), sitem, "padding masked (alignment-invariant)")

    plt = plot(unmasked..., masked_panel; layout = (2, 2), size = (1200, 950),
               plot_title = "Latent space by release site: padding alignments vs. masked loss")
    display(plt)
    return plt
end

# ---------------------------------------------------------------------------
# Entry point. Adjust data_root to wherever the R1/R2/R3 folders live.
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    # All sites together (clusters trivially by release site):
    # run_pigeons(; data_root = "data")

    # Each release site on its own VAE — examine WITHIN-site clustering:
    run_sites_separately(; data_root = "data")

    # Padding-alignment + masked-loss diagnostic (all sites):
    # run_alignment_comparison(; data_root = "data")
end

run_sites_separately(; data_root = "data")