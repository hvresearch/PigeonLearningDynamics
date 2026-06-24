# p786_project_VAEs.jl
#
# Julia translation of the second PyTorch VAE notebook (PHYS486/786, A. Shah).
# Builds on the same VAE core as phys786_kld_vs_recon.jl and adds:
#   * a depth-parametric VAE (covers the notebook's VAE / VAE3 / VAE5 classes),
#   * basic MNIST training + sampling + latent-space tiling,
#   * average-loss vs. latent-dimension sweep,
#   * Fashion-MNIST, including a 3-D latent scatter,
#   * EuroSAT (64x64 RGB) with a 5-hidden-layer VAE and RGB visualizations.
#
# Stack: Flux.jl (model/training), MLDatasets.jl (data), Plots.jl + Colors.jl
# (figures).
#
# Setup (run once):
#   using Pkg
#   Pkg.add(["Flux", "MLDatasets", "Plots", "Colors", "MLUtils"])
#   # Optional GPU:  Pkg.add("CUDA")
#
# PyTorch -> Flux notes (same conventions as the companion file):
#   * Flux is column-major: tensors are (features, batch), the transpose of
#     PyTorch's (batch, features). Images are flattened to (input_dim, N).
#   * nn.LeakyReLU(0.0) == ReLU, so `relu` is used.
#   * The notebook's VAE / VAE3 / VAE5 differ ONLY in their hidden-layer list,
#     so they are unified into one `VAE(z_dim; hidden_dims=[...])` constructor:
#         VAE(2)                                  <-> VAE(2; hidden_dims=[400,200])
#         VAE3(2, hidden_dim=[400,200,100])       <-> VAE(2; hidden_dims=[400,200,100])
#         VAE5(3, hidden_dim=[6000,3000,1000,200,64])
#                                                 <-> VAE(3; hidden_dims=[6000,3000,1000,200,64])

using Flux
using Flux: DataLoader, withgradient, setup, update!
using Flux.Losses: binarycrossentropy
using MLDatasets
using Random
using Statistics
using Colors
using Plots

# ---------------------------------------------------------------------------
# Device selection (Cells 0/1)
# ---------------------------------------------------------------------------
const HAS_CUDA = try
    using CUDA, cuDNN
    CUDA.functional()
catch
    false
end
device(x) = HAS_CUDA ? gpu(x) : x
@info "Compute device" gpu = HAS_CUDA

# ---------------------------------------------------------------------------
# Depth-parametric VAE (Cells 3/24/34)
# ---------------------------------------------------------------------------
struct VAE{E, M, L, D}
    encoder::E
    mean_layer::M
    logvar_layer::L
    decoder::D
end

Flux.@layer VAE

# `hidden_dims` lists the encoder widths between input and the mean/logvar
# heads; the decoder mirrors it. relu == LeakyReLU(0.0) from the notebook.
function VAE(z_dim::Int; input_dim::Int = 784, hidden_dims::Vector{Int} = [400, 200])
    enc_dims = vcat(input_dim, hidden_dims)
    encoder  = Chain([Dense(enc_dims[i] => enc_dims[i+1], relu) for i in 1:length(enc_dims)-1]...)

    mean_layer   = Dense(hidden_dims[end] => z_dim)
    logvar_layer = Dense(hidden_dims[end] => z_dim)

    dec_dims = vcat(z_dim, reverse(hidden_dims))
    dec_hidden = [Dense(dec_dims[i] => dec_dims[i+1], relu) for i in 1:length(dec_dims)-1]
    decoder  = Chain(dec_hidden..., Dense(hidden_dims[1] => input_dim, sigmoid))

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
# Loss functions (Cell 4)
# ---------------------------------------------------------------------------
recon_loss(x, x̂) = binarycrossentropy(x̂, x; agg = sum)
kld_loss(μ, logσ²) = -0.5f0 * sum(1f0 .+ logσ² .- μ .^ 2 .- exp.(logσ²))

loss_function(x, x̂, μ, logσ²)       = recon_loss(x, x̂) + kld_loss(μ, logσ²)
loss_function_recon(x, x̂, μ, logσ²) = recon_loss(x, x̂)
loss_function_kld(x, x̂, μ, logσ²)   = kld_loss(μ, logσ²)

# ---------------------------------------------------------------------------
# Training loop (Cells 5/16) — single generic version
# ---------------------------------------------------------------------------
function train!(model, opt_state, train_loader, epochs::Int;
                lossfn = loss_function, log_every::Int = 1)
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

# ===========================================================================
# Data loaders
# ===========================================================================
# MLDatasets features are Float32 in [0,1] (== torchvision ToTensor).
function load_grayscale(Dataset; batchsize::Int = 100)
    train = Dataset(split = :train)
    test  = Dataset(split = :test)
    Xtr = reshape(Float32.(train.features), :, size(train.features, 3))
    Xte = reshape(Float32.(test.features),  :, size(test.features, 3))
    train_loader = DataLoader((Xtr, train.targets); batchsize = batchsize, shuffle = true)
    return train_loader, Xte, test.targets
end

load_mnist(; kw...)         = load_grayscale(MNIST; kw...)
load_fashionmnist(; kw...)  = load_grayscale(FashionMNIST; kw...)

# ===========================================================================
# Grayscale visualizations (Cells 8/9/10/11/12/17)
# ===========================================================================
function generate_sample_out(model, z::AbstractVector, sample_size::Int)
    x̂ = cpu(decode(model, device(Float32.(z))))
    return reshape(x̂, sample_size, sample_size)
end

# Cell 8: decode and show a single (z1, z2) point.
function generate_digit(model, z1, z2; sample_size::Int = 28)
    digit = generate_sample_out(model, Float32[z1, z2], sample_size)
    heatmap(digit; color = :grays, yflip = true, aspect_ratio = 1,
            axis = false, legend = false, title = "z = ($z1, $z2)")
end

# Cells 11/12: 5x5 montage decoded from random N(0, I) latents.
function generate_random_samples(model, z_dim::Int; sample_size::Int = 28)
    imgs = [generate_sample_out(model, randn(Float32, z_dim), sample_size) for _ in 1:25]
    plts = [heatmap(im; color = :grays, yflip = true, aspect_ratio = 1,
                    axis = false, legend = false) for im in imgs]
    plot(plts...; layout = (5, 5), size = (500, 500))
end

# Cells 9/10: tile decoded digits over a regular grid in 2-D latent space.
function plot_latent_space(model, scale::Real, n::Int; digit_size::Int = 28)
    figure = zeros(Float32, digit_size * n, digit_size * n)
    grid   = range(-scale, scale; length = n)
    for (i, yi) in enumerate(grid), (j, xi) in enumerate(grid)
        digit = generate_sample_out(model, Float32[xi, yi], digit_size)
        figure[(i-1)*digit_size+1:i*digit_size, (j-1)*digit_size+1:j*digit_size] = digit
    end
    xt = round.(collect(grid); digits = 1)
    heatmap(figure; color = :grays, yflip = true, aspect_ratio = 1,
            title = "Latent space visualization", xlabel = "z[0]", ylabel = "z[1]",
            xticks = (range(digit_size ÷ 2, step = digit_size, length = n), xt),
            yticks = (range(digit_size ÷ 2, step = digit_size, length = n), xt))
end

# Cell 17: 2-D scatter of latent means, colored by label.
function plot_latent_space_raw(model, Xte, yte; title = "Latent space (raw)")
    μ, logσ² = encode(model, device(Xte))
    z = cpu(reparameterize(μ, logσ²))
    scatter(z[1, :], z[2, :]; group = yte, markersize = 1.2, markerstrokewidth = 0,
            legend = :outertopright, xlabel = "z[0]", ylabel = "z[1]",
            title = title, palette = :tab10)
end

# Cells 27/28: 3-D scatter of latent means (requires z_dim >= 3).
function plot_latent_space_raw3(model, Xte, yte; title = "Latent space (3-D)")
    μ, logσ² = encode(model, device(Xte))
    z = cpu(reparameterize(μ, logσ²))
    scatter(z[1, :], z[2, :], z[3, :]; group = yte, markersize = 1.2,
            markerstrokewidth = 0, legend = :outertopright,
            xlabel = "z[0]", ylabel = "z[1]", zlabel = "z[2]",
            title = title, palette = :tab10)
end

# ===========================================================================
# Section: basic MNIST (Cells 6–12)
# ===========================================================================
function run_mnist_basic()
    train_loader, Xte, yte = load_mnist(batchsize = 100)
    model = device(VAE(2))                          # hidden_dims = [400, 200]
    opt   = setup(Adam(1f-3), model)
    train!(model, opt, train_loader, 50)

    display(generate_digit(model, 0.0, 1.0))
    display(generate_digit(model, 1.0, 0.0))
    display(plot_latent_space(model, 3, 30))
    display(generate_random_samples(model, 2))
    return model
end

# ===========================================================================
# Section: average loss vs. latent dimension (Cells 13/14)
# ===========================================================================
function run_loss_vs_zdim(; z_dims = [1, 2, 5, 10, 20], epochs = 50)
    train_loader, _, _ = load_mnist(batchsize = 100)
    avg_loss = Float64[]
    for z_dim in z_dims
        model = device(VAE(z_dim))                  # hidden_dims = [400, 200]
        opt   = setup(Adam(1f-3), model)
        _, final = train!(model, opt, train_loader, epochs; log_every = epochs)
        push!(avg_loss, final)
        display(generate_random_samples(model, z_dim))
    end
    plt = plot(z_dims, avg_loss; marker = :circle, label = "avg. loss",
               title = "Loss vs latent space dimension",
               xlabel = "latent space dimension z_dim",
               ylabel = "average loss after $epochs epochs")
    display(plt)
    return z_dims, avg_loss
end

# ===========================================================================
# Section: Fashion-MNIST, including a deeper model + 3-D latent (Cells 15–28)
# ===========================================================================
function run_fashionmnist()
    train_loader, Xte, yte = load_fashionmnist(batchsize = 100)

    # 2-hidden-layer VAE (Cells 20–23)
    model = device(VAE(2; hidden_dims = [400, 200]))
    opt   = setup(Adam(1f-3), model)
    train!(model, opt, train_loader, 50)
    display(generate_random_samples(model, 2))
    display(plot_latent_space(model, 3, 30))
    display(plot_latent_space_raw(model, Xte, yte; title = "FashionMNIST — 2 layers"))

    # 3-hidden-layer VAE, z_dim = 2 (Cell 25)
    model3 = device(VAE(2; hidden_dims = [400, 200, 100]))
    opt3   = setup(Adam(1f-3), model3)
    train!(model3, opt3, train_loader, 50)
    display(generate_random_samples(model3, 2))
    display(plot_latent_space_raw(model3, Xte, yte; title = "FashionMNIST — 3 layers"))

    # 3-hidden-layer VAE, z_dim = 3 -> 3-D latent scatter (Cells 26/28)
    model3d = device(VAE(3; hidden_dims = [400, 200, 100]))
    opt3d   = setup(Adam(1f-3), model3d)
    train!(model3d, opt3d, train_loader, 50)
    display(generate_random_samples(model3d, 3))
    display(plot_latent_space_raw3(model3d, Xte, yte; title = "FashionMNIST — 3-D latent"))

    return model, model3, model3d
end

# ===========================================================================
# Section: EuroSAT — 64x64 RGB (Cells 29–42)
# ===========================================================================
# NOTE: MLDatasets does not ship a EuroSAT loader. Provide your own `load_eurosat`
# returning Xtr :: (input_dim, N) Float32 in [0,1] where each column is a
# flattened 64x64x3 image (reshape(img_whc, :)), plus labels. input_dim = 12288.
# All RGB helpers below assume the channels-last (W, H, C) layout used here.

# Map a flat decoded vector -> (W, H, 3) array (Cell 36 generate_digit_out_rgb).
function generate_sample_out_rgb(model, z::AbstractVector, sample_size::Int)
    x̂ = cpu(decode(model, device(Float32.(z))))
    return reshape(x̂, sample_size, sample_size, 3)
end

# (W, H, 3) array -> Matrix{RGB} for display (Cell 32 tensor_to_img).
function to_rgb_image(whc::AbstractArray{<:Real,3})
    c = clamp.(whc, 0f0, 1f0)
    return RGB.(c[:, :, 1], c[:, :, 2], c[:, :, 3])
end

# Cell 36: 5x5 montage of RGB samples (notebook draws z ~ N(0, 10^2)).
function generate_random_samples_rgb(model, z_dim::Int, sample_size::Int; scale = 10.0)
    plts = map(1:25) do _
        z   = Float32(scale) .* randn(Float32, z_dim)
        img = to_rgb_image(generate_sample_out_rgb(model, z, sample_size))
        plot(img; axis = false, legend = false, yflip = true)
    end
    plot(plts...; layout = (5, 5), size = (500, 500))
end

# Cell 35: tile RGB decodes over a 2-D latent grid.
function plot_latent_space_rgb(model, scale::Real, n::Int; digit_size::Int = 64)
    figure = zeros(Float32, digit_size * n, digit_size * n, 3)
    grid   = range(-scale, scale; length = n)
    for (i, yi) in enumerate(grid), (j, xi) in enumerate(grid)
        tile = generate_sample_out_rgb(model, Float32[xi, yi], digit_size)
        figure[(i-1)*digit_size+1:i*digit_size, (j-1)*digit_size+1:j*digit_size, :] = tile
    end
    plot(to_rgb_image(figure); yflip = true, aspect_ratio = 1,
         title = "Latent space visualization", xlabel = "z[0]", ylabel = "z[1]")
end

function run_eurosat(load_eurosat; epochs::Int = 10)
    Xtr, ytr, Xte, yte = load_eurosat()            # Xtr,Xte :: (12288, N)
    input_dim = size(Xtr, 1)                        # 64*64*3 == 12288 (Cell 33)
    train_loader = DataLoader((Xtr, ytr); batchsize = 100, shuffle = true)

    # 5-hidden-layer VAE (Cell 37): VAE5(z_dim=3, hidden_dim=[6000,3000,1000,200,64])
    model5 = device(VAE(3; input_dim = input_dim, hidden_dims = [6000, 3000, 1000, 200, 64]))
    opt5   = setup(Adam(1f-3), model5)
    _, loss5 = train!(model5, opt5, train_loader, epochs)
    @info "VAE5 final loss" loss5

    # 3-hidden-layer VAE used for the RGB plots (Cells 39–42)
    model3 = device(VAE(3; input_dim = input_dim, hidden_dims = [4000, 400, 100]))
    opt3   = setup(Adam(1f-3), model3)
    train!(model3, opt3, train_loader, epochs)

    display(generate_random_samples_rgb(model3, 3, 64))
    display(plot_latent_space_rgb(model3, 3, 30; digit_size = 64))
    return model5, model3
end

# ---------------------------------------------------------------------------
# Entry point — uncomment the section you want to run.
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    # run_mnist_basic()
    # run_loss_vs_zdim()
    # run_fashionmnist()
    # run_eurosat(my_eurosat_loader)
end

run_mnist_basic()