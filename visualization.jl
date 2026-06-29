using DelimitedFiles, CSV, DataFrames
using Plots
using PlotlyJS
using HTTP
using JSON3
using Statistics

pigeon_list = ["A_A35","A_A38","A_A45","A_D06","A_D09","A_D16","A_D83","A_D94","A_D96","B_A31","B_A34","B_D04","B_D12","B_D86","B_D90","B_D98","C_A39","C_A43","C_A47","C_D00","C_D01","C_D08","C_D10","C_D15","C_D88","C_D93"]

file_paths = ["data/R$(i)/$(pigeon_list[j])/$(pigeon_list[j])_R$(i)_0$(k).csv" for i in 1:3, j in 1:26, k in 1:6]

raw_data = [DataFrame(CSV.File(file_paths[i,j,k])) for i in 1:3, j in 1:26, k in 1:6]

times  = [raw_data[i,j,k]." Time"      for i in 1:3, j in 1:26, k in 1:6]
lats   = [raw_data[i,j,k]." Latitude"  for i in 1:3, j in 1:26, k in 1:6]
longs  = [raw_data[i,j,k]." Longitude" for i in 1:3, j in 1:26, k in 1:6]
alts   = [raw_data[i,j,k]." Altitude"  for i in 1:3, j in 1:26, k in 1:6]
speed  = [raw_data[i,j,k]." Speed"     for i in 1:3, j in 1:26, k in 1:6]

all_trajs = Plots.plot(title="all_trajectories", legend=false)
for i in 1:3, j in 1:26, k in 1:6
    Plots.plot!(all_trajs, longs[i,j,k], lats[i,j,k])
end
display(all_trajs)

# --- Your trajectory data (replace with your real trajectories) ---
trajectories = [
    (lats[i,j,k], longs[i,j,k]) for i in 1:3, j in 1:26, k in 1:6
]

function trajectories_bbox(trajectories; pad=0.01)
    all_lats = vcat([t[1] for t in trajectories]...)
    all_lons = vcat([t[2] for t in trajectories]...)
    south, north = minimum(all_lats) - pad, maximum(all_lats) + pad
    west,  east  = minimum(all_lons) - pad, maximum(all_lons) + pad
    return (south=south, west=west, north=north, east=east)
end

bbox = trajectories_bbox(trajectories)

const OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
const OSM_CACHE_FILE = joinpath(@__DIR__, "osm_cache.json")

function fetch_osm_features(bbox; max_retries=3, base_delay=10.0)
    query = """
    [out:json][timeout:90];
    (
      way["highway"]($(bbox.south),$(bbox.west),$(bbox.north),$(bbox.east));
      way["barrier"="hedge"]($(bbox.south),$(bbox.west),$(bbox.north),$(bbox.east));
      way["natural"="tree_row"]($(bbox.south),$(bbox.west),$(bbox.north),$(bbox.east));
    );
    out geom;
    """
    for (mi, url) in enumerate(OVERPASS_MIRRORS)
        for attempt in 1:max_retries
            try
                @info "Overpass request" mirror=mi attempt=attempt
                resp = HTTP.post(url, ["Content-Type" => "text/plain"], query;
                                 readtimeout=120, connect_timeout=30)
                data = JSON3.read(String(resp.body))
                open(OSM_CACHE_FILE, "w") do io; JSON3.write(io, data); end
                return data
            catch e
                is_transient = e isa HTTP.Exceptions.StatusError && e.status in (429, 502, 503, 504)
                is_transient || (mi == length(OVERPASS_MIRRORS) && attempt == max_retries) || rethrow()
                delay = base_delay * 2.0^(attempt - 1)
                @warn "Overpass failed, retrying" mirror=mi attempt=attempt delay=delay error=e
                sleep(delay)
            end
        end
    end
    error("All Overpass mirrors failed after retries")
end

function load_osm_features(bbox)
    if isfile(OSM_CACHE_FILE)
        @info "Loading OSM data from cache" file=OSM_CACHE_FILE
        return JSON3.read(read(OSM_CACHE_FILE, String))
    end
    return fetch_osm_features(bbox)
end

osm_data = load_osm_features(bbox)

function classify_features(osm_data)
    roads, hedges, treelines = Any[], Any[], Any[]
    for el in osm_data.elements
        haskey(el, :geometry) || continue
        coords = [(pt.lat, pt.lon) for pt in el.geometry]
        tags = get(el, :tags, nothing)
        tags === nothing && continue
        if get(tags, :natural, nothing) == "tree_row"
            push!(treelines, coords)
        elseif get(tags, :barrier, nothing) == "hedge"
            push!(hedges, coords)
        elseif haskey(tags, :highway)
            push!(roads, coords)
        end
    end
    return roads, hedges, treelines
end

roads, hedges, treelines = classify_features(osm_data)

function merge_lines(lines)
    lats = Union{Float64,Missing}[]
    lons = Union{Float64,Missing}[]
    for (i, line) in enumerate(lines)
        for (lat, lon) in line
            push!(lats, lat)
            push!(lons, lon)
        end
        i < length(lines) && (push!(lats, missing); push!(lons, missing))
    end
    return lats, lons
end

road_lats,  road_lons  = merge_lines(roads)
hedge_lats, hedge_lons = merge_lines(hedges)
tree_lats,  tree_lons  = merge_lines(treelines)

road_trace = scattermapbox(
    lat=road_lats, lon=road_lons, mode="lines",
    line=attr(width=1, color="lightgray"),
    hoverinfo="none", showlegend=false,
)
hedge_trace = scattermapbox(
    lat=hedge_lats, lon=hedge_lons, mode="lines",
    line=attr(width=2, color="green"),
    hoverinfo="none", showlegend=false,
)
tree_trace = scattermapbox(
    lat=tree_lats, lon=tree_lons, mode="lines",
    line=attr(width=2, color="darkgreen"),
    hoverinfo="none", showlegend=false,
)

const PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
]
palette_color(i) = PALETTE[mod1(i, length(PALETTE))]
palette_colors = Plots.palette(:jet,6)

function sample_trajectories(trajectories; sampling=10)
    sampled_trajectories = [ 
        (trajectories[i,j,k][1][1:sampling:end],trajectories[i,j,k][2][1:sampling:end])
        for i in axes(trajectories,1), j in axes(trajectories,2), k in axes(trajectories,3)
    ]
    return sampled_trajectories
end

function combine_group(group)
    lats, lons = Union{Float64,Missing}[], Union{Float64,Missing}[]
    for (i, (tlat, tlon)) in enumerate(group)
        append!(lats, tlat); append!(lons, tlon)
        i < length(group) && (push!(lats, missing); push!(lons, missing))
    end
    return lats, lons
end

function build_trajectory_traces(trajectories; group_size=26, group_names=nothing)
    n_groups = cld(length(trajectories), group_size)
    traces = Vector{GenericTrace}(undef, n_groups)
    for g in 1:n_groups
        lo, hi = (g - 1) * group_size + 1, min(g * group_size, length(trajectories))
        glats, glons = combine_group(trajectories[lo:hi])
        name = group_names !== nothing ? group_names[g] : "Group $g"
        color = palette_colors[mod(g-1, length(palette_colors))+1]
        traces[g] = scattermapbox(
            lat=glats, lon=glons, mode="lines",
            line=attr(width=1.5, color=color),
            name=name, legendgroup=name,
            showlegend=true, hoverinfo="none",
        )
    end
    return traces
end

#---plot trajectories---#

function plot_trajectory_traces(trajectories; group_size=26, save_html=nothing)
    trajectory_traces = build_trajectory_traces(trajectories; group_size=26)
    layout = Layout(
        mapbox=attr(
            style="white-bg",
            center=attr(
                lat=mean([bbox.south, bbox.north]),
                lon=mean([bbox.west,  bbox.east]),
            ),
            zoom=13,
        ),
        showlegend=true,
        margin=attr(l=0, r=0, t=0, b=0),
        height=800, width=1200,
    )
    p = PlotlyJS.plot([road_trace, hedge_trace, tree_trace, trajectory_traces...], layout)
    save_html !== nothing && PlotlyJS.savefig(p, save_html)
    display(p)
end

function score_flight(trajectory)
    m = (trajectory[1][end] - trajectory[1][1])/(trajectory[2][end] - trajectory[2][1])
    b = trajectory[1][1]
    score = mean([(m*trajectory[2][i] + b - trajectory[1][i])/sqrt(1+m^2) for i in axes(trajectory[2],1)])
    mean_lat, mean_long = mean(trajectory[1]), mean(trajectory[2])
    return score, mean_lat, mean_long
end

function get_scores(trajectories)
    scores, mean_lats, mean_longs = zeros(3,26,6), zeros(3,26,6), zeros(3,26,6)
    for i in 1:3, j in 1:26, k in 1:6
        scores[i,j,k], mean_lats[i,j,k], mean_longs[i,j,k] = score_flight(trajectories[i,j,k])
    end
    return scores, mean_lats, mean_longs
end

trajectories_s = sample_trajectories(trajectories; sampling=10)
plot_trajectory_traces(trajectories_s[3,:,1:6]; group_size=26, save_html="PigeonLearningDynamics/vis-R3-ts_10.html")

scores, mean_lats, mean_longs = get_scores(trajectories_s)

function combine_scores(scores;num_runs=2)
    run_dims = Int(size(scores,3)/num_runs)
    scores_combined = zeros(size(scores,1),num_runs*size(scores,2),run_dims)
    for i in 1:3
        for j in 1:run_dims
            scores_combined[i,:,j] .= reshape(scores[i,:,num_runs*(j-1)+1:num_runs*j],num_runs*size(scores,2))
        end
    end
    return scores_combined
end

function plot_clusters(scores, mean_lats, mean_longs, trajectories)
    score_plots = []
    mean_plots = []
    mean_tplots = []
    for i in 1:3
        binmin, binmax = extrema(scores[i,:,3])
        p = Plots.histogram(title="site = $(i)")
        bins = binmin:(binmax-binmin)/20:binmax
        for j in axes(scores,3)
            Plots.histogram!(scores[i,:,j],label="iteration $(j)",color=palette_colors[j], bins=bins)
        end
        push!(score_plots,p)
    end

    for i in 1:3
        q = Plots.plot(title="site = $(i)")
        for j in 1:6
            q = Plots.plot(title="site = $(i)")
            for jj in 1:j-1
                Plots.scatter!(q, mean_longs[i,:,jj], mean_lats[i,:,jj], label="run $(jj)", color=:gray, ma=0.3, msw=0)
            end
            Plots.scatter!(q, mean_longs[i,:,j], mean_lats[i,:,j], label="run $(j)", color=palette_colors[j], ma=0.8, msw=0)
            push!(mean_plots, q)
        end
    end

    mean_traj_lats = zeros(size(mean_lats,1),size(mean_lats,3))
    mean_traj_longs = zeros(size(mean_longs,1),size(mean_longs,3))
    std_traj_lats = zeros(size(mean_lats,1),size(mean_lats,3))
    std_traj_longs = zeros(size(mean_longs,1),size(mean_longs,3))
    for i in 1:3
        optimal_lat = 0.5 * (trajectories[i,1,1][1][1] + trajectories[i,1,1][1][end])
        optimal_long = 0.5 * (trajectories[i,1,1][2][1] + trajectories[i,1,1][2][end])

        r = Plots.plot(title="site = $(i)")
        for j in 1:6
            mean_traj_lats[i,j] = mean(mean_lats[i,:,j])
            mean_traj_longs[i,j] = mean(mean_longs[i,:,j])
            std_traj_lats[i,j] = std(mean_lats[i,:,j])
            std_traj_longs[i,j] = std(mean_longs[i,:,j])
            Plots.scatter!(
                (mean_traj_longs[i,j],mean_traj_lats[i,j]),
                xerror=std_traj_longs[i,j], yerror=std_traj_lats[i,j],
                label="run $(j)",color=palette_colors[j], msw=0.2
            )
        end
        Plots.scatter!((optimal_long,optimal_lat),label="optimal",color=:gray,ma=0.5)
        push!(mean_tplots,r)
    end
    return score_plots, mean_plots, mean_tplots
end

FIGPATH = "PigeonLearningDynamics/figures/"

scores_combined = combine_scores(scores;num_runs=2)
scoreplots, meanplots, mean_tplots = plot_clusters(scores_combined, mean_lats, mean_longs, trajectories_s)
Plots.plot(scoreplots..., layout=(2,2), size=(1000,1000), plot_title="trajectory scores")
Plots.savefig(FIGPATH*"flight_scores-i.svg")
Plots.plot(meanplots..., layout=(3,6), size=(3000,1500), plot_title="mean trajectory location")
Plots.savefig(FIGPATH*"flight_means-i.svg")
Plots.plot(mean_tplots...,layout=(2,2),size=(1000,1000), plot_title="mean trajectories vs run")
Plots.savefig(FIGPATH*"flight_means-i-p.svg")

function score_means(scores)
    mean_scores = [mean(scores[i,:,k]) for i in axes(scores,1), k in axes(scores,3)]
    return mean_scores
end

function plot_score_means(mean_scores)
    plots_mean_scores = []
    for i in axes(mean_scores,1)
        p = Plots.plot(title="site $(i)")
        Plots.plot!(p,mean_scores[i,:],xlabel="run #",ylabel="mean score",label=:none)
        push!(plots_mean_scores,p)
    end
    return plots_mean_scores
end

function nearest_roads(trajectories, roads)
    road_indices = Array{Vector{Int}}(undef, size(trajectories))
    total_distances = Array{Float64}(undef, size(trajectories))

    for idx in CartesianIndices(trajectories)
        tlats, tlons = trajectories[idx]
        n_points = length(tlats)
        indices = Vector{Int}(undef, n_points)
        total_dist = 0.0

        for p in 1:n_points
            plat, plon = tlats[p], tlons[p]
            min_dist_sq = Inf
            min_road_idx = 0
            for r in eachindex(roads)
                for (rlat, rlon) in roads[r]
                    d = (plat - rlat)^2 + (plon - rlon)^2
                    if d < min_dist_sq
                        min_dist_sq = d
                        min_road_idx = r
                    end
                end
            end
            indices[p] = min_road_idx
            total_dist += sqrt(min_dist_sq)
        end

        road_indices[idx] = indices
        total_distances[idx] = total_dist
    end

    return road_indices, total_distances
end

road_indices, road_distances = nearest_roads(trajectories_s, roads)