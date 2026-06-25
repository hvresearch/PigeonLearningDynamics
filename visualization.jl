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

lats[1,1,1]
longs[1,1,1]

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

function combine_group(group)
    lats, lons = Union{Float64,Missing}[], Union{Float64,Missing}[]
    for (i, (tlat, tlon)) in enumerate(group)
        append!(lats, tlat); append!(lons, tlon)
        i < length(group) && (push!(lats, missing); push!(lons, missing))
    end
    return lats, lons
end

function build_trajectory_traces(trajectories; group_size=26)
    n_groups = cld(length(trajectories), group_size)
    # n_groups = size(trajectories,3)
    traces = Vector{GenericTrace}(undef, n_groups)
    for g in 1:n_groups
        lo, hi = (g - 1) * group_size + 1, min(g * group_size, length(trajectories))
        glats, glons = combine_group(trajectories[lo:hi])
        traces[g] = scattermapbox(
            lat=glats, lon=glons, mode="lines",
            line=attr(width=1.5, color=palette_color(g)),
            hoverinfo="none", showlegend=false,
        )
    end
    return traces
end

#---example trajectories plot---#
trajectory_traces = build_trajectory_traces(trajectories[1,:,:]; group_size=26)

layout = Layout(
    mapbox=attr(
        style="white-bg",
        center=attr(
            lat=mean([bbox.south, bbox.north]),
            lon=mean([bbox.west,  bbox.east]),
        ),
        zoom=13,
    ),
    showlegend=false,
    margin=attr(l=0, r=0, t=0, b=0),
    height=650,
)

display(PlotlyJS.plot([road_trace, hedge_trace, tree_trace, trajectory_traces...], layout))

#---plot trajectories---#

function plot_trajectory_traces(trajectories; group_size=26)
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
        showlegend=false,
        margin=attr(l=0, r=0, t=0, b=0),
        height=650,
    )
    display(PlotlyJS.plot([road_trace, hedge_trace, tree_trace, trajectory_traces...], layout))
end

plot_trajectory_traces(trajectories[1,:,1]; group_size=26)
