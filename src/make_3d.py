"""3D flight-path viewer (deck.gl + MapLibre) for the 2011 Flack data.

Paths drawn in 3D (lon, lat, altitude), colored first->last release. Adds 3D
terrain, and navigation-cue overlays: the sun's true direction (azimuth +
elevation, per release) and the geomagnetic field vector (declination +
66.7 deg inclination). Pitch / orbit / rotate with mouse, buttons, or keys.

Output: flight_paths_3d.html  (self-contained; deck.gl + MapLibre from CDN)
"""

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from geo_simplify import keep_indices
from sun_calc import sun_position

DATA = Path("data")
DATA_2021 = Path("data_2021.json")
EPS_M, CAP = 4.0, 1500
GEOMAG = {"dec": -1.97, "inc": 66.73, "F": 48.7}  # IGRF, Oxford, epoch 2011.4
GEOMAG_2021 = {"dec": -0.35, "inc": 66.45, "F": 49.2}  # IGRF, Oxford, epoch 2018.5


def load_flight(csv_path):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for c in ("Latitude", "Longitude", "Altitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude", "Altitude"])
    df = df[df["Latitude"].between(45, 60) & df["Longitude"].between(-10, 5)
            & df["Altitude"].between(-50, 2000)]
    if len(df) < 2:
        return None
    idx = keep_indices(df[["Latitude", "Longitude"]].to_numpy(), EPS_M, CAP)
    sub = df.iloc[idx]
    path = [[round(lo, 5), round(la, 5), round(al, 1)]
            for la, lo, al in zip(sub["Latitude"], sub["Longitude"],
                                  sub["Altitude"])]
    # sun position at release (first fix); local BST -> UTC = local - 1h
    r0 = df.iloc[0]
    y, mo, d = (int(x) for x in str(r0["Date"]).strip().split("/"))
    hh, mm, ss = (int(x) for x in str(r0["Time"]).strip().split(":"))
    utc = dt.datetime(y, mo, d, hh, mm, ss) - dt.timedelta(hours=1)
    az, el = sun_position(float(r0["Latitude"]), float(r0["Longitude"]), utc)
    return {"path": path, "sunAz": az, "sunEl": el,
            "date": f"{y:04d}-{mo:02d}-{d:02d}", "time": f"{hh:02d}:{mm:02d}"}


def load_2021(json_path):
    """2021 collective-exploration data -> viewer units (Solo & Pair only).

    Coords are 2D [lat, lon]; we lay them flat (altitude 0). No per-release
    timestamps, so sun cues are unavailable for this dataset.
    """
    raw = json.loads(Path(json_path).read_text())
    keep = {"Solo", "Pair"}
    units = []
    for u in raw["units"]:
        if u["group"] not in keep:
            continue
        flights = []
        for fl in u["flights"]:
            coords = fl.get("coords", [])
            if len(coords) < 2:
                continue
            path = [[round(c[1], 5), round(c[0], 5), 0.0] for c in coords]
            flights.append({"path": path, "n": fl.get("n", "")})
        if flights:
            units.append({"id": u["id"], "group": u["group"], "flights": flights})
    return units


def load_2011():
    units, alt_all = [], []
    for bird_dir in sorted(p for p in DATA.glob("R*/*") if p.is_dir()):
        route, bird_id = bird_dir.parts[1], bird_dir.parts[2]
        flights = []
        for csv in sorted(bird_dir.glob("*.csv")):
            f = load_flight(csv)
            if f:
                f["n"] = csv.stem.split("_")[-1]
                flights.append(f)
                alt_all += [p[2] for p in f["path"]]
        if flights:
            units.append({"id": bird_id, "group": route, "flights": flights})
    amin, amax = (min(alt_all), max(alt_all)) if alt_all else (0, 300)
    print(f"2011: {len(units)} birds, {sum(len(u['flights']) for u in units)} "
          f"flights, altitude {amin:.0f}-{amax:.0f} m")
    return units


def main():
    units_2011 = load_2011()
    units_2021 = load_2021(DATA_2021)
    n_solo = sum(u["group"] == "Solo" for u in units_2021)
    n_pair = sum(u["group"] == "Pair" for u in units_2021)
    print(f"2021: {len(units_2021)} birds ({n_solo} Solo, {n_pair} Pair), "
          f"{sum(len(u['flights']) for u in units_2021)} flights, flat (no alt)")

    datasets = {
        "2011": {"units": units_2011, "geomag": GEOMAG,
                 "label": "Flack et al. (2014) · solo route learning",
                 "hasAlt": True, "hasSun": True},
        "2021": {"units": units_2021, "geomag": GEOMAG_2021,
                 "label": "Sasaki/Valentini et al. (2021) · collective exploration",
                 "hasAlt": False, "hasSun": False},
    }
    payload = json.dumps({"datasets": datasets}, separators=(",", ":"))
    Path("viewers/flight_paths_3d.html").write_text(HTML.replace("__DATA__", payload))
    print(f"Saved flight_paths_3d.html "
          f"({Path('viewers/flight_paths_3d.html').stat().st_size/1e6:.1f} MB)")


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pigeon flight paths in 3D — Flack et al. (2014)</title>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet"/>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/deck.gl@9.0.36/dist.min.js"></script>
<style>
  :root{--accent:#6ea8fe;--bg:rgba(18,22,33,.82);--line:rgba(255,255,255,.12)}
  html,body{margin:0;height:100%;overflow:hidden;
    font-family:'Inter',system-ui,-apple-system,sans-serif;color:#e8ecf3}
  #map{position:absolute;inset:0;background:#0b0e16}
  #panel{position:absolute;top:16px;left:16px;z-index:10;width:284px;
    background:var(--bg);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
    border:1px solid var(--line);border-radius:14px;padding:16px 18px;
    box-shadow:0 8px 30px rgba(0,0,0,.45)}
  #panel h3{margin:0 0 2px;font-size:16px;font-weight:600;letter-spacing:.2px}
  .sub{color:#9aa6bd;font-size:11.5px;margin-bottom:12px}
  select{width:100%;padding:7px 9px;font-size:13px;background:#1c2233;color:#e8ecf3;
    border:1px solid var(--line);border-radius:8px;margin-bottom:8px}
  .row{display:flex;gap:6px;margin-bottom:8px}
  .row button{flex:1;padding:7px 4px;font-size:12.5px;cursor:pointer;color:#cdd6e6;
    border:1px solid var(--line);background:#1c2233;border-radius:8px;
    transition:background .15s,color .15s,border-color .15s}
  .row button:hover{background:#28304a;color:#fff}
  .row button.on{background:var(--accent);color:#0b0e16;border-color:var(--accent);
    font-weight:600}
  .row.disabled{opacity:.4;pointer-events:none;filter:grayscale(1)}
  .toggle.disabled{opacity:.4;pointer-events:none}
  #info{font-size:12px;margin:2px 0 10px;color:#c7d0e0}
  #info b{color:#fff}
  .lbl{font-size:11px;color:#9aa6bd;text-transform:uppercase;letter-spacing:.6px;
    margin:12px 0 5px;font-weight:600}
  .toggle{display:flex;align-items:center;gap:8px;font-size:12.5px;margin:5px 0;
    cursor:pointer;color:#cdd6e6}
  .toggle input{accent-color:var(--accent)}
  input[type=range]{width:100%;accent-color:var(--accent)}
  .legend{margin-top:6px;font-size:11.5px;color:#aeb8cc}
  .bar{height:10px;border-radius:5px;margin:5px 0;background:linear-gradient(
    to right,hsl(240,80%,62%),hsl(190,80%,55%),hsl(135,68%,52%),
    hsl(55,90%,58%),hsl(8,85%,60%))}
  .barlbl{display:flex;justify-content:space-between;color:#8d97ac;font-size:10.5px}
  .navinfo{font-size:11px;color:#aeb8cc;margin-top:4px;line-height:1.5}
  .sw{display:inline-block;width:9px;height:9px;border-radius:2px;vertical-align:0}
  .hint{font-size:10.5px;color:#788299;margin-top:12px;line-height:1.5}
  .divider{height:1px;background:var(--line);margin:12px -18px}
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <h3>Pigeon flight paths · 3D</h3>
  <div class="sub" id="subtitle">Flack et al. (2014) · altitude, sun &amp; magnetic field</div>

  <div class="lbl">Dataset</div>
  <div class="row">
    <button id="ds2011" class="on">2011 · older</button>
    <button id="ds2021">2021 · newer</button>
  </div>

  <div class="lbl" id="condLbl">Condition <span style="text-transform:none;letter-spacing:0;color:#788299">(2021 only)</span></div>
  <div class="row" id="condRow">
    <button id="condSolo" class="on">1 bird · Solo</button>
    <button id="condPair">2 birds · Pair</button>
  </div>

  <select id="birdSel"></select>
  <div class="row"><button id="prev">← Prev</button><button id="next">Next →</button></div>
  <div id="info"></div>

  <div class="lbl">Flight <span id="flightInfo" style="text-transform:none;letter-spacing:0;color:#9aa6bd;font-weight:400"></span></div>
  <input type="range" id="flightSlider" min="0" max="1" value="0">

  <div class="divider"></div>
  <div class="lbl">👁 Field-of-view animation</div>
  <div class="row">
    <button id="play">▶ Play</button><button id="animStop">Reset</button>
  </div>
  <input type="range" id="animSlider" min="0" max="1000" value="0">
  <div class="lbl">Sight range · <span id="sightVal">5</span> km</div>
  <input type="range" id="sightCap" min="1" max="40" value="5">
  <div class="hint" style="margin-top:4px">~2–5 km = landmark recognition (acuity-limited).
    Push higher toward the geometric horizon (≈36 km at 100 m).</div>
  <div class="lbl">Speed · <span id="spdVal">1.0</span>×</div>
  <input type="range" id="animSpeed" min="25" max="300" value="100">
  <div class="navinfo" id="animInfo"></div>

  <div class="lbl">Camera</div>
  <div class="row">
    <button id="rotL">⟲</button><button id="orbit">Orbit</button><button id="rotR">⟳</button>
  </div>
  <div class="row">
    <button id="pitchUp">Tilt ↑</button><button id="pitchDn">Tilt ↓</button>
    <button id="reset">Reset</button><button id="terrain" class="on">Terrain</button>
  </div>

  <div class="lbl">Vertical exaggeration · <span id="exagVal">8</span>×</div>
  <input type="range" id="exag" min="1" max="30" value="8">

  <div class="divider"></div>
  <div class="lbl">Navigation cues</div>
  <label class="toggle"><input type="checkbox" id="sunTog"> ☀ Sun direction (per release)</label>
  <label class="toggle"><input type="checkbox" id="geoTog"> 🧭 Geomagnetic field vector</label>
  <div class="navinfo" id="navInfo"></div>

  <div class="legend">
    <div class="lbl" style="margin-top:10px">Release order</div>
    <div class="bar"></div>
    <div class="barlbl"><span>first</span><span>last</span></div>
  </div>
  <div class="hint">Right-drag / ctrl-drag rotates &amp; tilts · left-drag pans ·
    scroll zooms · keys Q/E rotate, R/F tilt, ←/→ change bird</div>
</div>
<script>
const DATA = __DATA__;
const DS = DATA.datasets;
let dsKey = '2011', cond = 'Solo', idx = 0, exag = 8;
let flightSel = -1, fitKey = '';  // flightSel: -1 = all flights, else flight index
// field-of-view animation state
let animMode=false, playing=false, animT=0, animFlight=0, animRAF=null,
    animPrevTerrain=true, lastTs=0;
const R_EARTH=6371000;
function haversine(a,b){ const r=Math.PI/180;
  const dLat=(b[1]-a[1])*r, dLon=(b[0]-a[0])*r, la1=a[1]*r, la2=b[1]*r;
  const h=Math.sin(dLat/2)**2+Math.cos(la1)*Math.cos(la2)*Math.sin(dLon/2)**2;
  return 2*R_EARTH*Math.asin(Math.min(1,Math.sqrt(h))); }
function flightGeom(f){ const pts=f.path, cum=[0];
  for(let i=1;i<pts.length;i++) cum.push(cum[i-1]+haversine(pts[i-1],pts[i]));
  return {pts,cum,total:cum[cum.length-1]||1}; }
function sightRadius(altM){
  const cap=(+document.getElementById('sightCap').value)*1000;
  if(dsKey!=='2011') return cap;                 // no altitude -> constant cap
  const h=Math.max(altM||0,20);                   // floor so disc isn't tiny near ground
  return Math.min(Math.sqrt(2*R_EARTH*h+h*h), cap);
}
function sampleAt(geom,t){                         // t in [0,1] -> position along path
  const target=t*geom.total, cum=geom.cum, pts=geom.pts;
  let i=1; while(i<cum.length && cum[i]<target) i++;
  if(i>=pts.length){ const p=pts[pts.length-1]; return {pos:p,i:pts.length-1}; }
  const seg=(cum[i]-cum[i-1])||1, f=(target-cum[i-1])/seg, a=pts[i-1], b=pts[i];
  return {pos:[a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f], i};
}
const curDS = () => DS[dsKey];
function activeUnits(){
  const us = curDS().units;
  return dsKey === '2021' ? us.filter(u => u.group === cond) : us;
}

const map = new maplibregl.Map({
  container:'map',
  style:'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
  center:[-1.28,51.86], zoom:10.5, pitch:58, bearing:-17, antialias:true
});
map.addControl(new maplibregl.NavigationControl({visualizePitch:true}));
const overlay = new deck.MapboxOverlay({interleaved:true, layers:[]});
map.addControl(overlay);

let terrainOn = true;
function setupTerrain(){
  if(map.getSource('dem')) return;
  map.addSource('dem',{type:'raster-dem',tileSize:256,maxzoom:14,encoding:'terrarium',
    tiles:['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png']});
  map.addLayer({id:'hillshade',type:'hillshade',source:'dem',
    paint:{'hillshade-exaggeration':0.5,
      'hillshade-shadow-color':'#05070d','hillshade-highlight-color':'#3a4a5c'}});
  try{ map.setSky({'sky-color':'#1b2740','horizon-color':'#3a4a66',
    'fog-color':'#0b0e16','sky-horizon-blend':0.7,'horizon-fog-blend':0.6}); }catch(e){}
  applyTerrain();
}
function applyTerrain(){
  map.setTerrain(terrainOn?{source:'dem',exaggeration:exag}:null);
  const b=document.getElementById('terrain');
  b.classList.toggle('on',terrainOn); b.textContent=terrainOn?'Terrain':'Flat';
  if(map.getLayer('hillshade'))
    map.setLayoutProperty('hillshade','visibility',terrainOn?'visible':'none');
}

function grad(t){
  const h=240*(1-t)/360,s=0.8,l=0.58,a=s*Math.min(l,1-l);
  const k=n=>(n+h*12)%12, f=n=>l-a*Math.max(-1,Math.min(k(n)-3,9-k(n),1));
  return [Math.round(255*f(0)),Math.round(255*f(8)),Math.round(255*f(4))];
}
const D2R=Math.PI/180;
function offset(lon,lat,eastM,northM){
  return [lon+eastM/(111320*Math.cos(lat*D2R)), lat+northM/110540];
}
// direction (az from N clockwise, elev up) -> [east,north,up] components * len
function vec(azDeg,elDeg,len){
  const a=azDeg*D2R, e=elDeg*D2R, h=len*Math.cos(e);
  return [h*Math.sin(a), h*Math.cos(a), len*Math.sin(e)];
}

function render(){
  const units=activeUnits(), GEO=curDS().geomag, hasSun=curDS().hasSun;
  if(!units.length){ overlay.setProps({layers:[]});
    document.getElementById('info').innerHTML='No birds in this condition.'; return; }
  idx=Math.max(0,Math.min(idx,units.length-1));
  const u=units[idx], N=u.flights.length;
  if(animMode){ renderAnim(u,N); return; }
  // bird centre + max altitude (for cue anchoring)
  let mnx=180,mny=90,mxx=-180,mxy=-90,maxAlt=0;
  u.flights.forEach(f=>f.path.forEach(p=>{mnx=Math.min(mnx,p[0]);mxx=Math.max(mxx,p[0]);
    mny=Math.min(mny,p[1]);mxy=Math.max(mxy,p[1]);maxAlt=Math.max(maxAlt,p[2]);}));
  const cx=(mnx+mxx)/2, cy=(mny+mxy)/2, anchor=maxAlt*exag+600;

  const paths=u.flights.map((f,k)=>({path:f.path.map(p=>[p[0],p[1],p[2]*exag]),
    color:grad(N>1?k/(N-1):0), sel:(flightSel<0||k===flightSel)}));
  const uT={getPath:exag,getColor:flightSel,getWidth:flightSel};
  const layers=[
    new deck.PathLayer({id:'glow',data:paths,getPath:d=>d.path,
      getColor:d=>[...d.color,d.sel?70:0],widthUnits:'pixels',
      getWidth:d=>d.sel?7:0,widthMinPixels:6,
      capRounded:true,jointRounded:true,parameters:{depthTest:true},
      updateTriggers:uT}),
    new deck.PathLayer({id:'paths',data:paths,getPath:d=>d.path,
      getColor:d=>d.sel?[...d.color,255]:[...d.color,42],widthUnits:'pixels',
      getWidth:d=>d.sel?2.6:1.1,widthMinPixels:1,
      capRounded:true,jointRounded:true,pickable:true,parameters:{depthTest:true},
      updateTriggers:uT})
  ];
  const info=[];

  // ☀ sun: a disc per release in its true (az,el) direction, color-matched
  if(hasSun && document.getElementById('sunTog').checked){
    const rays=[],discs=[]; const L=5200;
    u.flights.forEach((f,k)=>{
      if(f.sunEl<=0 || (flightSel>=0 && k!==flightSel)) return;
      const v=vec(f.sunAz,f.sunEl,L), tip=offset(cx,cy,v[0],v[1]);
      const tp=[tip[0],tip[1],anchor+v[2]], col=grad(N>1?k/(N-1):0);
      rays.push({s:[cx,cy,anchor],t:tp,c:col}); discs.push({p:tp,c:col});
    });
    layers.push(new deck.LineLayer({id:'sunray',data:rays,getSourcePosition:d=>d.s,
      getTargetPosition:d=>d.t,getColor:d=>[...d.c,120],getWidth:1.5}));
    layers.push(new deck.ScatterplotLayer({id:'sunglow',data:discs,getPosition:d=>d.p,
      getRadius:15,radiusUnits:'pixels',getFillColor:d=>[...d.c,60]}));
    layers.push(new deck.ScatterplotLayer({id:'sundisc',data:discs,getPosition:d=>d.p,
      getRadius:7,radiusUnits:'pixels',getFillColor:d=>d.c,stroked:true,
      getLineColor:[255,255,255,200],lineWidthMinPixels:1}));
    info.push('☀ each disc = sun position at one release');
  }

  // 🧭 geomagnetic field: grid of vectors, true declination + inclination dip
  if(document.getElementById('geoTog').checked){
    const shafts=[],tips=[],n=3,L=Math.max(900,maxAlt*exag*0.35);
    for(let i=0;i<n;i++)for(let j=0;j<n;j++){
      const lo=mnx+(mxx-mnx)*(i+0.5)/n, la=mny+(mxy-mny)*(j+0.5)/n;
      const v=vec(GEO.dec,-GEO.inc,L), tip=offset(lo,la,v[0],v[1]);
      const s=[lo,la,anchor], t=[tip[0],tip[1],anchor+v[2]];
      shafts.push({s,t}); tips.push({p:t});
    }
    layers.push(new deck.LineLayer({id:'geo',data:shafts,getSourcePosition:d=>d.s,
      getTargetPosition:d=>d.t,getColor:[230,68,46,220],getWidth:2.5}));
    layers.push(new deck.ScatterplotLayer({id:'geotip',data:tips,getPosition:d=>d.p,
      getRadius:4,radiusUnits:'pixels',getFillColor:[230,68,46,255]}));
    info.push('<span class="sw" style="background:#e6442e"></span> '+
      'Mag N: dec '+GEO.dec+'°, inc '+GEO.inc+'° (dip), '+GEO.F+' µT');
  }

  overlay.setProps({layers});
  document.getElementById('navInfo').innerHTML=info.join('<br>');
  document.getElementById('info').innerHTML=
    '<b>'+u.id+'</b> · '+u.group+' · '+N+' releases<br>bird '+(idx+1)+' of '+units.length;
  document.getElementById('birdSel').value=idx;

  // flight slider: 0 = all, 1..N = single release
  const fSlider=document.getElementById('flightSlider');
  fSlider.max=N; if(+fSlider.value>N) fSlider.value=0;
  let fTxt='all '+N+' releases';
  if(flightSel>=0){ const f=u.flights[flightSel];
    fTxt='release '+(f.n||(flightSel+1))+' of '+N+(f.date?' · '+f.date:''); }
  document.getElementById('flightInfo').textContent='· '+fTxt;

  // only re-frame the camera when the bird/dataset/condition changes, not on scrub
  const fk=dsKey+'|'+cond+'|'+idx;
  if(fk!==fitKey){ fitKey=fk;
    map.fitBounds([[mnx,mny],[mxx,mxy]],
      {padding:90,pitch:map.getPitch(),bearing:map.getBearing(),duration:600}); }
}
function renderAnim(u,N){
  const fi=Math.max(0,Math.min(animFlight,N-1)), f=u.flights[fi];
  const geom=flightGeom(f), s=sampleAt(geom,animT), head=[s.pos[0],s.pos[1]];
  const rad=sightRadius(s.pos[2]), col=grad(N>1?fi/(N-1):0);
  const full=f.path.map(p=>[p[0],p[1]]);
  const traveled=geom.pts.slice(0,s.i).map(p=>[p[0],p[1]]).concat([head]);
  const start=[f.path[0][0],f.path[0][1]], home=[f.path[N0(f)][0],f.path[N0(f)][1]];
  overlay.setProps({layers:[
    new deck.PathLayer({id:'ghost',data:[{path:full}],getPath:d=>d.path,
      getColor:[150,160,180,55],widthUnits:'pixels',getWidth:1.4,widthMinPixels:1}),
    new deck.LineLayer({id:'crow',data:[{s:start,t:home}],getSourcePosition:d=>d.s,
      getTargetPosition:d=>d.t,getColor:[255,205,90,190],getWidth:1.8,widthMinPixels:1.5}),
    new deck.ScatterplotLayer({id:'fov',data:[{p:head}],getPosition:d=>d.p,
      getRadius:rad,radiusUnits:'meters',getFillColor:[110,168,254,40],
      stroked:true,getLineColor:[110,168,254,160],lineWidthMinPixels:1.5}),
    new deck.PathLayer({id:'trail',data:[{path:traveled}],getPath:d=>d.path,
      getColor:[...col,255],widthUnits:'pixels',getWidth:3,widthMinPixels:2.5,
      capRounded:true,jointRounded:true}),
    new deck.ScatterplotLayer({id:'rel',data:[{p:start}],getPosition:d=>d.p,
      getRadius:6,radiusUnits:'pixels',getFillColor:[60,220,120,255],stroked:true,
      getLineColor:[255,255,255,220],lineWidthMinPixels:1}),
    new deck.ScatterplotLayer({id:'home',data:[{p:home}],getPosition:d=>d.p,
      getRadius:7,radiusUnits:'pixels',getFillColor:[230,68,46,255],stroked:true,
      getLineColor:[255,255,255,220],lineWidthMinPixels:1}),
    new deck.ScatterplotLayer({id:'bird',data:[{p:head}],getPosition:d=>d.p,
      getRadius:5,radiusUnits:'pixels',getFillColor:[255,255,255,255],stroked:true,
      getLineColor:col,lineWidthMinPixels:2})
  ]});
  const km=(rad/1000).toFixed(1), prog=Math.round(animT*100);
  const bee=(haversine(start,home)/1000).toFixed(1);
  const altTxt=dsKey==='2011'?(' · alt '+Math.round(s.pos[2])+' m'):'';
  document.getElementById('animInfo').innerHTML=
    '<span class="sw" style="background:#3cdc78"></span> release '+
    '<span class="sw" style="background:#e6442e;margin-left:6px"></span> home '+
    '<span class="sw" style="background:#ffcd5a;margin-left:6px"></span> beeline<br>'+
    prog+'% home · sight '+km+' km'+altTxt+'<br>beeline (crow) '+bee+' km';
  document.getElementById('info').innerHTML=
    '<b>'+u.id+'</b> · '+u.group+' · animating release '+(f.n||(fi+1));
  document.getElementById('animSlider').value=Math.round(animT*1000);
  const ak='anim|'+dsKey+'|'+cond+'|'+idx+'|'+fi;
  if(fitKey!==ak){ fitKey=ak;
    let mnx=180,mny=90,mxx=-180,mxy=-90;
    full.forEach(p=>{mnx=Math.min(mnx,p[0]);mxx=Math.max(mxx,p[0]);
      mny=Math.min(mny,p[1]);mxy=Math.max(mxy,p[1]);});
    map.fitBounds([[mnx,mny],[mxx,mxy]],{padding:80,pitch:0,bearing:0,duration:700}); }
}
function N0(f){ return f.path.length-1; }

function tick(ts){
  if(!playing) return;
  if(!lastTs) lastTs=ts;
  const dt=(ts-lastTs)/1000; lastTs=ts;
  const spd=(+document.getElementById('animSpeed').value)/100, dur=12/spd;
  animT+=dt/dur;
  if(animT>=1){ animT=1; pauseAnim(); }
  render();
  if(playing) animRAF=requestAnimationFrame(tick);
}
function startAnim(){
  animMode=true; animFlight=(flightSel>=0?flightSel:0); flightSel=animFlight;
  document.getElementById('flightSlider').value=animFlight+1;
  animPrevTerrain=terrainOn; terrainOn=false; applyTerrain();
  if(animT>=1) animT=0;
  playing=true; lastTs=0;
  const b=document.getElementById('play'); b.textContent='⏸ Pause'; b.classList.add('on');
  animRAF=requestAnimationFrame(tick);
}
function pauseAnim(){ playing=false; if(animRAF)cancelAnimationFrame(animRAF);
  const b=document.getElementById('play'); b.textContent='▶ Play'; b.classList.remove('on'); }
function stopAnim(){ pauseAnim(); animMode=false; animT=0;
  document.getElementById('animSlider').value=0;
  terrainOn=(dsKey==='2011' && animPrevTerrain); applyTerrain();
  if(terrainOn) map.setTerrain({source:'dem',exaggeration:exag});
  fitKey=''; render(); }

function go(i){ const n=activeUnits().length; if(!n){render();return;}
  if(animMode){ pauseAnim(); animMode=false; animT=0;
    document.getElementById('animSlider').value=0;
    terrainOn=(dsKey==='2011'); applyTerrain(); }
  idx=(i+n)%n; flightSel=-1; document.getElementById('flightSlider').value=0; render(); }

const sel=document.getElementById('birdSel');
function buildSelect(){
  sel.innerHTML='';
  activeUnits().forEach((u,i)=>{const o=document.createElement('option');
    o.value=i;o.text=u.id+'  ('+u.group+', '+u.flights.length+')';sel.add(o);});
}
sel.onchange=e=>go(+e.target.value);
document.getElementById('flightSlider').oninput=e=>{
  const v=+e.target.value; flightSel=(v===0?-1:v-1);
  if(animMode){ animFlight=Math.max(0,flightSel); animT=0; }
  render(); };
document.getElementById('play').onclick=()=>{ playing?pauseAnim():startAnim(); };
document.getElementById('animStop').onclick=stopAnim;
document.getElementById('animSlider').oninput=e=>{
  if(!animMode){ animMode=true; animFlight=(flightSel>=0?flightSel:0); flightSel=animFlight;
    document.getElementById('flightSlider').value=animFlight+1;
    animPrevTerrain=terrainOn; terrainOn=false; applyTerrain(); }
  pauseAnim(); animT=(+e.target.value)/1000; render(); };
document.getElementById('sightCap').oninput=e=>{
  document.getElementById('sightVal').textContent=e.target.value;
  if(animMode) render(); };
document.getElementById('animSpeed').oninput=e=>{
  document.getElementById('spdVal').textContent=((+e.target.value)/100).toFixed(2); };

// --- dataset (2011 / 2021) + condition (Solo / Pair) toggles ---
function syncToggles(){
  document.getElementById('ds2011').classList.toggle('on',dsKey==='2011');
  document.getElementById('ds2021').classList.toggle('on',dsKey==='2021');
  document.getElementById('condSolo').classList.toggle('on',cond==='Solo');
  document.getElementById('condPair').classList.toggle('on',cond==='Pair');
  // condition only meaningful for 2021
  document.getElementById('condRow').classList.toggle('disabled',dsKey!=='2021');
  // sun cue needs per-release timestamps (2011 only)
  const sunTog=document.getElementById('sunTog');
  sunTog.disabled=!curDS().hasSun;
  if(!curDS().hasSun) sunTog.checked=false;
  sunTog.closest('.toggle').classList.toggle('disabled',!curDS().hasSun);
  document.getElementById('subtitle').textContent=curDS().label;
}
function switchDataset(k){
  if(dsKey===k) return;
  dsKey=k; idx=0;
  // 2021 paths are flat (no altitude) -> drop terrain so they stay visible
  terrainOn=(dsKey==='2011'); applyTerrain();
  if(terrainOn) map.setTerrain({source:'dem',exaggeration:exag});
  syncToggles(); buildSelect(); go(0);
}
function switchCond(c){
  if(cond===c || dsKey!=='2021') return;
  cond=c; idx=0; syncToggles(); buildSelect(); go(0);
}
document.getElementById('ds2011').onclick=()=>switchDataset('2011');
document.getElementById('ds2021').onclick=()=>switchDataset('2021');
document.getElementById('condSolo').onclick=()=>switchCond('Solo');
document.getElementById('condPair').onclick=()=>switchCond('Pair');
document.getElementById('prev').onclick=()=>go(idx-1);
document.getElementById('next').onclick=()=>go(idx+1);
document.getElementById('exag').oninput=e=>{exag=+e.target.value;
  document.getElementById('exagVal').textContent=exag; render();
  if(terrainOn) map.setTerrain({source:'dem',exaggeration:exag});};
document.getElementById('terrain').onclick=()=>{terrainOn=!terrainOn; applyTerrain();};
document.getElementById('sunTog').onchange=render;
document.getElementById('geoTog').onchange=render;
const rotate=d=>map.easeTo({bearing:map.getBearing()+d,duration:350});
const tilt=d=>map.easeTo({pitch:Math.max(0,Math.min(80,map.getPitch()+d)),duration:350});
document.getElementById('rotL').onclick=()=>rotate(-30);
document.getElementById('rotR').onclick=()=>rotate(30);
document.getElementById('pitchUp').onclick=()=>tilt(10);
document.getElementById('pitchDn').onclick=()=>tilt(-10);
document.getElementById('reset').onclick=()=>map.easeTo({pitch:58,bearing:-17,duration:500});
let orbit=null;
document.getElementById('orbit').onclick=function(){
  if(orbit){clearInterval(orbit);orbit=null;this.classList.remove('on');this.textContent='Orbit';}
  else{this.classList.add('on');this.textContent='Stop';
    orbit=setInterval(()=>map.setBearing(map.getBearing()+0.5),40);}
};
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowLeft')go(idx-1); else if(e.key==='ArrowRight')go(idx+1);
  else if(e.key==='q'||e.key==='Q')rotate(-20); else if(e.key==='e'||e.key==='E')rotate(20);
  else if(e.key==='r'||e.key==='R')tilt(10); else if(e.key==='f'||e.key==='F')tilt(-10);
});
map.on('load', ()=>{ setupTerrain(); syncToggles(); buildSelect(); go(0); });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
