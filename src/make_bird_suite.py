"""Build the interactive pigeon viewer with TWO toggleable datasets:

  2011 - Flack et al.: routes R1/R2/R3, 6 releases/bird, with ERA5 weather + cloud
  2021 - Valentini et al.: Solo / Pair / Gen1-5 conditions, many releases (no weather)

Single-bird mode  : cycle units, releases drawn first->last on a colour gradient.
Heatmap mode      : pool all birds by group (route or condition) and release stage.
Basemaps          : light / street / satellite / topo / dark.

Inputs : data/R*/*/*.csv, flight_weather.csv, cloud_grid.json, data_2021.json
Output : bird_route_development.html  (self-contained; Leaflet from CDN)
"""

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from geo_simplify import simplify
from sun_calc import sun_position

DATA = Path("data")
EPS_M = 4.0    # RDP tolerance in meters (smaller = more detail)
CAP = 1500     # safety ceiling on points per flight
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Geomagnetic field at the Oxford release site (IGRF, ppigrf), per dataset epoch.
# declination (deg E of true N), inclination (deg below horizontal), F (uT).
GEOMAG = {
    "2011": {"dec": -1.97, "inc": 66.73, "F": 48.7},   # epoch 2011.4
    "2021": {"dec": -1.09, "inc": 66.71, "F": 48.8},   # epoch 2016.4 (collection)
}


def load_track(csv_path):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[df["Latitude"].between(45, 60) & df["Longitude"].between(-10, 5)]
    if df.empty:
        return []
    return simplify(list(zip(df["Latitude"], df["Longitude"])),
                    eps_m=EPS_M, cap=CAP)


def load_weather():
    wx = {}
    if not Path("flight_weather.csv").exists():
        return wx
    df = pd.read_csv("flight_weather.csv")
    for _, r in df.iterrows():
        deg = float(r["wind_direction_10m"])
        t = str(r["start_time"])
        hh, mm, ss = (int(x) for x in t.split(":"))
        y, mo, d = (int(x) for x in str(r["date"]).split("-"))
        # local Europe/London is BST (UTC+1) in May-Jun -> UTC = local - 1h
        utc = dt.datetime(y, mo, d, hh, mm, ss) - dt.timedelta(hours=1)
        az, el = sun_position(float(r["rel_lat"]), float(r["rel_lon"]), utc)
        wx[(r["route"], r["bird"], int(r["flight"]))] = {
            "date": str(r["date"]), "hour": hh,
            "cloud": round(float(r["cloud_cover"])),
            "wind": round(float(r["wind_speed_10m"])),
            "wcomp": COMPASS[int((deg + 22.5) // 45) % 8],
            "temp": round(float(r["temperature_2m"]), 1),
            "sunAz": az, "sunEl": el,
        }
    return wx


def build_2011():
    wx = load_weather()
    units = []
    for bird_dir in sorted(p for p in DATA.glob("R*/*") if p.is_dir()):
        route, bird_id = bird_dir.parts[1], bird_dir.parts[2]
        flights = []
        for csv in sorted(bird_dir.glob("*.csv")):
            coords = load_track(csv)
            if not coords:
                continue
            n = csv.stem.split("_")[-1]
            f = {"n": n, "coords": coords}
            f.update(wx.get((route, bird_id, int(n)), {}))
            flights.append(f)
        if flights:
            units.append({"id": bird_id, "group": route, "flights": flights})
    maxrel = max((int(f["n"]) for u in units for f in u["flights"]), default=6)
    return {
        "key": "2011", "label": "2011 · Flack (routes)",
        "groupLabel": "Release site", "groups": ["R1", "R2", "R3"],
        "hasWeather": True, "maxRel": maxrel, "units": units,
        "geomag": GEOMAG["2011"],
    }


def build_2021():
    p = Path("data_2021.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    maxrel = max((int(f["n"]) for u in d["units"] for f in u["flights"]),
                 default=12)
    return {
        "key": "2021", "label": "2021 · Valentini (solo/pair/gen)",
        "groupLabel": "Condition", "groups": d["groups"],
        "hasWeather": False, "maxRel": maxrel, "units": d["units"],
        "geomag": GEOMAG["2021"],
    }


def main():
    datasets = {}
    d2011 = build_2011()
    datasets[d2011["key"]] = d2011
    print(f"2011: {len(d2011['units'])} units")
    d2021 = build_2021()
    if d2021:
        datasets[d2021["key"]] = d2021
        print(f"2021: {len(d2021['units'])} units, groups {d2021['groups']}")
    else:
        print("2021: data_2021.json not found — run prep_2021.py first")

    cloud = "null"
    if Path("cloud_grid.json").exists():
        cloud = Path("cloud_grid.json").read_text()

    html = (HTML_TEMPLATE
            .replace("__DATASETS__", json.dumps(datasets, separators=(",", ":")))
            .replace("__CLOUD__", cloud))
    out = "viewers/bird_route_development.html"
    Path(out).write_text(html)
    print(f"Saved {out} ({Path(out).stat().st_size/1e6:.1f} MB)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pigeon route development — Flack 2011 & Valentini 2021</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>
  html,body{margin:0;height:100%;font-family:system-ui,sans-serif}
  #map{position:absolute;top:0;bottom:0;left:0;right:0}
  #sunShade{position:absolute;top:0;bottom:0;left:0;right:0;z-index:450;
    pointer-events:none;display:none}
  #panel{position:absolute;top:12px;left:12px;z-index:1000;background:#fff;
    padding:12px 14px;border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,.3);
    width:308px;max-height:calc(100% - 30px);overflow:auto}
  #panel h3{margin:0 0 4px;font-size:15px}
  #panel .sub{color:#666;font-size:12px;margin-bottom:8px}
  select{width:100%;padding:5px;font-size:14px;margin-bottom:8px}
  .row{display:flex;gap:6px;margin-bottom:8px}
  .row button{flex:1;padding:6px;font-size:13px;cursor:pointer;
    border:1px solid #bbb;background:#f7f7f7;border-radius:5px}
  .row button:hover{background:#eee}
  .row button.on{background:#3367d6;color:#fff;border-color:#3367d6}
  .dsbar button.on{background:#9b1d64;border-color:#9b1d64}
  #info,#heatInfo{font-size:12px;color:#333;margin-bottom:8px}
  table.rel{width:100%;border-collapse:collapse;font-size:11px;margin-bottom:8px}
  table.rel th{text-align:left;color:#888;font-weight:600;padding:2px 3px}
  table.rel td{padding:2px 3px;border-top:1px solid #eee;cursor:pointer}
  table.rel tr.sel{background:#fff4d6}
  .sw{display:inline-block;width:11px;height:11px;border-radius:2px;
    vertical-align:-1px}
  .toggle{font-size:12px;margin:6px 0;display:flex;align-items:center;gap:6px}
  .chips{display:flex;flex-wrap:wrap;gap:5px;margin:4px 0 10px}
  .chip{padding:4px 9px;font-size:12px;cursor:pointer;border:1px solid #bbb;
    background:#f7f7f7;border-radius:14px;user-select:none}
  .chip.on{background:#2b8a3e;color:#fff;border-color:#2b8a3e}
  .lbl{font-size:12px;color:#555;margin:4px 0 2px;font-weight:600}
  input[type=range]{width:100%}
  .bar{height:12px;border-radius:3px;margin:4px 0;
    background:linear-gradient(to right,hsl(240,78%,47%),hsl(180,78%,45%),
      hsl(120,70%,42%),hsl(60,80%,45%),hsl(0,80%,50%))}
  .cbar{height:12px;border-radius:3px;margin:4px 0;
    background:linear-gradient(to right,#ffffff,#9aa3ad,#3b424b)}
  .hbar{height:12px;border-radius:3px;margin:4px 0;
    background:linear-gradient(to right,#0000ff,#00ff00,#ff0000)}
  .barlbl{display:flex;justify-content:space-between;color:#666}
  .legend{font-size:12px;margin-top:6px}
</style>
</head>
<body>
<div id="map"></div>
<div id="sunShade"></div>
<div id="panel">
  <h3>Pigeon route development</h3>
  <div class="sub" id="dsSub"></div>

  <div class="lbl">Dataset</div>
  <div class="row dsbar" id="dsBar"></div>

  <div class="row">
    <button id="modeBird" class="on">Single bird</button>
    <button id="modeHeat">Heatmap</button>
  </div>

  <div id="birdMode">
    <select id="birdSel"></select>
    <div class="row">
      <button id="prev">&#8592; Prev</button>
      <button id="next">Next &#8594;</button>
    </div>
    <div id="info"></div>
    <table class="rel" id="relTbl"></table>
    <label class="toggle" id="cloudTogWrap"><input type="checkbox" id="cloudTog">
      Show cloud cover (selected release)</label>
    <label class="toggle" id="sunTogWrap"><input type="checkbox" id="sunTog">
      Sun direction (selected release)</label>
    <label class="toggle"><input type="checkbox" id="geoTog">
      Geomagnetic field (compass cue)</label>
    <div id="navInfo" style="font-size:11px;color:#555;margin:2px 0 8px"></div>
    <div class="legend">
      <div>Release order</div><div class="bar"></div>
      <div class="barlbl"><span>first</span><span>last</span></div>
      <div id="cloudLeg">
        <div style="margin-top:6px">Cloud cover</div><div class="cbar"></div>
        <div class="barlbl"><span>0%</span><span>100%</span></div>
      </div>
    </div>
  </div>

  <div id="heatMode" style="display:none">
    <div class="lbl" id="groupLbl">Group (toggle)</div>
    <div class="chips" id="groupChips"></div>
    <label class="toggle"><input type="checkbox" id="stageAll" checked>
      All releases pooled</label>
    <div id="stageWrap" style="display:none">
      <div class="lbl">Release stage: <span id="stageVal">1</span></div>
      <input type="range" id="stageSlider" min="1" max="6" value="1">
    </div>
    <div id="heatInfo"></div>
    <div class="legend">
      <div>Bird density (all birds pooled)</div><div class="hbar"></div>
      <div class="barlbl"><span>sparse</span><span>dense</span></div>
    </div>
  </div>
</div>
<script>
const DATASETS = __DATASETS__;
const CLOUD = __CLOUD__;

const map = L.map('map');
const CARTO_ATTR='&copy; OpenStreetMap &copy; CARTO', OSM_ATTR='&copy; OpenStreetMap contributors', ESRI='Tiles &copy; Esri';
const baseLayers = {
  'Light': L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{attribution:CARTO_ATTR,maxZoom:20}),
  'Street (OSM)': L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:OSM_ATTR,maxZoom:19}),
  'Satellite': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{attribution:ESRI+' — Maxar',maxZoom:20}),
  'Topographic': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',{attribution:ESRI,maxZoom:20}),
  'Dark': L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{attribution:CARTO_ATTR,maxZoom:20}),
};
baseLayers['Light'].addTo(map);
const overlayLayers = {'Place labels': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',{attribution:ESRI,maxZoom:20})};
L.control.layers(baseLayers, overlayLayers, {position:'topright'}).addTo(map);

function grad(t){ return 'hsl(' + Math.round(240*(1-t)) + ',78%,47%)'; }

let trackLayer = L.layerGroup().addTo(map);
let cloudLayer = L.layerGroup().addTo(map);
let heatLayer = null;
let dsKey = Object.keys(DATASETS)[0];
let idx = 0, selRel = 0, mode = 'bird';
let selGroups = new Set();
let stageAll = true, stageVal = 1;
const DS = () => DATASETS[dsKey];

// ---------- single-bird ----------
function drawCloud(){
  cloudLayer.clearLayers();
  if(!DS().hasWeather || !CLOUD) return;
  if(!document.getElementById('cloudTog').checked) return;
  const f = DS().units[idx].flights[selRel];
  if(f.date===undefined) return;
  const field = CLOUD.grid[f.date] && CLOUD.grid[f.date][f.hour];
  if(!field) return;
  const lat=CLOUD.lat, lon=CLOUD.lon, dLat=(lat[1]-lat[0])/2, dLon=(lon[1]-lon[0])/2;
  for(let i=0;i<lat.length;i++) for(let j=0;j<lon.length;j++){
    const c=field[i*lon.length+j]; if(c==null) continue;
    L.rectangle([[lat[i]-dLat,lon[j]-dLon],[lat[i]+dLat,lon[j]+dLon]],
      {stroke:false,fillColor:'#3b424b',fillOpacity:0.05+0.55*(c/100)}).addTo(cloudLayer);
  }
}
function wxLine(f){
  if(f.cloud===undefined) return '';
  return '&#9729;'+f.cloud+'% &middot; '+f.wind+'km/h '+f.wcomp+' &middot; '+f.temp+'&deg;C';
}

// ---------- navigation cues: sun + geomagnetic field ----------
let navLayer = L.layerGroup().addTo(map);
function destLatLng(lat, lon, bearingDeg, distM){
  const br = bearingDeg*Math.PI/180;
  const dLat = distM*Math.cos(br)/111320;
  const dLon = distM*Math.sin(br)/(111320*Math.cos(lat*Math.PI/180));
  return [lat+dLat, lon+dLon];
}
function arrow(a, bearing, distM, color, weight){
  const b = destLatLng(a[0], a[1], bearing, distM);
  const h = distM*0.22;
  [[a,b],
   [b, destLatLng(b[0],b[1],bearing+150,h)],
   [b, destLatLng(b[0],b[1],bearing-150,h)]
  ].forEach(s => L.polyline(s,{color,weight:weight||3,opacity:0.9}).addTo(navLayer));
  return b;
}
const COMPASS8=['N','NE','E','SE','S','SW','W','NW'];
function compass(az){ return COMPASS8[Math.round(az/45)%8]; }

function drawNav(){
  navLayer.clearLayers();
  const shade=document.getElementById('sunShade'); shade.style.display='none';
  if(mode!=='bird'){ document.getElementById('navInfo').innerHTML=''; return; }
  const ds=DS(), bnds=map.getBounds();
  const spanM=(bnds.getNorth()-bnds.getSouth())*111320;
  const info=[];
  // geomagnetic field — uniform grid of arrows to magnetic north
  if(document.getElementById('geoTog').checked && ds.geomag){
    const g=ds.geomag, n=4;
    const dLat=(bnds.getNorth()-bnds.getSouth())/(n+1);
    const dLon=(bnds.getEast()-bnds.getWest())/(n+1);
    for(let i=1;i<=n;i++) for(let j=1;j<=n;j++){
      arrow([bnds.getSouth()+dLat*i, bnds.getWest()+dLon*j], g.dec, spanM/(n+2),
            '#c0392b', 2);
    }
    info.push('<span style="color:#c0392b">&#9650;</span> Magnetic N: dec '+
      g.dec+'°, inc '+g.inc+'°, '+g.F+' µT');
  }
  // sun — arrow toward the sun + directional shade
  const u=ds.units[idx], f=u?u.flights[selRel]:null;
  if(document.getElementById('sunTog').checked && f && f.sunEl!==undefined){
    if(f.sunEl>0){
      const tip=arrow(f.coords[0], f.sunAz, spanM*0.18, '#e67e22', 4);
      L.circleMarker(tip,{radius:6,color:'#e67e22',fillColor:'#ffd24d',
        fillOpacity:1,weight:1}).bindTooltip('☀ az '+f.sunAz+'°, el '+
        f.sunEl+'°',{sticky:true}).addTo(navLayer);
      shade.style.display='block';
      shade.style.background='linear-gradient('+f.sunAz+'deg,'+
        'rgba(15,20,45,0.22), rgba(255,210,120,0.22))';
      info.push('☀ Sun: '+compass(f.sunAz)+' (az '+f.sunAz+'°), elev '+
        f.sunEl+'°');
    } else { info.push('☀ Sun below horizon'); }
  }
  document.getElementById('navInfo').innerHTML=info.join('<br>');
}
function draw(i){
  const U = DS().units;
  idx = (i + U.length) % U.length; selRel = 0;
  const b = U[idx], N = b.flights.length, all = [];
  trackLayer.clearLayers();
  b.flights.forEach((f,k) => {
    const color = grad(N>1?k/(N-1):0);
    const tip = b.id+' &middot; release '+f.n +
      (f.date? '<br>'+f.date+' '+String(f.hour).padStart(2,'0')+':00<br>'+wxLine(f):'');
    L.polyline(f.coords,{color,weight:3,opacity:0.85}).bindTooltip(tip,{sticky:true}).addTo(trackLayer);
    L.circleMarker(f.coords[0],{radius:3,color,fillColor:color,fillOpacity:1,weight:1}).addTo(trackLayer);
    all.push(...f.coords);
  });
  if(all.length) map.fitBounds(all,{padding:[30,30]});
  document.getElementById('info').innerHTML =
    '<b>'+b.id+'</b> &middot; '+b.group+' &middot; '+N+' releases<br>unit '+(idx+1)+' of '+U.length;
  const wx = DS().hasWeather;
  let rows = '<tr><th></th><th>rel</th>' + (wx?'<th>date</th><th>cloud</th><th>wind</th>':'<th>pts</th>') + '</tr>';
  b.flights.forEach((f,k) => {
    const sw='<span class="sw" style="background:'+grad(N>1?k/(N-1):0)+'"></span>';
    rows += '<tr data-k="'+k+'"'+(k===selRel?' class="sel"':'')+'><td>'+sw+'</td><td>'+f.n+'</td>' +
      (wx? '<td>'+(f.date?f.date.slice(5):'-')+'</td><td>'+(f.cloud!==undefined?f.cloud+'%':'-')+
           '</td><td>'+(f.wind!==undefined?f.wind+' '+f.wcomp:'-')+'</td>'
         : '<td>'+f.coords.length+'</td>') + '</tr>';
  });
  document.getElementById('relTbl').innerHTML = rows;
  document.querySelectorAll('#relTbl tr[data-k]').forEach(tr => {
    tr.onclick = () => { selRel=+tr.dataset.k;
      document.querySelectorAll('#relTbl tr').forEach(x=>x.classList.remove('sel'));
      tr.classList.add('sel'); drawCloud(); drawNav(); };
  });
  document.getElementById('birdSel').value = idx;
  drawCloud(); drawNav();
}

// ---------- heatmap ----------
function buildHeatControls(){
  const gc = document.getElementById('groupChips'); gc.innerHTML='';
  document.getElementById('groupLbl').textContent = DS().groupLabel + ' (toggle)';
  DS().groups.forEach(g => {
    const c=document.createElement('span');
    c.className='chip'+(selGroups.has(g)?' on':''); c.textContent=g;
    c.onclick=()=>{ selGroups.has(g)?selGroups.delete(g):selGroups.add(g);
      c.classList.toggle('on'); drawHeat(); };
    gc.appendChild(c);
  });
  const sl=document.getElementById('stageSlider');
  sl.max = DS().maxRel; if(stageVal>DS().maxRel){stageVal=1; sl.value=1;
    document.getElementById('stageVal').textContent=1;}
}
function drawHeat(){
  if(heatLayer){ map.removeLayer(heatLayer); heatLayer=null; }
  if(mode!=='heat') return;
  const pts=[]; let n=0;
  DS().units.forEach(u => {
    if(!selGroups.has(u.group)) return;
    u.flights.forEach(f => {
      if(!stageAll && parseInt(f.n)!==stageVal) return;
      n++; for(const c of f.coords) pts.push([c[0],c[1],0.4]);
    });
  });
  document.getElementById('heatInfo').innerHTML =
    '<b>'+[...selGroups].join(', ')+'</b> &middot; '+(stageAll?'all releases':'release #'+stageVal)+
    '<br>'+n+' flights, '+pts.length.toLocaleString()+' GPS points';
  if(!pts.length) return;
  heatLayer = L.heatLayer(pts,{radius:14,blur:11,minOpacity:0.25,maxZoom:13,max:1.0}).addTo(map);
  map.fitBounds(pts.map(p=>[p[0],p[1]]),{padding:[30,30]});
}

// ---------- mode / dataset ----------
function setMode(m){
  mode=m;
  document.getElementById('modeBird').classList.toggle('on',m==='bird');
  document.getElementById('modeHeat').classList.toggle('on',m==='heat');
  document.getElementById('birdMode').style.display=m==='bird'?'block':'none';
  document.getElementById('heatMode').style.display=m==='heat'?'block':'none';
  if(m==='bird'){
    if(heatLayer){map.removeLayer(heatLayer);heatLayer=null;}
    map.addLayer(trackLayer); map.addLayer(cloudLayer); map.addLayer(navLayer);
    draw(idx);
  } else {
    map.removeLayer(trackLayer); map.removeLayer(cloudLayer);
    map.removeLayer(navLayer); document.getElementById('sunShade').style.display='none';
    drawHeat();
  }
}
function initDataset(){
  selGroups = new Set(DS().groups);
  document.getElementById('dsSub').innerHTML = DS().label +
    (DS().hasWeather? ' &middot; ERA5 weather' : ' &middot; no weather data');
  // bird dropdown
  const sel=document.getElementById('birdSel'); sel.innerHTML='';
  DS().units.forEach((b,i)=>{ const o=document.createElement('option');
    o.value=i; o.text=b.id+'  ('+b.flights.length+' rel)'; sel.add(o); });
  // weather UI visibility
  const wx=DS().hasWeather;
  document.getElementById('cloudTogWrap').style.display=wx?'flex':'none';
  document.getElementById('cloudLeg').style.display=wx?'block':'none';
  document.getElementById('sunTogWrap').style.display=wx?'flex':'none';
  idx=0; buildHeatControls(); setMode(mode);
}
// dataset buttons
const dsBar=document.getElementById('dsBar');
Object.keys(DATASETS).forEach(k=>{
  const btn=document.createElement('button');
  btn.textContent=k; btn.className=(k===dsKey?'on':'');
  btn.onclick=()=>{ dsKey=k;
    [...dsBar.children].forEach(b=>b.classList.toggle('on',b.textContent===k));
    initDataset(); };
  dsBar.appendChild(btn);
});

document.getElementById('birdSel').onchange=e=>draw(+e.target.value);
document.getElementById('prev').onclick=()=>draw(idx-1);
document.getElementById('next').onclick=()=>draw(idx+1);
document.getElementById('cloudTog').onchange=drawCloud;
document.getElementById('sunTog').onchange=drawNav;
document.getElementById('geoTog').onchange=drawNav;
map.on('moveend', ()=>{ if(mode==='bird') drawNav(); });
document.getElementById('modeBird').onclick=()=>setMode('bird');
document.getElementById('modeHeat').onclick=()=>setMode('heat');
document.getElementById('stageAll').onchange=e=>{
  stageAll=e.target.checked;
  document.getElementById('stageWrap').style.display=stageAll?'none':'block';
  drawHeat();
};
document.getElementById('stageSlider').oninput=e=>{
  stageVal=+e.target.value; document.getElementById('stageVal').textContent=stageVal; drawHeat();
};
document.addEventListener('keydown',e=>{
  if(mode!=='bird') return;
  if(e.key==='ArrowLeft') draw(idx-1);
  if(e.key==='ArrowRight') draw(idx+1);
});

initDataset();
draw(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
