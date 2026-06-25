"""Build an interactive scatter viewer: 2011 path length vs each weather variable.

Cycle through weather variables (dropdown / prev-next / arrow keys), color points by
release stage, filter to a single stage, toggle log-Y. Each panel shows the Spearman
rho for the current selection + a least-squares guide line. Self-contained (Plotly CDN).

Output: weather_scatter.html
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from weather_length_corr import load_track, haversine_km, bearing, haversine_pairs, R

WVARS = [
    ("cloud_cover", "Total cloud cover", "%"),
    ("cloud_cover_low", "Low cloud cover", "%"),
    ("cloud_cover_mid", "Mid cloud cover", "%"),
    ("cloud_cover_high", "High cloud cover", "%"),
    ("wind_speed_10m", "Wind speed (10 m)", "km/h"),
    ("wind_gusts_10m", "Wind gusts (10 m)", "km/h"),
    ("tailwind_home", "Tailwind toward home", "km/h"),
    ("crosswind", "Crosswind", "km/h"),
    ("temperature_2m", "Temperature (2 m)", "°C"),
    ("precipitation", "Precipitation", "mm"),
    ("shortwave_radiation", "Shortwave radiation (sun)", "W/m²"),
    ("surface_pressure", "Surface pressure", "hPa"),
]


def build_df():
    w = pd.read_csv("flight_weather.csv")
    lengths, slat, slon, elat, elon = [], [], [], [], []
    for _, r in w.iterrows():
        t = load_track(r["route"], r["bird"], int(r["flight"]))
        if t is None:
            for L in (lengths, slat, slon, elat, elon):
                L.append(np.nan)
            continue
        lat, lon = t
        lengths.append(float(haversine_km(lat, lon).sum()))
        slat.append(lat[0]); slon.append(lon[0]); elat.append(lat[-1]); elon.append(lon[-1])
    w["path_km"] = lengths
    w["start_lat"] = slat; w["start_lon"] = slon
    home_lat, home_lon = np.nanmedian(elat), np.nanmedian(elon)
    brg = bearing(w["start_lat"].to_numpy(), w["start_lon"].to_numpy(), home_lat, home_lon)
    ang = np.radians((w["wind_direction_10m"].to_numpy() + 180) % 360 - brg)
    w["tailwind_home"] = w["wind_speed_10m"].to_numpy() * np.cos(ang)
    w["crosswind"] = w["wind_speed_10m"].to_numpy() * np.abs(np.sin(ang))
    return w.dropna(subset=["path_km"]).rename(columns={"flight": "stage"})


def main():
    w = build_df()
    stages = sorted(int(s) for s in w["stage"].unique())

    # records + precomputed Spearman rho (overall and per stage) for annotations
    records = [{"stage": int(r["stage"]), "path_km": round(float(r["path_km"]), 2),
                **{v: (None if pd.isna(r[v]) else round(float(r[v]), 3))
                   for v, _, _ in WVARS}}
               for _, r in w.iterrows()]
    rho = {}
    for v, _, _ in WVARS:
        d = {}
        rr, pp = spearmanr(w["path_km"], w[v], nan_policy="omit")
        d["all"] = [None if np.isnan(rr) else round(rr, 3),
                    None if np.isnan(pp) else round(pp, 4), int(len(w))]
        for s in stages:
            sub = w[w["stage"] == s]
            rr, pp = spearmanr(sub["path_km"], sub[v], nan_policy="omit")
            d[str(s)] = [None if np.isnan(rr) else round(rr, 3),
                         None if np.isnan(pp) else round(pp, 4), int(len(sub))]
        rho[v] = d

    payload = json.dumps({
        "vars": [v for v, _, _ in WVARS],
        "labels": {v: lbl for v, lbl, _ in WVARS},
        "units": {v: u for v, _, u in WVARS},
        "stages": stages, "records": records, "rho": rho,
    }, separators=(",", ":"))
    Path("viewers/weather_scatter.html").write_text(HTML.replace("__DATA__", payload))
    print(f"{len(records)} flights, {len(WVARS)} variables, stages {stages}")
    print(f"Saved weather_scatter.html "
          f"({Path('viewers/weather_scatter.html').stat().st_size/1e3:.0f} KB)")


HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Path length vs weather — 2011 pigeons</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root{--accent:#6ea8fe;--bg:#0e1118;--panel:#161b27;--line:#283044;--ink:#e8ecf3;
    --mut:#9aa6bd;--ctl:#1c2233;--ctlh:#28304a}
  body.light{--accent:#2f6fed;--bg:#f6f7fb;--panel:#ffffff;--line:#d8deea;--ink:#1a2030;
    --mut:#5a6478;--ctl:#eef1f7;--ctlh:#e0e6f2}
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
    font-family:'Inter',system-ui,-apple-system,sans-serif}
  #wrap{display:flex;flex-direction:column;height:100%}
  header{padding:14px 20px;border-bottom:1px solid var(--line);background:var(--panel)}
  header h1{margin:0;font-size:17px;font-weight:600}
  header .sub{color:var(--mut);font-size:12px;margin-top:2px}
  #bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:12px 20px;
    border-bottom:1px solid var(--line);background:var(--panel)}
  select,button{background:var(--ctl);color:var(--ink);border:1px solid var(--line);
    border-radius:8px;padding:7px 11px;font-size:13px;cursor:pointer}
  button:hover{background:var(--ctlh)}
  button.on{background:var(--accent);color:#0b0e16;border-color:var(--accent);font-weight:600}
  .grp{display:flex;gap:5px;align-items:center}
  .grp .lab{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-right:3px}
  .sbtn{padding:6px 9px;font-size:12px}
  label.chk{display:flex;gap:6px;align-items:center;font-size:12.5px;color:#cdd6e6;cursor:pointer}
  #plot{flex:1 1 0;min-height:0;position:relative;overflow:hidden}
  #plot>.js-plotly-plot{position:absolute;inset:0}
  #plot .js-plotly-plot,#plot .plot-container{height:100%}
  #cap{flex:0 0 auto;min-height:62px;padding:12px 20px;border-top:2px solid var(--accent);
    background:var(--panel);font-size:12.5px;color:var(--ink);line-height:1.7;
    box-shadow:0 -6px 18px rgba(0,0,0,.28);z-index:2}
  #cap b{color:#fff}
  .pill{display:inline-block;padding:1px 7px;border-radius:10px;background:var(--ctl);
    border:1px solid var(--line);margin-left:6px}
  #theme{margin-left:auto}
</style></head>
<body><div id="wrap">
  <header>
    <h1>Pigeon path length vs weather · 2011 (Flack et al.)</h1>
    <div class="sub">Each point = one flight. Color = release stage (1→6). Cycle variables with ◀ ▶ or arrow keys.</div>
  </header>
  <div id="bar">
    <div class="grp"><span class="lab">Variable</span>
      <button id="prev">◀</button>
      <select id="vsel"></select>
      <button id="next">▶</button></div>
    <div class="grp" id="stagebar"><span class="lab">Stage</span></div>
    <label class="chk"><input type="checkbox" id="logY"> log path length</label>
    <label class="chk"><input type="checkbox" id="fit" checked> models (null · linear · exp)</label>
    <button id="theme">☀ Light mode</button>
  </div>
  <div id="plot"></div>
  <div id="cap"></div>
</div>
<script>
const D = __DATA__;
let vi = 0, stage = 'all', logY = false, showFit = true, theme = 'dark';
document.getElementById('theme').onclick = ()=>{
  theme = theme==='dark' ? 'light' : 'dark';
  document.body.classList.toggle('light', theme==='light');
  document.getElementById('theme').textContent = theme==='dark' ? '☀ Light mode' : '🌙 Dark mode';
  draw();
};
const STAGES = D.stages;
// discrete blue->red gradient by stage
function stageColor(s){ const t=STAGES.length>1?(s-STAGES[0])/(STAGES[STAGES.length-1]-STAGES[0]):0;
  const h=240*(1-t); return 'hsl('+h+',75%,58%)'; }

const vsel = document.getElementById('vsel');
D.vars.forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.text=D.labels[v];vsel.add(o);});
vsel.onchange = e => { vi=+e.target.value; draw(); };
document.getElementById('prev').onclick = ()=>{ vi=(vi-1+D.vars.length)%D.vars.length; draw(); };
document.getElementById('next').onclick = ()=>{ vi=(vi+1)%D.vars.length; draw(); };
document.getElementById('logY').onchange = e => { logY=e.target.checked; draw(); };
document.getElementById('fit').onchange = e => { showFit=e.target.checked; draw(); };

const sbar = document.getElementById('stagebar');
function mkBtn(val,txt){ const b=document.createElement('button'); b.className='sbtn'; b.textContent=txt;
  b.dataset.val=val; b.onclick=()=>{ stage=val; draw(); }; sbar.appendChild(b); return b; }
mkBtn('all','All');
STAGES.forEach(s=>mkBtn(String(s),'R'+s));

document.addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'){vi=(vi+1)%D.vars.length;draw();}
  else if(e.key==='ArrowLeft'){vi=(vi-1+D.vars.length)%D.vars.length;draw();}
});

function lsline(xs,ys){ const n=xs.length; if(n<2) return null;
  let sx=0,sy=0,sxx=0,sxy=0; for(let i=0;i<n;i++){sx+=xs[i];sy+=ys[i];sxx+=xs[i]*xs[i];sxy+=xs[i]*ys[i];}
  const d=n*sxx-sx*sx; if(Math.abs(d)<1e-9) return null;
  const m=(n*sxy-sx*sy)/d, b=(sy-m*sx)/n; return {m,b}; }

function draw(){
  const v = D.vars[vi], lbl = D.labels[v], unit = D.units[v];
  vsel.value = vi;
  [...sbar.querySelectorAll('button')].forEach(b=>b.classList.toggle('on',b.dataset.val===stage));

  const recs = D.records.filter(r => r[v]!=null && (stage==='all'||String(r.stage)===stage));
  const traces = [];
  const groups = stage==='all' ? STAGES : [parseInt(stage)];
  groups.forEach(s=>{
    const pts = recs.filter(r=>r.stage===s);
    traces.push({x:pts.map(r=>r[v]), y:pts.map(r=>r.path_km), mode:'markers', type:'scatter',
      name:'R'+s, marker:{color:stageColor(s), size:7, opacity:.72, line:{width:0}},
      hovertemplate:lbl+': %{x}<br>path: %{y:.1f} km<br>release R'+s+'<extra></extra>'});
  });
  let modelTxt='';
  if(showFit && recs.length>2){
    const xs=recs.map(r=>r[v]), ys=recs.map(r=>r.path_km), n=xs.length;
    const mean=ys.reduce((a,b)=>a+b,0)/n;
    const sstot=ys.reduce((a,y)=>a+(y-mean)**2,0);
    const xmin=Math.min(...xs), xmax=Math.max(...xs);
    const stat=yhat=>{const ssr=ys.reduce((a,y,i)=>a+(y-yhat[i])**2,0);
      return {rmse:Math.sqrt(ssr/n), r2:sstot>0?1-ssr/sstot:0};};
    // null model = the mean (R^2 = 0 baseline)
    const sN=stat(ys.map(()=>mean));
    traces.push({x:[xmin,xmax],y:[mean,mean],mode:'lines',name:'null (mean)',
      line:{color:'#8d97ac',width:1.6,dash:'dot'},hoverinfo:'skip'});
    // linear least-squares
    const lin=lsline(xs,ys); let sL=null;
    if(lin){ sL=stat(xs.map(x=>lin.m*x+lin.b));
      traces.push({x:[xmin,xmax],y:[lin.m*xmin+lin.b,lin.m*xmax+lin.b],mode:'lines',
        name:'linear',line:{color:'#ffcd5a',width:2},hoverinfo:'skip'}); }
    // exponential y=a*exp(b x), fit on log y
    const ef=lsline(xs,ys.map(Math.log)); let sE=null;
    if(ef){ const a=Math.exp(ef.b), b=ef.m;
      sE=stat(xs.map(x=>a*Math.exp(b*x)));
      const N=60,cx=[],cy=[]; for(let i=0;i<N;i++){const x=xmin+(xmax-xmin)*i/(N-1);
        cx.push(x);cy.push(a*Math.exp(b*x));}
      traces.push({x:cx,y:cy,mode:'lines',name:'exponential',
        line:{color:'#5fe0c0',width:2},hoverinfo:'skip'}); }
    const f=s=>s?('RMSE '+s.rmse.toFixed(1)+' · R² '+s.r2.toFixed(3)):'n/a';
    modelTxt='<span class="pill">null: '+f(sN)+'</span>'+
             '<span class="pill">linear: '+f(sL)+'</span>'+
             '<span class="pill">exp: '+f(sE)+'</span>';
  }
  const dark = theme==='dark';
  const bg = dark?'#0e1118':'#ffffff', ink = dark?'#c7d0e0':'#2a3140',
        grid = dark?'#222a3a':'#e3e7f0';
  const layout={
    paper_bgcolor:bg, plot_bgcolor:bg, font:{color:ink,size:12},
    margin:{l:64,r:20,t:14,b:52}, showlegend:true,
    legend:{orientation:'h',y:1.06,font:{size:11}},
    xaxis:{title:lbl+(unit?' ('+unit+')':''), gridcolor:grid, zeroline:false},
    yaxis:{title:'path length (km)', gridcolor:grid, zeroline:false,
      type:logY?'log':'linear'}
  };
  Plotly.react('plot',traces,layout,{responsive:true,displaylogo:false})
    .then(()=>Plotly.Plots.resize('plot'));

  const r = D.rho[v][stage] || [null,null,0];
  const rv = r[0]==null?'n/a':r[0].toFixed(3);
  const pv = r[1]==null?'':(r[1]<0.001?'p<0.001':'p='+r[1].toFixed(3));
  const sig = r[1]==null?'':(r[1]<0.001?'***':r[1]<0.01?'**':r[1]<0.05?'*':'(n.s.)');
  document.getElementById('cap').innerHTML =
    '<b>'+lbl+'</b> vs path length · '+(stage==='all'?'all releases':'release '+stage)+
    ' <span class="pill">Spearman ρ = '+rv+' '+sig+'</span>'+
    '<span class="pill">'+pv+'</span><span class="pill">n = '+r[2]+'</span>'+
    (modelTxt?'<br>'+modelTxt:'');
}
draw();
</script></body></html>
"""


if __name__ == "__main__":
    main()
