"""Interactive vision-vs-altitude viewer -> vision_altitude.html.

Plots the pigeon recognition radius vs altitude (horizon limit, acuity limit, and
their min), with sliders for landmark size / acuity / recognition elements /
visibility cap / ground elevation. Overlays the observed 2011 flight-altitude
distribution so you can see which regime the birds actually operate in.

Mirrors the math in vision_model.py (kept in sync by hand).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

import vision_model as vm


def main():
    # true above-ground altitude distribution, DEM-corrected by terrain_agl.py
    hist = json.loads(Path("agl_hist.json").read_text())
    payload = json.dumps({
        "R": vm.R_EARTH,
        "histCenters": hist["centers"],
        "histCounts": hist["counts"],
    }, separators=(",", ":"))
    Path("viewers/vision_altitude.html").write_text(HTML.replace("__DATA__", payload))
    print(f"alt bins: {len(hist['centers'])}, total fixes: {int(sum(hist['counts'])):,}")
    print(f"Saved vision_altitude.html "
          f"({Path('viewers/vision_altitude.html').stat().st_size/1e3:.0f} KB)")


HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pigeon vision vs altitude</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root{--accent:#6ea8fe;--bg:#0e1118;--panel:#161b27;--line:#283044;--ink:#e8ecf3;
    --mut:#9aa6bd;--ctl:#1c2233;--ctlh:#28304a}
  body.light{--accent:#2f6fed;--bg:#f6f7fb;--panel:#ffffff;--line:#d8deea;--ink:#1a2030;
    --mut:#5a6478;--ctl:#eef1f7;--ctlh:#e0e6f2}
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
    font-family:'Inter',system-ui,-apple-system,sans-serif}
  #wrap{display:flex;height:100%}
  #side{width:300px;flex:0 0 300px;background:var(--panel);border-right:1px solid var(--line);
    padding:16px 18px;overflow-y:auto}
  #side h1{font-size:16px;margin:0 0 2px}
  #side .sub{color:var(--mut);font-size:11.5px;margin-bottom:14px;line-height:1.5}
  .ctl{margin:13px 0}
  .ctl label{display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:5px}
  .ctl label b{color:var(--accent);font-weight:600}
  input[type=range]{width:100%;accent-color:var(--accent)}
  #main{flex:1;display:flex;flex-direction:column;min-width:0}
  #plot{flex:1 1 0;min-height:0;position:relative;overflow:hidden}
  #plot>.js-plotly-plot{position:absolute;inset:0}
  #cap{flex:0 0 auto;min-height:78px;padding:12px 18px;border-top:2px solid var(--accent);
    background:var(--panel);font-size:12.5px;color:var(--ink);line-height:1.7}
  #cap b{color:var(--accent)} .pill{display:inline-block;padding:1px 7px;border-radius:10px;
    background:var(--ctl);border:1px solid var(--line);margin:0 6px 4px 0}
  button{background:var(--ctl);color:var(--ink);border:1px solid var(--line);border-radius:8px;
    padding:6px 11px;font-size:12.5px;cursor:pointer;margin-top:8px}
  button:hover{background:var(--ctlh)}
</style></head>
<body><div id="wrap">
  <div id="side">
    <h1>Pigeon vision × altitude</h1>
    <div class="sub">Recognition radius = min(geometric horizon, visual-acuity range).
      Gray histogram = real 2011 <b>in-flight</b> altitude (speed &gt;15 km/h),
      <b>true height above ground</b> (GPS sea-level altitude minus a Terrarium DEM).</div>
    <div class="ctl"><label>Landmark size <b><span id="lLab">100</span> m</b></label>
      <input type="range" id="L" min="5" max="1000" step="5" value="100"></div>
    <div class="ctl"><label>Visual acuity <b><span id="aLab">12</span> cyc/deg</b></label>
      <input type="range" id="acuity" min="2" max="30" step="0.5" value="12"></div>
    <div class="ctl"><label>Recognition elements <b><span id="kLab">5</span> bars</b></label>
      <input type="range" id="k" min="1" max="20" step="1" value="5"></div>
    <div class="ctl"><label>Visibility cap <b><span id="vLab">off</span></b></label>
      <input type="range" id="vis" min="0" max="60" step="1" value="0"></div>
    <button id="reset">Reset defaults</button>
    <button id="theme">☀ Light mode</button>
  </div>
  <div id="main"><div id="plot"></div><div id="cap"></div></div>
</div>
<script>
const D = __DATA__, R = D.R;
let theme='dark';
const $=id=>document.getElementById(id);

function horizon(h){ h=Math.max(0,h); return Math.sqrt(2*R*h+h*h); }
function mraDeg(a){ return 1/(2*a); }
function acuityRange(L,a,k){ const th=(k*mraDeg(a))*Math.PI/180; return L/(2*Math.tan(th/2)); }
function crossover(L,a,k){ const d=acuityRange(L,a,k); return d*d/(2*R); }

function params(){ return {L:+$('L').value, a:+$('acuity').value, k:+$('k').value,
  vis:(+$('vis').value)||0}; }

function pctile(p){ // percentile of AGL altitude from histogram (already AGL)
  const tot=D.histCounts.reduce((x,y)=>x+y,0); let c=0;
  for(let i=0;i<D.histCenters.length;i++){ c+=D.histCounts[i];
    if(c/tot>=p) return Math.max(0, D.histCenters[i]); }
  return 0;
}
function fracBelow(altAGL){ const tot=D.histCounts.reduce((x,y)=>x+y,0); let c=0;
  for(let i=0;i<D.histCenters.length;i++){ if(D.histCenters[i]<altAGL) c+=D.histCounts[i]; }
  return c/tot;
}

function draw(){
  const p=params();
  $('lLab').textContent=p.L; $('aLab').textContent=p.a; $('kLab').textContent=p.k;
  $('vLab').textContent=p.vis?p.vis+' km':'off';
  const dark=theme==='dark';
  const bg=dark?'#0e1118':'#fff', ink=dark?'#c7d0e0':'#2a3140', grid=dark?'#222a3a':'#e3e7f0';

  const aR=acuityRange(p.L,p.a,p.k)/1000;           // km, flat in altitude
  const cross=crossover(p.L,p.a,p.k);                // m AGL
  const visKm=p.vis||Infinity;
  const H=[]; for(let h=0;h<=300;h+=2) H.push(h);
  const horizKm=H.map(h=>horizon(h)/1000);
  const effKm=H.map(h=>Math.min(horizon(h)/1000, aR, visKm));

  // observed AGL altitude histogram (secondary axis)
  const aglC=D.histCenters;
  const maxCount=Math.max(...D.histCounts);

  const traces=[
    {x:aglC, y:D.histCounts.map(c=>c/maxCount), type:'bar', name:'2011 flight altitude',
     marker:{color:dark?'rgba(150,160,180,.30)':'rgba(120,130,150,.30)'},
     yaxis:'y2', hovertemplate:'AGL %{x:.0f} m<extra></extra>', width:5},
    {x:H, y:horizKm, mode:'lines', name:'horizon limit',
     line:{color:'#6ea8fe',width:2,dash:'dot'}},
    {x:[0,300], y:[aR,aR], mode:'lines', name:'acuity limit',
     line:{color:'#5fe0c0',width:2,dash:'dash'}},
    {x:H, y:effKm, mode:'lines', name:'effective recognition radius',
     line:{color:'#ffcd5a',width:3.5}},
  ];
  if(p.vis) traces.push({x:[0,300],y:[visKm,visKm],mode:'lines',name:'visibility cap',
     line:{color:'#e6708a',width:1.6,dash:'dashdot'}});

  const medAGL=pctile(0.5), p95=pctile(0.95);
  const shapes=[
    {type:'line',x0:cross,x1:cross,y0:0,y1:1,yref:'paper',
     line:{color:dark?'#8d97ac':'#888',width:1,dash:'dot'}},
    {type:'line',x0:medAGL,x1:medAGL,y0:0,y1:1,yref:'paper',
     line:{color:'#ff8f5a',width:1.4}}
  ];
  const ann=[
    {x:cross,y:1,yref:'paper',text:'crossover '+cross.toFixed(0)+' m',showarrow:false,
     font:{size:10,color:ink},yanchor:'bottom',xanchor:cross>150?'right':'left'},
    {x:medAGL,y:1,yref:'paper',text:'median bird '+medAGL.toFixed(0)+' m',showarrow:false,
     font:{size:10,color:'#ff8f5a'},yanchor:'bottom',xanchor:'left'}
  ];
  const layout={paper_bgcolor:bg,plot_bgcolor:bg,font:{color:ink,size:12},
    margin:{l:60,r:60,t:24,b:48},legend:{orientation:'h',y:1.12,font:{size:11}},
    xaxis:{title:'altitude above ground (m)',gridcolor:grid,zeroline:false,range:[0,300]},
    yaxis:{title:'recognition radius (km)',gridcolor:grid,zeroline:false,rangemode:'tozero'},
    yaxis2:{overlaying:'y',side:'right',showgrid:false,range:[0,4],showticklabels:false},
    shapes,annotations:ann,barmode:'overlay'};
  Plotly.react('plot',traces,layout,{responsive:true,displaylogo:false})
    .then(()=>Plotly.Plots.resize('plot'));

  const effMed=Math.min(horizon(medAGL)/1000,aR,visKm);
  const regime=medAGL<cross?'horizon-limited':'acuity-limited';
  const fHoriz=(fracBelow(cross)*100).toFixed(0);
  $('cap').innerHTML=
    '<b>At the median bird altitude ('+medAGL.toFixed(0)+' m AGL):</b> can recognize a '+p.L+
    ' m landmark out to <b>'+effMed.toFixed(1)+' km</b> ('+regime+').'+
    '<br><span class="pill">acuity range = '+aR.toFixed(1)+' km</span>'+
    '<span class="pill">crossover = '+cross.toFixed(0)+' m AGL</span>'+
    '<span class="pill">'+fHoriz+'% of fixes are horizon-limited</span>'+
    '<span class="pill">95th-pct altitude = '+p95.toFixed(0)+' m AGL</span>';
}

['L','acuity','k','vis'].forEach(id=>$(id).oninput=draw);
$('reset').onclick=()=>{$('L').value=100;$('acuity').value=12;$('k').value=5;
  $('vis').value=0;draw();};
$('theme').onclick=()=>{ theme=theme==='dark'?'light':'dark';
  document.body.classList.toggle('light',theme==='light');
  $('theme').textContent=theme==='dark'?'☀ Light mode':'🌙 Dark mode'; draw(); };
draw();
</script></body></html>
"""


if __name__ == "__main__":
    main()
