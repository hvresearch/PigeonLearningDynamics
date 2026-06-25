"""Per-flight path length tagged with release number + group, for an interactive
viewer that steps through the learning process (release by release).

Output: flight_length_learning.html  (self-contained, no CDN; inline SVG)
"""

import base64
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

R = 6371.0


def hlen(lat, lon):
    lat = np.radians(lat); lon = np.radians(lon)
    dlat = np.diff(lat); dlon = np.diff(lon)
    a = np.sin(dlat/2)**2 + np.cos(lat[:-1])*np.cos(lat[1:])*np.sin(dlon/2)**2
    return float(np.sum(2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))))


def net_disp(lat, lon):
    la1, lo1, la2, lo2 = map(np.radians, (lat[0], lon[0], lat[-1], lon[-1]))
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return float(2*R*np.arcsin(np.sqrt(min(1.0, a))))


def rec(ds, grp, rel, lat, lon):
    p = hlen(lat, lon); n = net_disp(lat, lon)
    return {"ds": ds, "grp": grp, "rel": rel, "km": round(p, 2),
            "tor": round(p / n, 3) if n > 0.05 else None}


def recs_2011():
    out = []
    for f in glob.glob("data/R*/*/*.csv"):
        parts = f.split("/")
        route = parts[1]
        rel = int(Path(f).stem.split("_")[-1])
        df = pd.read_csv(f, skipinitialspace=True, usecols=["Latitude", "Longitude"])
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna()
        df = df[df["Latitude"].between(45, 60) & df["Longitude"].between(-10, 5)]
        if len(df) > 1:
            out.append(rec("2011", route, rel,
                           df["Latitude"].values, df["Longitude"].values))
    return out


def recs_2021():
    out = []
    for f in glob.glob("2021/data/raw/*.csv"):
        toks = Path(f).stem.replace("_bare", "").split("_")
        if len(toks) < 3 or not toks[-1].isdigit():
            continue
        cond = re.sub(r"gen", "Gen", toks[0], flags=re.I).capitalize()
        cond = re.sub(r"Gen(\d)", r"Gen\1", cond)
        rel = int(toks[-1])
        try:
            a = pd.read_csv(f, header=None, usecols=[0, 1]).values
        except Exception:
            continue
        a = a[(a[:, 0] > 45) & (a[:, 0] < 60) & (a[:, 1] > -10) & (a[:, 1] < 5)]
        if len(a) > 1:
            out.append(rec("2021", cond, rel, a[:, 0], a[:, 1]))
    return out


def main():
    recs = recs_2011()
    print(f"2011: {len(recs)} flights")
    r21 = recs_2021()
    print(f"2021: {len(r21)} flights")
    recs += r21
    pl = Path("figures/flight_length_stats.png")
    img = base64.b64encode(pl.read_bytes()).decode() if pl.exists() else ""
    if not img:
        print("WARN: flight_length_stats.png not found — run flight_length_stats.py")
    html = (HTML.replace("__DATA__", json.dumps(recs, separators=(",", ":")))
                .replace("__POWERLAW_IMG__", img))
    Path("viewers/flight_length_learning.html").write_text(html)
    print(f"Saved flight_length_learning.html "
          f"({Path('flight_length_learning.html').stat().st_size/1e3:.0f} KB)")


HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flight length across the learning process</title>
<style>
  :root{--accent:#3367d6}
  body{margin:0;font-family:system-ui,sans-serif;background:#f4f5f8;color:#1c2230}
  .wrap{max-width:920px;margin:0 auto;padding:22px}
  h1{font-size:19px;margin:0 0 2px}
  .sub{color:#6a7385;font-size:13px;margin-bottom:16px}
  .card{background:#fff;border:1px solid #e3e6ee;border-radius:12px;
    padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,.05);margin-bottom:16px}
  .ctrls{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin-bottom:6px}
  .seg button,.btn{padding:6px 12px;font-size:13px;border:1px solid #c2c8d6;
    background:#f7f8fb;border-radius:7px;cursor:pointer}
  .seg button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
  .btn:hover{background:#eef0f5}
  select{padding:6px 9px;font-size:13px;border:1px solid #c2c8d6;border-radius:7px}
  .relrow{display:flex;align-items:center;gap:12px;margin:14px 0 4px}
  .relrow input[type=range]{flex:1;accent-color:var(--accent)}
  .relbig{font-size:15px;font-weight:600;min-width:150px}
  .stats{display:flex;flex-wrap:wrap;gap:18px;font-size:12.5px;color:#3a4256;margin-top:8px}
  .stats b{color:#0d1220}
  svg{width:100%;max-width:480px;height:auto;display:block;margin:0 auto}
  .axis{stroke:#c7ccd8;stroke-width:1}
  .tick{fill:#8a92a6;font-size:10px}
  .bar{fill:var(--accent);opacity:.85}
  .barref{fill:#c9d2e8;opacity:.6}
  .med{stroke:#e6442e;stroke-width:1.5;stroke-dasharray:4 3}
  .trendline{fill:none;stroke:#3367d6;stroke-width:2}
  .legend{font-size:11.5px;color:#6a7385;margin-top:6px}
  .sw{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:0}
</style></head>
<body><div class="wrap">
  <h1>Flight path length across the learning process</h1>
  <div class="sub">Step through release number and watch the distribution of
    flight lengths evolve. Each flight = one bird on one release.</div>

  <div class="card">
    <div class="ctrls">
      <div class="seg" id="dsSeg"></div>
      <div class="seg" id="metricSeg"></div>
      <label>Group: <select id="grpSel"></select></label>
      <button class="btn" id="play">▶ Play</button>
      <label class="legend"><input type="checkbox" id="refTog" checked>
        show all-releases reference</label>
    </div>
    <div class="relrow">
      <button class="btn" id="prev">← Prev</button>
      <input type="range" id="rel" min="1" max="6" value="1">
      <button class="btn" id="next">Next →</button>
      <span class="relbig" id="relLbl"></span>
    </div>
    <svg id="hist" viewBox="0 0 480 480"></svg>
    <div class="stats" id="stats"></div>
    <div class="legend"><span class="sw" style="background:#3367d6"></span> this release &nbsp;
      <span class="sw" style="background:#c9d2e8"></span> all releases (reference) &nbsp;
      <span class="sw" style="background:#e6442e"></span> median (dashed)</div>
  </div>

  <div class="card">
    <div class="sub" style="margin:0 0 6px">Median flight length vs release
      (the learning curve) — current release highlighted</div>
    <svg id="trend" viewBox="0 0 480 480"></svg>
  </div>

  <div class="card">
    <div class="sub" style="margin:0 0 8px">Power-law analysis (pooled over all
      releases) — log-log CCDF with Clauset MLE fits: flight length (left) and
      tortuosity (right). Tails are heavy-tailed; bulk is set by the homing distance.</div>
    <img src="data:image/png;base64,__POWERLAW_IMG__" alt="power-law CCDF"
      style="width:100%;display:block;border-radius:6px">
  </div>
</div>
<script>
const ALL = __DATA__;
let ds='2011', grp='(all)', rel=1, playing=null, metric='km';
const META={km:{lbl:'flight path length (km, log)',unit:'km',
                ticks:[1,2,5,10,20,50,100,200,500]},
            tor:{lbl:'tortuosity (path ÷ straight-line, log)',unit:'×',
                ticks:[1,1.5,2,3,5,10,20,50,100,200]}};
function val(r){ return r[metric]; }

function subset(){ return ALL.filter(r=>r.ds===ds && (grp==='(all)'||r.grp===grp)); }
function maxRel(){ return Math.max(...subset().map(r=>r.rel)); }
function bins(){ // fixed log bins over the dataset's full range (current metric)
  const xs=ALL.filter(r=>r.ds===ds).map(val).filter(v=>v!=null && v>0);
  const lo=Math.max(metric==='tor'?1:0.5,Math.min(...xs)), hi=Math.max(...xs), n=26;
  return Array.from({length:n+1},(_,i)=>lo*Math.pow(hi/lo,i/n));
}
function hist(vals,edges){ const c=new Array(edges.length-1).fill(0);
  vals.forEach(v=>{for(let i=0;i<c.length;i++) if(v>=edges[i]&&v<edges[i+1]){c[i]++;break;}}); return c; }
function median(a){ if(!a.length)return 0; const s=[...a].sort((x,y)=>x-y),m=s.length>>1;
  return s.length%2?s[m]:(s[m-1]+s[m])/2; }
function pct(a,p){ if(!a.length)return 0; const s=[...a].sort((x,y)=>x-y);
  return s[Math.min(s.length-1,Math.floor(p/100*s.length))]; }

function drawHist(){
  const sub=subset(), edges=bins();
  const cur=sub.filter(r=>r.rel===rel).map(val).filter(v=>v!=null);
  const refCounts=hist(sub.map(val).filter(v=>v!=null),edges).map(c=>c/maxRel());
  const curCounts=hist(cur,edges);
  const showRef=document.getElementById('refTog').checked;
  const W=480,H=480,L=44,Rt=12,T=14,B=46, pw=W-L-Rt, ph=H-T-B;
  const ymax=Math.max(1,...curCounts,...(showRef?refCounts:[0]));
  const x=i=>L+pw*i/(edges.length-1), y=v=>T+ph*(1-v/ymax);
  let s=`<line class="axis" x1="${L}" y1="${T+ph}" x2="${W-Rt}" y2="${T+ph}"/>`;
  s+=`<line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${T+ph}"/>`;
  // bars
  for(let i=0;i<curCounts.length;i++){
    const bx=x(i),bw=x(i+1)-x(i)-1;
    if(showRef) s+=`<rect class="barref" x="${bx}" y="${y(refCounts[i])}" width="${bw}" height="${T+ph-y(refCounts[i])}"/>`;
    s+=`<rect class="bar" x="${bx}" y="${y(curCounts[i])}" width="${bw}" height="${T+ph-y(curCounts[i])}"/>`;
  }
  // median line
  const md=median(cur);
  if(md){ // position on log axis
    const lo=edges[0],hi=edges[edges.length-1];
    const fx=L+pw*Math.log(md/lo)/Math.log(hi/lo);
    s+=`<line class="med" x1="${fx}" y1="${T}" x2="${fx}" y2="${T+ph}"/>`;
    s+=`<text class="tick" x="${fx+3}" y="${T+10}" fill="#e6442e">median ${md.toFixed(2)}${META[metric].unit}</text>`;
  }
  // x ticks (log)
  META[metric].ticks.forEach(t=>{
    const lo=edges[0],hi=edges[edges.length-1]; if(t<lo||t>hi)return;
    const fx=L+pw*Math.log(t/lo)/Math.log(hi/lo);
    s+=`<line class="axis" x1="${fx}" y1="${T+ph}" x2="${fx}" y2="${T+ph+4}"/>`;
    s+=`<text class="tick" x="${fx}" y="${T+ph+16}" text-anchor="middle">${t}</text>`;
  });
  s+=`<text class="tick" x="${(L+W-Rt)/2}" y="${H-6}" text-anchor="middle">${META[metric].lbl}</text>`;
  s+=`<text class="tick" x="${L-6}" y="${T+8}" text-anchor="end">${Math.round(ymax)}</text>`;
  s+=`<text class="tick" x="${L-6}" y="${T+ph}" text-anchor="end">0</text>`;
  document.getElementById('hist').innerHTML=s;

  const u=META[metric].unit;
  document.getElementById('stats').innerHTML=
    `release <b>${rel}</b> · <b>${cur.length}</b> flights &nbsp;|&nbsp; median <b>${md.toFixed(2)}</b>${u}`+
    ` &nbsp;|&nbsp; IQR ${pct(cur,25).toFixed(2)}–${pct(cur,75).toFixed(2)} &nbsp;|&nbsp;`+
    ` max <b>${(cur.length?Math.max(...cur):0).toFixed(1)}</b>${u}`;
  document.getElementById('relLbl').textContent=`Release ${rel} of ${maxRel()}`;
}

function drawTrend(){
  const sub=subset(), mx=maxRel();
  const meds=[]; for(let r=1;r<=mx;r++){ const v=sub.filter(d=>d.rel===r).map(val).filter(x=>x!=null);
    meds.push(v.length?median(v):null); }
  const W=480,H=480,L=44,Rt=12,T=14,B=40, pw=W-L-Rt, ph=H-T-B;
  const vals=meds.filter(v=>v!=null); const ymax=Math.max(...vals)*1.1, ymin=Math.min(...vals)*0.9;
  const x=r=>L+pw*(r-1)/Math.max(1,mx-1), y=v=>T+ph*(1-(v-ymin)/(ymax-ymin));
  let s=`<line class="axis" x1="${L}" y1="${T+ph}" x2="${W-Rt}" y2="${T+ph}"/>`+
        `<line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${T+ph}"/>`;
  const pts=meds.map((v,i)=>v==null?null:`${x(i+1)},${y(v)}`).filter(Boolean).join(' ');
  s+=`<polyline class="trendline" points="${pts}"/>`;
  meds.forEach((v,i)=>{ if(v==null)return; const r=i+1;
    s+=`<circle cx="${x(r)}" cy="${y(v)}" r="${r===rel?5:3}" fill="${r===rel?'#e6442e':'#3367d6'}"/>`; });
  s+=`<text class="tick" x="${L-6}" y="${T+8}" text-anchor="end">${ymax.toFixed(metric==='tor'?2:0)}</text>`;
  s+=`<text class="tick" x="${L-6}" y="${T+ph}" text-anchor="end">${ymin.toFixed(metric==='tor'?2:0)}${META[metric].unit}</text>`;
  for(let r=1;r<=mx;r+=Math.ceil(mx/12)){ s+=`<text class="tick" x="${x(r)}" y="${T+ph+16}" text-anchor="middle">${r}</text>`; }
  s+=`<text class="tick" x="${(L+W-Rt)/2}" y="${H-4}" text-anchor="middle">release number · median ${metric==='tor'?'tortuosity':'length'}</text>`;
  document.getElementById('trend').innerHTML=s;
}

function refresh(){ drawHist(); drawTrend(); }

function setDS(d){ ds=d; grp='(all)';
  [...document.querySelectorAll('#dsSeg button')].forEach(b=>b.classList.toggle('on',b.dataset.d===d));
  const groups=['(all)',...new Set(ALL.filter(r=>r.ds===d).map(r=>r.grp))];
  const gs=document.getElementById('grpSel'); gs.innerHTML='';
  groups.forEach(g=>{const o=document.createElement('option');o.value=g;o.text=g;gs.add(o);});
  rel=1; const sl=document.getElementById('rel'); sl.max=maxRel(); sl.value=1; refresh();
}
const seg=document.getElementById('dsSeg');
['2011','2021'].forEach(d=>{const b=document.createElement('button');b.textContent=d;b.dataset.d=d;
  b.onclick=()=>setDS(d);seg.appendChild(b);});
const mseg=document.getElementById('metricSeg');
[['km','Length'],['tor','Tortuosity']].forEach(([m,lab])=>{
  const b=document.createElement('button');b.textContent=lab;b.dataset.m=m;
  b.classList.toggle('on',m===metric);
  b.onclick=()=>{metric=m;
    [...mseg.children].forEach(x=>x.classList.toggle('on',x.dataset.m===m));refresh();};
  mseg.appendChild(b);});
document.getElementById('grpSel').onchange=e=>{grp=e.target.value;
  const sl=document.getElementById('rel');sl.max=maxRel();if(rel>maxRel()){rel=maxRel();sl.value=rel;}refresh();};
const sl=document.getElementById('rel');
sl.oninput=e=>{rel=+e.target.value;refresh();};
document.getElementById('prev').onclick=()=>{rel=Math.max(1,rel-1);sl.value=rel;refresh();};
document.getElementById('next').onclick=()=>{rel=Math.min(maxRel(),rel+1);sl.value=rel;refresh();};
document.getElementById('refTog').onchange=refresh;
document.getElementById('play').onclick=function(){
  if(playing){clearInterval(playing);playing=null;this.textContent='▶ Play';}
  else{this.textContent='⏸ Pause';playing=setInterval(()=>{
    rel=rel>=maxRel()?1:rel+1;sl.value=rel;refresh();},700);}
};
setDS('2011');
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
