"""
helpers/html_utils.py
─────────────────────
Builds a self-contained interactive HTML page for PMUT prediction results.

All prediction data is embedded as a compact JSON object in a <script> block.
Plotly.js (CDN) renders figures at runtime based on user selections:

  • Model checkboxes  — toggle RF / GB / MLP on every visible plot
  • PMUT dropdown     — show all PMUTs or jump to a single one
  • Approach radio    — Both / Vector / Pointwise
  • Scope radio       — Both / Full Spectrum / FWHM ROI
  • Regime radio      — (07 only) All / Regime 1 / Regime 2 / Regime 3
"""

import json


# ── Styles ────────────────────────────────────────────────────────────────────
_CSS = """
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;max-width:1440px;
     margin:0 auto;padding:20px 24px;background:#F8F9FA;color:#212121}
h1{color:#1565C0;margin:0 0 4px}
.subtitle{color:#666;margin:0 0 18px;font-size:.9em;line-height:1.5}

/* ── controls ── */
.controls-panel{background:#fff;border:1px solid #E0E0E0;border-radius:10px;
  padding:14px 20px;margin-bottom:22px;box-shadow:0 1px 4px rgba(0,0,0,.07)}
.ctrl-row{display:flex;flex-wrap:wrap;gap:14px 28px;align-items:flex-start}
.ctrl-grp{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.ctrl-lbl{font-weight:700;font-size:.78em;color:#555;white-space:nowrap;
  text-transform:uppercase;letter-spacing:.4px;min-width:58px}
.chip{display:inline-flex;align-items:center;gap:4px;padding:4px 11px;
  border-radius:20px;background:#F5F5F5;border:1px solid #DDD;
  cursor:pointer;font-size:.82em;user-select:none;transition:background .12s}
.chip:hover{background:#EBEBEB}
.chip input{margin:0;cursor:pointer}
.chip.model-chip{border-color:var(--c,#999)}
.chip.model-chip:has(input:checked){background:color-mix(in srgb,var(--c)18%,white);
  border-color:var(--c)}
select{padding:5px 9px;border-radius:6px;border:1px solid #CCC;
  font-size:.85em;background:#fff;cursor:pointer}

/* ── PMUT cards ── */
.pmut-card{background:#fff;border:1px solid #E0E0E0;border-radius:10px;
  padding:16px 20px;margin-bottom:22px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.cross-card{background:#FFFDE7;border-color:#F9A825}
.cross-banner{background:#F57F17;color:#fff;padding:3px 12px;border-radius:5px;
  display:inline-block;font-size:.79em;font-weight:700;margin-bottom:8px;
  letter-spacing:.3px}
.card-hdr{display:flex;align-items:baseline;gap:14px;margin-bottom:12px;
  flex-wrap:wrap}
.card-hdr h2{margin:0;font-size:1.1em;color:#1565C0}
.card-meta{font-size:.81em;color:#777}

/* ── plots grid ── */
.plots-grid{display:grid;grid-template-columns:repeat(2,1fr);
  gap:10px;margin-bottom:10px}
.plot-cell{min-height:370px}
.plot-div{width:100%;height:370px}

/* ── score table ── */
.score-sec{margin-top:8px}
.score-sec summary{cursor:pointer;font-size:.82em;color:#666;font-weight:600;
  padding:3px 0}
.score-tbl{border-collapse:collapse;font-size:.79em;margin-top:6px;width:auto}
.score-tbl th{background:#37474F;color:#fff;padding:5px 10px;text-align:right;
  white-space:nowrap}
.score-tbl th:nth-child(-n+2){text-align:left}
.score-tbl td{padding:4px 10px;text-align:right;border-bottom:1px solid #EEE}
.score-tbl td:nth-child(-n+2){text-align:left}
.score-tbl tr:nth-child(even){background:#F9F9F9}
.model-cell{font-weight:700}
.r2-cell{font-family:monospace;font-size:.92em}
"""


# ── JavaScript (uses __DATA_JSON__ placeholder) ────────────────────────────
_JS = r"""
const DATA   = __DATA_JSON__;
const COLORS = {RF:'#1565C0',GB:'#E65100',MLP:'#6A1B9A',PINN:'#E91E63','MDOF-PINN':'#8E24AA','PINN-Informed-Loss':'#D81B60','RF-MDOF-PINN':'#00796B',Actual:'#2E7D32'};
const RENDERED = new Set();

/* populate PMUT dropdown */
(function(){
  const sel = document.getElementById('sel-pmut');
  Object.keys(DATA).map(Number).sort((a,b)=>a-b).forEach(n=>{
    const o = document.createElement('option');
    o.value = n; o.text = 'PMUT '+n; sel.appendChild(o);
  });
})();

function getModels(){
  return ['RF','GB','MLP','PINN','MDOF-PINN','PINN-Informed-Loss','RF-MDOF-PINN','Theoretical','BVD','MDOF'].filter(m=>{
    const el = document.getElementById('cb-'+m);
    return el ? el.checked : false;
  });
}




/* build Plotly trace array for one approach×scope combination */
function buildTraces(n, approach, scope){
  const pd = DATA[String(n)];
  const sel = getModels();
  let freqs=[...pd.freqs], actual=[...pd.actual];
  if(scope==='ROI'){
    const mk = freqs.map(f=>f>=pd.fwhm_lo&&f<=pd.fwhm_hi);
    freqs  = freqs.filter((_,i)=>mk[i]);
    actual = actual.filter((_,i)=>mk[i]);
  }
  const traces = [{
    x:freqs, y:actual, name:'Actual',
    mode:'lines', line:{color:COLORS.Actual,width:2.5},
    hovertemplate:'%{y:.5f}<extra>Actual</extra>',
    meta:{model:'Actual'}
  }];
  if(pd.theoretical){
    let tf=[...pd.freqs], th=[...pd.theoretical];
    if(scope==='ROI'){
      const mk = tf.map(f=>f>=pd.fwhm_lo&&f<=pd.fwhm_hi);
      tf = tf.filter((_,i)=>mk[i]);
      th = th.filter((_,i)=>mk[i]);
    }
    traces.push({
      x:tf, y:th, name:'Theoretical (True SDOF)',
      mode:'lines', line:{color:'#00897B',width:2,dash:'dash'},
      visible: sel.includes('Theoretical') ? true : 'legendonly',
      hovertemplate:'%{y:.5f}<extra>Theoretical SDOF</extra>',
      meta:{model:'Theoretical'}
    });
  }
  if(pd.bvd){
    let bf=[...pd.freqs], bv=[...pd.bvd];
    if(scope==='ROI'){
      const mk = bf.map(f=>f>=pd.fwhm_lo&&f<=pd.fwhm_hi);
      bf = bf.filter((_,i)=>mk[i]);
      bv = bv.filter((_,i)=>mk[i]);
    }
    traces.push({
      x:bf, y:bv, name:'BVD Equivalent Circuit',
      mode:'lines', line:{color:'#00ACC1',width:2,dash:'dot'},
      visible: sel.includes('BVD') ? true : 'legendonly',
      hovertemplate:'%{y:.5f}<extra>BVD Model</extra>',
      meta:{model:'BVD'}
    });
  }
  if(pd.mdof){
    let mf=[...pd.freqs], mv=[...pd.mdof];
    if(scope==='ROI'){
      const mk = mf.map(f=>f>=pd.fwhm_lo&&f<=pd.fwhm_hi);
      mf = mf.filter((_,i)=>mk[i]);
      mv = mv.filter((_,i)=>mk[i]);
    }
    traces.push({
      x:mf, y:mv, name:'MDOF Mechanical Model',
      mode:'lines', line:{color:'#2E7D32',width:2,dash:'dashdot'},
      visible: sel.includes('MDOF') ? true : 'legendonly',
      hovertemplate:'%{y:.5f}<extra>MDOF Model</extra>',
      meta:{model:'MDOF'}
    });
  }
  for(const model of ['RF','GB','MLP','PINN','MDOF-PINN','PINN-Informed-Loss','RF-MDOF-PINN']){
    const apd = pd[approach]; if(!apd||!apd[model]) continue;
    const md = apd[model];
    let mf=[...pd.freqs], pr=[...md.pred];
    if(scope==='ROI'){
      const mk = mf.map(f=>f>=pd.fwhm_lo&&f<=pd.fwhm_hi);
      mf = mf.filter((_,i)=>mk[i]);
      pr = pr.filter((_,i)=>mk[i]);
    }
    const mae = scope==='Full' ? md.mae_full : md.mae_roi;
    const r2  = scope==='Full' ? md.r2_full  : md.r2_roi;
    const ms  = mae!=null ? mae.toFixed(4) : 'N/A';
    const rs  = r2 !=null ? (r2>=0?'+':'')+r2.toFixed(3) : 'N/A';
    traces.push({
      x:mf, y:pr,
      name: model+'  MAE='+ms+'  R²='+rs,
      mode:'lines', line:{color:COLORS[model],width:1.6},
      visible: sel.includes(model) ? true : 'legendonly',
      meta:{model}
    });
  }
  return traces;
}

/* render one approach×scope subplot */
function renderPlot(n, approach, scope){
  const divId = 'plot-'+n+'-'+approach+'-'+scope;
  if(RENDERED.has(divId)) return;
  const pd = DATA[String(n)];
  const traces = buildTraces(n, approach, scope);
  const scLbl = scope==='Full'
    ? 'Full Spectrum'
    : 'FWHM ROI ['+pd.fwhm_lo.toFixed(3)+'–'+pd.fwhm_hi.toFixed(3)+' MHz]';
  Plotly.newPlot(divId, traces, {
    title:{text:approach+' — '+scLbl, font:{size:12}},
    xaxis:{title:'Frequency (MHz)', gridcolor:'#EEEEEE'},
    yaxis:{title:'Amplitude (R)',   gridcolor:'#EEEEEE'},
    height:370, paper_bgcolor:'#fff', plot_bgcolor:'#F8FEFF',
    legend:{orientation:'h', y:-0.30, font:{size:9}},
    margin:{t:40,b:105,l:55,r:15},
    shapes: scope==='Full' ? [{
      type:'rect', xref:'x', yref:'paper',
      x0:pd.fwhm_lo, x1:pd.fwhm_hi, y0:0, y1:1,
      fillcolor:'rgba(0,100,200,.05)', line:{width:0}
    }] : []
  }, {responsive:true, displayModeBar:false});
  RENDERED.add(divId);
}

/* render all 4 approach×scope plots for one PMUT */
function renderCard(n){
  ['Vector','Pointwise'].forEach(ap=>['Full','ROI'].forEach(sc=>renderPlot(n,ap,sc)));
}

/* toggle model visibility on all already-rendered plots */
function updateModels(){
  const sel = getModels();
  RENDERED.forEach(divId=>{
    const el = document.getElementById(divId);
    if(!el||!el.data) return;
    Plotly.restyle(divId,{
      visible: el.data.map(t=>
        t.meta?.model==='Actual' ? true :
        sel.includes(t.meta?.model) ? true : 'legendonly'
      )
    });
  });
}




/* show/hide approach/scope cells and resize visible plots */
function updateLayout(){
  const ap = document.querySelector('input[name="approach"]:checked').value;
  const sc = document.querySelector('input[name="scope"]:checked').value;
  document.querySelectorAll('.plot-cell').forEach(cell=>{
    const ok = (ap==='both'||ap===cell.dataset.approach)
            && (sc==='both'||sc===cell.dataset.scope);
    cell.style.display = ok ? 'block' : 'none';
  });
  const nc = ap==='both' ? 2 : 1;
  document.querySelectorAll('.plots-grid').forEach(g=>{
    g.style.gridTemplateColumns = 'repeat('+nc+',1fr)';
  });
  /* resize only visible plots so they fill their container */
  RENDERED.forEach(id=>{
    const el = document.getElementById(id);
    if(el && el.offsetParent!==null) Plotly.Plots.resize(id);
  });
}

/* filter cards by PMUT + regime selection */
function updateCards(){
  const pv  = document.getElementById('sel-pmut').value;
  const rvEl = document.querySelector('input[name="regime"]:checked');
  const rv  = rvEl ? rvEl.value : 'all';
  document.querySelectorAll('.pmut-card').forEach(card=>{
    const n  = card.dataset.n;
    const rg = card.dataset.regime || 'all';
    const ok = (pv==='all'||pv===n) && (rv==='all'||rv===rg);
    card.style.display = ok ? 'block' : 'none';
  });
}

function onPmutChange(){
  updateCards();
  const pv = document.getElementById('sel-pmut').value;
  if(pv !== 'all'){
    const card = document.getElementById('card-'+pv);
    if(card){
      renderCard(parseInt(pv));
      setTimeout(()=>card.scrollIntoView({behavior:'smooth',block:'start'}),80);
    }
  }
}

/* initial page load — render all cards, then apply filters */
document.addEventListener('DOMContentLoaded',()=>{
  /* temporarily show all cells so Plotly can measure dimensions */
  document.querySelectorAll('.plot-cell').forEach(c=>c.style.display='block');
  document.querySelectorAll('.pmut-card').forEach(card=>{
    renderCard(parseInt(card.dataset.n));
  });
  updateLayout();
  updateCards();
});
"""


# ── HTML builders ─────────────────────────────────────────────────────────────

def _score_row(row):
    approach, model, mae_f, r2_f, mae_r, r2_r = row
    r2f  = (f'+{r2_f:.4f}' if r2_f  is not None and r2_f  >= 0
            else (f'{r2_f:.4f}'  if r2_f  is not None else 'N/A'))
    r2r  = (f'+{r2_r:.4f}' if r2_r  is not None and r2_r  >= 0
            else (f'{r2_r:.4f}'  if r2_r  is not None else 'N/A'))
    maer = f'{mae_r:.5f}' if mae_r is not None else 'N/A'
    return (f'<tr><td>{approach}</td><td class="model-cell">{model}</td>'
            f'<td>{mae_f:.5f}</td><td class="r2-cell">{r2f}</td>'
            f'<td>{maer}</td><td class="r2-cell">{r2r}</td></tr>')


def _physics_sec(n, d):
    pt = d.get("phys_true")
    pe = d.get("phys_est")
    if not pt or not pe:
        return ""
    
    def r(label, key, unit="", fmt=".5f"):
        vt, ve = pt[key], pe[key]
        diff = abs(vt - ve) / vt * 100 if vt > 0 else 0
        u_str = f" {unit}" if unit else ""
        return f"<tr><td class='model-cell'>{label}</td><td>{vt:{fmt}}{u_str}</td><td>{ve:{fmt}}{u_str}</td><td style='color:#D81B60;font-weight:600'>{diff:.1f}%</td></tr>"
        
    rows = [
        r("Mass (m)", "m"),
        r("Resonant Freq (f0)", "f0_MHz", unit="MHz", fmt=".4f"),
        r("Quality Factor (Q)", "Q", fmt=".2f"),
        r("Derived Damping (c_N)", "c"),
        r("Derived Stiffness (k_N)", "k"),
        r("Force Coeff (&alpha;_N)", "alpha"),
        r("Gain (G_N)", "G"),
        r("Background (A_bg)", "A_bg"),
        r("Peak Displ (|X|_peak)", "X_peak", fmt=".5f"),
    ]
    return (
        f'  <details class="score-sec" style="margin-top:10px" open>\n'
        f'    <summary>Physics SDOF Parameters (True vs PINN Estimated) — PMUT {n}</summary>\n'
        f'    <table class="score-tbl">\n'
        f'      <thead><tr><th>Parameter</th><th>True (Direct Fit)</th><th>PINN Estimated</th><th>Difference %</th></tr></thead>\n'
        f'      <tbody>{"".join(rows)}</tbody>\n'
        f'    </table>\n'
        f'  </details>\n'
    )


def _bvd_sec(n, d):
    bp = d.get("bvd_params")
    if not bp:
        return ""
    
    c0 = bp.get("C0", 0.0)
    abg = bp.get("A_bg", 0.0)
    loss = bp.get("loss", 0.0)
    f0 = bp.get("f0", [])
    q = bp.get("Q", [])
    rm = bp.get("R_m", [])
    lm = bp.get("L_m", [])
    cm = bp.get("C_m", [])

    mode_rows = ""
    for idx in range(len(f0)):
        mode_rows += (
            f"<tr><td class='model-cell'>Mode {idx+1}</td>"
            f"<td>{f0[idx]:.4f} MHz</td>"
            f"<td>{q[idx]:.1f}</td>"
            f"<td>{rm[idx]:.4e} &Omega;</td>"
            f"<td>{lm[idx]:.4e} H</td>"
            f"<td>{cm[idx]:.4e} F</td></tr>"
        )

    return (
        f'  <details class="score-sec" style="margin-top:10px" open>\n'
        f'    <summary>Butterworth-Van Dyke (BVD) Circuit Parameters — PMUT {n} (Fit Loss = {loss:.4e})</summary>\n'
        f'    <div style="font-size:0.82em;margin:6px 0;color:#444"><b>Static Parallel Capacitance (C<sub>0</sub>):</b> {c0:.4e} F &nbsp;|&nbsp; <b>Background Offset (A<sub>bg</sub>):</b> {abg:.4f}</div>\n'
        f'    <table class="score-tbl">\n'
        f'      <thead><tr><th>Motional Branch</th><th>Resonant Freq (f<sub>0</sub>)</th><th>Quality Factor (Q)</th><th>Resistance (R<sub>m</sub>)</th><th>Inductance (L<sub>m</sub>)</th><th>Capacitance (C<sub>m</sub>)</th></tr></thead>\n'
        f'      <tbody>{mode_rows}</tbody>\n'
        f'    </table>\n'
        f'  </details>\n'
    )


def _mdof_sec(n, d):
    mp = d.get("mdof_params")
    if not mp:
        return ""
    
    abg = mp.get("A_bg", 0.0)
    s_factor = mp.get("S", 1.0)
    loss = mp.get("loss", 0.0)
    f0 = mp.get("f0", [])
    q = mp.get("Q", [])
    k_val = mp.get("k", [])
    c_val = mp.get("c", [])
    alpha = mp.get("alpha", [])

    mode_rows = ""
    for idx in range(len(f0)):
        mode_rows += (
            f"<tr><td class='model-cell'>Mode {idx+1}</td>"
            f"<td>{f0[idx]:.4f} MHz</td>"
            f"<td>{q[idx]:.1f}</td>"
            f"<td>{k_val[idx]:.4e}</td>"
            f"<td>{c_val[idx]:.4e}</td>"
            f"<td>{alpha[idx]:.4e}</td></tr>"
        )

    return (
        f'  <details class="score-sec" style="margin-top:10px" open>\n'
        f'    <summary>Multi-Mode Mechanical Superposition (MDOF) Parameters — PMUT {n} (Fit Loss = {loss:.4e})</summary>\n'
        f'    <div style="font-size:0.82em;margin:6px 0;color:#444"><b>Baseline Offset (A<sub>bg</sub>):</b> {abg:.4f} &nbsp;|&nbsp; <b>Scale Factor (S):</b> {s_factor:.4f}</div>\n'
        f'    <table class="score-tbl">\n'
        f'      <thead><tr><th>Mechanical Mode</th><th>Resonant Freq (f<sub>0</sub>)</th><th>Quality Factor (Q)</th><th>Stiffness (k)</th><th>Damping (c)</th><th>Force Coeff (&alpha;)</th></tr></thead>\n'
        f'      <tbody>{mode_rows}</tbody>\n'
        f'    </table>\n'
        f'  </details>\n'
    )


def _card(n, d, regime=None, is_cross=False):
    train_str    = ', '.join(map(str, d['train']))
    regime_attr  = f'data-regime="{regime}"' if regime is not None else ''
    cross_banner = '<div class="cross-banner">★ Cross-Regime</div>' if is_cross else ''
    cross_cls    = ' cross-card' if is_cross else ''
    score_rows   = '\n'.join(_score_row(r) for r in d.get('score_rows', []))
    phys_sec     = _physics_sec(n, d)
    bvd_sec      = _bvd_sec(n, d)
    mdof_sec     = _mdof_sec(n, d)
    return (
        f'<div class="pmut-card{cross_cls}" data-n="{n}" id="card-{n}" {regime_attr}>\n'
        f'  {cross_banner}\n'
        f'  <div class="card-hdr">'
        f'<h2>PMUT {n}</h2>'
        f'<span class="card-meta">Train: [{train_str}] → Predict PMUT {n}'
        f' &nbsp;|&nbsp; FWHM: {d["fwhm_lo"]:.3f}–{d["fwhm_hi"]:.3f} MHz</span>'
        f'</div>\n'
        f'  <div class="plots-grid" id="grid-{n}">\n'
        f'    <div class="plot-cell" data-approach="Vector"    data-scope="Full">'
        f'<div id="plot-{n}-Vector-Full"    class="plot-div"></div></div>\n'
        f'    <div class="plot-cell" data-approach="Pointwise" data-scope="Full">'
        f'<div id="plot-{n}-Pointwise-Full" class="plot-div"></div></div>\n'
        f'    <div class="plot-cell" data-approach="Vector"    data-scope="ROI">'
        f'<div id="plot-{n}-Vector-ROI"     class="plot-div"></div></div>\n'
        f'    <div class="plot-cell" data-approach="Pointwise" data-scope="ROI">'
        f'<div id="plot-{n}-Pointwise-ROI"  class="plot-div"></div></div>\n'
        f'  </div>\n'
        f'  <details class="score-sec">\n'
        f'    <summary>Score Table — PMUT {n}</summary>\n'
        f'    <table class="score-tbl">\n'
        f'      <thead><tr><th>Approach</th><th>Model</th>'
        f'<th>Full MAE</th><th>Full R²</th>'
        f'<th>ROI MAE</th><th>ROI R²</th></tr></thead>\n'
        f'      <tbody>{score_rows}</tbody>\n'
        f'    </table>\n'
        f'  </details>\n'
        f'{phys_sec}\n'
        f'{bvd_sec}\n'
        f'{mdof_sec}\n'
        f'</div>'
    )




def build_interactive_html(
    title: str,
    subtitle: str,
    all_data: dict,
    regime_map: dict = None,
    cross_pmuts: set = None,
    regime_labels: list = None,
) -> str:
    """
    Build a fully self-contained interactive HTML page.

    Parameters
    ----------
    title         : <h1> text
    subtitle      : descriptive paragraph
    all_data      : {str(n): {train, freqs, actual, fwhm_lo, fwhm_hi,
                               Vector:    {RF:{pred,mae_full,r2_full,mae_roi,r2_roi}, GB:…, MLP:…},
                               Pointwise: same structure,
                               score_rows: [[approach,model,mae_f,r2_f,mae_r,r2_r], …]}}
    regime_map    : None → no regime filter; {int n: int regime_id} → adds regime radio
    cross_pmuts   : set of n_pmut values that are cross-regime predictions
    regime_labels : [(value_str, display_label), …] for regime radio options
    """
    cross_pmuts   = cross_pmuts   or set()
    regime_labels = regime_labels or []

    # ── Regime filter (optional) ──────────────────────────────────────────────
    regime_ctrl = ''
    if regime_map is not None:
        opts = '<label class="chip"><input type="radio" name="regime" value="all" checked onchange="updateCards()"> All</label>\n'
        for val, lbl in regime_labels:
            opts += (f'<label class="chip"><input type="radio" name="regime" '
                     f'value="{val}" onchange="updateCards()"> {lbl}</label>\n')
        regime_ctrl = (
            f'<div class="ctrl-grp">'
            f'<span class="ctrl-lbl">Regime</span>\n{opts}</div>'
        )

    controls = f"""<div class="controls-panel">
  <div class="ctrl-row">
    <div class="ctrl-grp">
      <span class="ctrl-lbl">Models</span>
      <label class="chip model-chip" style="--c:#1565C0"><input type="checkbox" id="cb-RF"   checked onchange="updateModels()"> RF</label>
      <label class="chip model-chip" style="--c:#E65100"><input type="checkbox" id="cb-GB"   checked onchange="updateModels()"> GB</label>
      <label class="chip model-chip" style="--c:#6A1B9A"><input type="checkbox" id="cb-MLP"  checked onchange="updateModels()"> MLP</label>
      <label class="chip model-chip" style="--c:#E91E63"><input type="checkbox" id="cb-PINN" checked onchange="updateModels()"> PINN</label>
      <label class="chip model-chip" style="--c:#8E24AA"><input type="checkbox" id="cb-MDOF-PINN" checked onchange="updateModels()"> MDOF-PINN</label>
      <label class="chip model-chip" style="--c:#D81B60"><input type="checkbox" id="cb-PINN-Informed-Loss" checked onchange="updateModels()"> PINN-Informed-Loss</label>
      <label class="chip model-chip" style="--c:#00796B"><input type="checkbox" id="cb-RF-MDOF-PINN" checked onchange="updateModels()"> RF-MDOF-PINN</label>
      <label class="chip model-chip" style="--c:#00897B"><input type="checkbox" id="cb-Theoretical" checked onchange="updateModels()"> Theoretical SDOF</label>
      <label class="chip model-chip" style="--c:#00ACC1"><input type="checkbox" id="cb-BVD" checked onchange="updateModels()"> BVD Circuit</label>
      <label class="chip model-chip" style="--c:#2E7D32"><input type="checkbox" id="cb-MDOF" checked onchange="updateModels()"> MDOF Mechanical</label>



    </div>
    <div class="ctrl-grp">
      <span class="ctrl-lbl">PMUT</span>
      <select id="sel-pmut" onchange="onPmutChange()"><option value="all">All PMUTs</option></select>
    </div>
    <div class="ctrl-grp">
      <span class="ctrl-lbl">Approach</span>
      <label class="chip"><input type="radio" name="approach" value="both"      checked onchange="updateLayout()"> Both</label>
      <label class="chip"><input type="radio" name="approach" value="Vector"           onchange="updateLayout()"> Vector</label>
      <label class="chip"><input type="radio" name="approach" value="Pointwise"        onchange="updateLayout()"> Pointwise</label>
    </div>
    <div class="ctrl-grp">
      <span class="ctrl-lbl">Scope</span>
      <label class="chip"><input type="radio" name="scope" value="both" checked onchange="updateLayout()"> Both</label>
      <label class="chip"><input type="radio" name="scope" value="Full"         onchange="updateLayout()"> Full</label>
      <label class="chip"><input type="radio" name="scope" value="ROI"          onchange="updateLayout()"> FWHM ROI</label>
    </div>
    {regime_ctrl}
  </div>
</div>"""

    cards = '\n'.join(
        _card(int(n), all_data[n],
              regime   = regime_map.get(int(n)) if regime_map else None,
              is_cross = int(n) in cross_pmuts)
        for n in sorted(all_data.keys(), key=int)
    )

    data_json = json.dumps(all_data, ensure_ascii=False, separators=(',', ':'))
    js        = _JS.replace('__DATA_JSON__', data_json)

    return (
        f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        f'<meta charset="utf-8">\n'
        f'<title>{title}</title>\n'
        f'<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
        f'<style>{_CSS}</style>\n'
        f'</head>\n<body>\n'
        f'<h1>{title}</h1>\n'
        f'<p class="subtitle">{subtitle}</p>\n'
        f'{controls}\n'
        f'<div id="plots-container">\n{cards}\n</div>\n'
        f'<script>{js}</script>\n'
        f'</body>\n</html>'
    )
