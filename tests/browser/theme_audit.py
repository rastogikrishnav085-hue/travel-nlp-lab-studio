"""Rendered-contrast audit: loads each page in real Chromium and measures every visible text element and control border.

Usage (app must be running):  python tests/browser/theme_audit.py "<theme>[,<theme>]" Light,Dark,SysLight,SysDark all
SysLight / SysDark = System appearance with the operating system set to light / dark. Exit code 1 if anything fails."""
import asyncio, json, sys, urllib.parse
from playwright.async_api import async_playwright

import os
BASE = os.environ.get("STUDIO_URL", "http://127.0.0.1:8501").rstrip("/")
PAGES = ["","overview","model-lab","data-colab","dataset","intents","models","confusion","predictions","errors","confidence","compare","files"]

JS = r"""
() => {
  const parse = c => { const m = c.match(/rgba?\(([^)]+)\)/); if(!m) return [0,0,0,0];
    const p = m[1].split(/[ ,\/]+/).filter(Boolean).map(Number); return [p[0],p[1],p[2], p.length>3?p[3]:1]; };
  const comp = (top, bot) => { const a = top[3]; return [top[0]*a+bot[0]*(1-a), top[1]*a+bot[1]*(1-a), top[2]*a+bot[2]*(1-a), 1]; };
  const lum = c => { const f = v => { v/=255; return v<=0.03928? v/12.92 : Math.pow((v+0.055)/1.055,2.4)}; return 0.2126*f(c[0])+0.7152*f(c[1])+0.0722*f(c[2]); };
  const ratio = (a,b) => { const l1=lum(a), l2=lum(b); return (Math.max(l1,l2)+0.05)/(Math.min(l1,l2)+0.05); };
  const gradStops = el => { let e=el; while(e && e.nodeType===1){ const bi=getComputedStyle(e).backgroundImage; if(bi && bi.includes('linear-gradient') && !bi.includes('radial')){ const m=[...bi.matchAll(/rgba?\([^)]+\)/g)].map(x=>parse(x[0])).filter(c=>c[3]>0.5); if(m.length) return m; } e=e.parentElement; } return null; };
  const bgOf = el => { const layers=[]; let e=el; while(e && e.nodeType===1){ const cs=getComputedStyle(e); const c=parse(cs.backgroundColor);
      if(e!==el){ const r=e.getBoundingClientRect(), q=el.getBoundingClientRect(), cx=q.x+q.width/2, cy=q.y+q.height/2;
        if(r.width<200 && r.height<200 && (cx<r.left||cx>r.right||cy<r.top||cy>r.bottom)){ e=e.parentElement; continue; } }   // text drawn outside this small box is not on its colour
      if(c[3]>0) layers.push(c); if(c[3]>=1) break; e=e.parentElement; }
    let cur = layers.length && layers[layers.length-1][3]>=1 ? layers.pop() : [255,255,255,1];
    while(layers.length){ cur = comp(layers.pop(), cur); } return cur; };
  const out=[]; const seen=new Set(); let nChecked=0;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n; while(n = walker.nextNode()){
    const t = n.nodeValue.trim(); if(!t) continue; nChecked++;
    const el = n.parentElement; if(!el || ['SCRIPT','STYLE','NOSCRIPT'].includes(el.tagName)) continue;
    const r = el.getBoundingClientRect(); if(r.width<2||r.height<2) continue;
    const cs = getComputedStyle(el); if(cs.visibility==='hidden'||cs.display==='none') continue;
    let op=1, e=el; while(e){ op*=parseFloat(getComputedStyle(e).opacity); e=e.parentElement; }
    if(op<0.05) continue;
    const isSvg = el instanceof SVGElement;
    let fg = parse(isSvg? cs.fill : cs.color);
    if(cs.webkitTextFillColor && cs.webkitTextFillColor.startsWith('rgb')) fg = parse(cs.webkitTextFillColor);
    if(fg[3]===0 || (cs.backgroundClip==='text')) continue;
    fg[3]*=op;
    const target = isSvg? (el.closest('svg')?.parentElement||el) : el;
    const stops = isSvg? null : gradStops(target);
    const bgs = stops && stops.length ? stops.map(c=>[c[0],c[1],c[2],1]) : [bgOf(target)];
    let cr = 99, bg = bgs[0], f = fg;
    for(const b of bgs){ const ff = comp(fg, b); const r_ = ratio(ff,b); if(r_<cr){ cr=r_; bg=b; f=ff; } }
    const px = parseFloat(cs.fontSize), bold = parseInt(cs.fontWeight)>=700;
    const large = px>=24 || (px>=18.66 && bold);
    const need = large?3:4.5;
    if(cr < need){
      const key = el.tagName+'.'+(el.className&&el.className.baseVal!==undefined?el.className.baseVal:el.className)+'|'+t.slice(0,30);
      if(seen.has(key)) continue; seen.add(key);
      let path=[]; e=el; for(let i=0;i<4&&e&&e.tagName!=='BODY';i++){ path.push(e.tagName.toLowerCase()+(e.getAttribute('data-testid')?'['+e.getAttribute('data-testid')+']':'')); e=e.parentElement; }
      out.push({text:t.slice(0,40), cr:+cr.toFixed(2), need, fg:f.slice(0,3).map(Math.round).join(','), bg:bg.slice(0,3).map(Math.round).join(','), path:path.join('<')});
    }
  }
  // non-text contrast of control boundaries (WCAG 1.4.11): border or fill must reach 3:1 against the surrounding colour
  const ctl=[]; const sels=['[data-testid="stSelectbox"] [role="group"]','[data-testid="stTextInputRootElement"]','[data-testid="stFileUploaderDropzone"]','button[data-testid^="stBaseButton-secondary"]','[data-testid="stButtonGroup"] button','[data-testid="stExpander"] details'];
  for(const sel of sels){ for(const el of document.querySelectorAll(sel)){ const r=el.getBoundingClientRect(); if(r.width<4||r.height<4) continue;
      const cs=getComputedStyle(el); const parentBg=bgOf(el.parentElement);
      const bc=parse(cs.borderTopColor); const bw=parseFloat(cs.borderTopWidth);
      const own=parse(cs.backgroundColor);
      let best=0; if(bw>=1 && bc[3]>0.5) best=Math.max(best, ratio(comp(bc,parentBg),parentBg));
      if(own[3]>0.9) best=Math.max(best, ratio(own,parentBg));
      if(best<3){ ctl.push({sel, ratio:+best.toFixed(2), text:(el.innerText||'').slice(0,25)}); } } }
  return {fails: out, n: nChecked, controls: ctl};
}
"""

async def settle(pg):
    await pg.wait_for_timeout(500)
    for _ in range(80):
        if await pg.locator('[data-testid="stStatusWidget"]').count()==0: break
        await pg.wait_for_timeout(250)
    await pg.wait_for_timeout(900)

async def audit_page(pg, theme, appearance, page):
    url = f"{BASE}/{page}?theme={urllib.parse.quote(theme)}&appearance={appearance}"
    await pg.goto(url, wait_until="domcontentloaded")
    await pg.wait_for_selector('[data-testid="stMain"]', timeout=30000)
    await pg.wait_for_selector('[data-testid="stSidebarNavLink"]', timeout=60000)
    await pg.wait_for_selector('.nf-h1', timeout=90000)          # the page's own heading: proves the page really rendered
    await settle(pg)
    # every chart must be drawn before measuring
    for _ in range(60):
        pend = await pg.evaluate("()=>[...document.querySelectorAll('[data-testid=\"stPlotlyChart\"]')].filter(e=>!e.querySelector('.main-svg')).length")
        if pend==0: break
        await pg.wait_for_timeout(300)
    await pg.wait_for_timeout(400)
    res = await pg.evaluate(JS)
    return res

async def run(themes, modes, pages, shots=None, verbose=True):
    total = {}
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--no-sandbox"])
        for theme in themes:
            for label, appearance, scheme in modes:
                ctx = await b.new_context(viewport={"width":1440,"height":900}, color_scheme=scheme, reduced_motion="reduce")
                pg = await ctx.new_page()
                for page in pages:
                    full = await audit_page(pg, theme, appearance, page)
                    res = full['fails']
                    if full['n'] < 15: res = res + [{'text':'PAGE NOT RENDERED','cr':0,'need':0,'fg':'-','bg':'-','path':'body'}]
                    for c in full['controls']:
                        res.append({'text':c['text'] or c['sel'],'cr':c['ratio'],'need':3,'fg':'ctl','bg':'ctl','path':'CONTROL '+c['sel']})
                    total[(theme,label,page)] = res
                    if shots:
                        await pg.screenshot(path=f"{shots}/{theme.replace(' ','_')}_{label}_{page}.png")
                    if verbose:
                        print(f"{theme:14s} {label:12s} {page or 'home':11s} n={full['n']:3d} fails={len(res)}")
                        grp = {}
                        for r in res:
                            k = r['path'].split('<')[0] + ' <' + '<'.join(r['path'].split('<')[1:3]) + f" (fg {r['fg']} on {r['bg']})"
                            grp.setdefault(k, []).append(r)
                        for k,v in list(grp.items())[:int(__import__('os').environ.get('MAXG','8'))]:
                            print(f"      {len(v):3d}x min {min(x['cr'] for x in v)} {k} e.g. '{v[0]['text']}'")
                await ctx.close()
        await b.close()
    return total

if __name__=="__main__":
    themes = sys.argv[1].split(",")
    modeargs = sys.argv[2].split(",")
    M = {"Light":("Light","Light","light"),"Dark":("Dark","Dark","dark"),"SysLight":("System-OSlight","System","light"),"SysDark":("System-OSdark","System","dark")}
    pages = sys.argv[3].split(",") if len(sys.argv)>3 and sys.argv[3]!="all" else PAGES
    shots = sys.argv[4] if len(sys.argv)>4 else None
    res = asyncio.run(run(themes,[M[m] for m in modeargs],pages,shots))
    bad = sum(len(v) for v in res.values())
    print(f'TOTAL failures: {bad}')
    sys.exit(1 if bad else 0)
