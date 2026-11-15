"""CyberX-style UI theme + animations for MailGuard ML.

Injected once per page run from ``app.py`` via :func:`inject_styles`. Pure CSS / inline SVG
(no JavaScript, no external images), so it works unchanged on Streamlit Community Cloud.
All motion is disabled for users who set ``prefers-reduced-motion``.

Public helpers
--------------
inject_styles()      global theme, animated background, HUD effects
command_center(...)  animated "sources -> core -> verdict" pipeline panel
meter(...)           animated probability bar
stamp(...)           SPAM / HAM rubber stamp
"""

from __future__ import annotations

import base64
import math

import streamlit as st

# =========================================================================== global CSS
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --bg: #04040D;
  --line: rgba(255, 255, 255, .08);
  --indigo: #4F46E5;
  --blue: #3B3BFF;
  --blue-soft: #8EA2FF;
  --cyan: #22D3EE;
  --violet: #8B5CF6;
  --red: #F43F5E;
  --teal: #2DD4BF;
  --ease: cubic-bezier(.22, 1, .36, 1);
}

html, body, .stApp, [class*="css"] { font-family: 'Outfit', system-ui, sans-serif !important; }
code, pre, kbd, .mono { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }

/* ---------------------------------------------------------------- keyframes */
@keyframes fade-up    { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: none; } }
@keyframes fade-in    { from { opacity: 0; } to { opacity: 1; } }
@keyframes slide-in   { from { opacity: 0; transform: translateX(-26px); } to { opacity: 1; transform: none; } }
@keyframes slide-left { from { opacity: 0; transform: translateX(26px); } to { opacity: 1; transform: none; } }
@keyframes pop        { 0% { opacity: 0; transform: scale(.9); } 60% { transform: scale(1.03); } 100% { opacity: 1; transform: scale(1); } }
@keyframes shimmer    { to { background-position: 200% center; } }
@keyframes float      { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@keyframes spin       { to { transform: rotate(360deg); } }
@keyframes pulse-ring { 0% { box-shadow: 0 0 0 0 rgba(79,70,229,.7); } 70% { box-shadow: 0 0 0 16px rgba(79,70,229,0); } 100% { box-shadow: 0 0 0 0 rgba(79,70,229,0); } }
@keyframes glow-red   { 0%,100% { box-shadow: 0 0 0 rgba(244,63,94,0); } 50% { box-shadow: 0 0 32px rgba(244,63,94,.5); } }
@keyframes glow-blue  { 0%,100% { box-shadow: 0 0 0 rgba(59,59,255,0); } 50% { box-shadow: 0 0 32px rgba(59,59,255,.55); } }
@keyframes shake      { 0%,100% { transform: translateX(0); } 20% { transform: translateX(-6px); } 40% { transform: translateX(6px); } 60% { transform: translateX(-4px); } 80% { transform: translateX(4px); } }
@keyframes sweep      { 0% { transform: translateX(-120%); } 100% { transform: translateX(320%); } }
@keyframes bar-grow   { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes border-run { 0% { background-position: 0% 50%; } 100% { background-position: 300% 50%; } }
@keyframes blink      { 50% { opacity: 0; } }
@keyframes ticker     { from { transform: translateX(0); } to { transform: translateX(-50%); } }
@keyframes drift      { 0% { transform: translateY(105vh) translateX(0); opacity: 0; } 10% { opacity: .85; } 90% { opacity: .85; } 100% { transform: translateY(-10vh) translateX(var(--x, 30px)); opacity: 0; } }
@keyframes vscan      { 0% { transform: translateY(-20vh); opacity: 0; } 10% { opacity: 1; } 90% { opacity: 1; } 100% { transform: translateY(110vh); opacity: 0; } }
@keyframes stamp      { 0% { opacity: 0; transform: scale(3) rotate(-22deg); } 55% { opacity: 1; transform: scale(.9) rotate(-9deg); } 75% { transform: scale(1.08) rotate(-12deg); } 100% { opacity: 1; transform: scale(1) rotate(-10deg); } }
@keyframes underline  { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes flicker    { 0%,19%,21%,23%,100% { opacity: 1; } 20%,22% { opacity: .55; } }
@keyframes hud        { 0%,100% { opacity: .55; } 50% { opacity: 1; } }
@keyframes skeleton   { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
@property --n { syntax: '<integer>'; initial-value: 0; inherits: false; }
@keyframes count      { from { --n: 0; } to { --n: var(--t); } }

/* ---------------------------------------------------------------- app background */
.stApp {
  background:
    radial-gradient(1100px 560px at 55% -8%, rgba(67, 56, 202, .60), transparent 65%),
    radial-gradient(700px 420px at 100% 30%, rgba(30, 27, 140, .30), transparent 70%),
    radial-gradient(700px 420px at 0% 80%, rgba(34, 211, 238, .07), transparent 70%),
    var(--bg) !important;
}
[data-testid="stAppViewContainer"] { position: relative; z-index: 1; background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }

.mg-bg { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
.mg-dots {
  position: absolute; inset: -40px; opacity: .7; background-image: var(--img-dots); background-size: 36px 36px;
  -webkit-mask-image: radial-gradient(ellipse 95% 85% at 50% 35%, #000 25%, transparent 100%);
          mask-image: radial-gradient(ellipse 95% 85% at 50% 35%, #000 25%, transparent 100%);
}
.mg-bigring {
  position: absolute; right: -22vw; top: 8vh; width: 78vw; height: 78vw; max-width: 1100px; max-height: 1100px;
  background: var(--img-ring) center / contain no-repeat; opacity: .55; animation: spin 240s linear infinite;
}
.mg-bigring.l { right: auto; left: -30vw; top: auto; bottom: -34vw; opacity: .35; animation-direction: reverse; animation-duration: 300s; }
.mg-particle {
  position: absolute; bottom: 0; border-radius: 50%; width: var(--s, 3px); height: var(--s, 3px);
  background: var(--c, #8EA2FF); box-shadow: 0 0 10px 2px var(--c, #8EA2FF);
  animation: drift var(--d, 20s) linear var(--delay, 0s) infinite;
}
.mg-vscan { position: absolute; left: 0; right: 0; height: 160px; top: 0;
  background: linear-gradient(180deg, transparent, rgba(79,70,229,.10), rgba(34,211,238,.10), transparent);
  animation: vscan 11s linear infinite; }
.mg-topbar { position: fixed; top: 0; left: 0; right: 0; height: 2px; z-index: 999; pointer-events: none;
  background: linear-gradient(90deg, transparent, var(--blue), var(--cyan), var(--violet), transparent); background-size: 200% 100%;
  animation: border-run 4s linear infinite; }

/* ---------------------------------------------------------------- headings */
h1 {
  font-weight: 700 !important; letter-spacing: -0.03em;
  background: linear-gradient(90deg, #FFFFFF 0%, #C7D2FE 30%, var(--blue-soft) 50%, #FFFFFF 75%);
  background-size: 200% auto; -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  animation: fade-up .7s var(--ease) both, shimmer 7s linear infinite;
}
h2, h3 { font-weight: 600 !important; letter-spacing: -0.01em; animation: fade-up .6s var(--ease) both; }
h2::after, h3::after {
  content: ""; display: block; height: 2px; width: 52px; margin-top: 6px; border-radius: 3px; transform-origin: left;
  background: linear-gradient(90deg, var(--blue), var(--cyan)); animation: underline .8s .2s var(--ease) both;
}
[data-testid="stSidebar"] h3::after, .cc-title::after { display: none; }
[data-testid="stCaptionContainer"] { animation: fade-in .9s ease both; opacity: .7; }

[data-testid="stMain"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],
[data-testid="stMain"] [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] { animation: fade-up .65s var(--ease) both; }
[data-testid="stVerticalBlock"] > *:nth-child(2) { animation-delay: .05s; } [data-testid="stVerticalBlock"] > *:nth-child(3) { animation-delay: .10s; }
[data-testid="stVerticalBlock"] > *:nth-child(4) { animation-delay: .15s; } [data-testid="stVerticalBlock"] > *:nth-child(5) { animation-delay: .20s; }
[data-testid="stVerticalBlock"] > *:nth-child(6) { animation-delay: .25s; } [data-testid="stVerticalBlock"] > *:nth-child(n+7) { animation-delay: .30s; }

/* ---------------------------------------------------------------- sidebar (pill nav, like the reference) */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(8,7,30,.92), rgba(4,4,13,.96)) !important;
  border-right: 1px solid var(--line); backdrop-filter: blur(16px); animation: slide-in .6s var(--ease) both;
}
[data-testid="stSidebar"] h3 span[data-testid="stIconMaterial"], [data-testid="stSidebar"] h3 .material-symbols-rounded {
  color: #fff; background: linear-gradient(135deg, var(--blue), var(--indigo)); padding: 6px; border-radius: 10px;
  animation: pulse-ring 2.6s infinite, float 4s ease-in-out infinite;
}
[data-testid="stSidebarNav"] a, [data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
  border-radius: 999px; border: 1px solid transparent; margin: 3px 0; padding: 8px 14px;
  transition: transform .3s var(--ease), background .3s ease, border-color .3s ease, box-shadow .3s ease;
}
[data-testid="stSidebarNav"] a:hover, [data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {
  transform: translateX(6px); background: rgba(79,70,229,.14) !important; border-color: var(--line);
}
[data-testid="stSidebar"] a[aria-current="page"] {
  background: linear-gradient(135deg, rgba(255,255,255,.08), rgba(255,255,255,.02)) !important; border-color: rgba(255,255,255,.14);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.12), 0 8px 24px rgba(59,59,255,.25);
}
[data-testid="stSidebar"] [data-testid="stAlert"] { animation: pop .7s var(--ease) both; }

/* ---------------------------------------------------------------- glass cards + HUD corner brackets */
[data-testid="stMetric"], .stPlotlyChart, [data-testid="stExpander"], [data-testid="stDataFrame"] {
  background: linear-gradient(145deg, rgba(255,255,255,.07), rgba(255,255,255,.015)) !important;
  border: 1px solid var(--line) !important; border-radius: 22px !important; backdrop-filter: blur(12px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.08);
}
[data-testid="stMetric"] {
  position: relative; overflow: hidden; padding: 16px 20px; animation: pop .6s var(--ease) both;
  transition: transform .35s var(--ease), box-shadow .35s ease, border-color .35s ease;
}
[data-testid="stMetric"]::before, [data-testid="stMetric"]::after {  /* HUD corners */
  content: ""; position: absolute; width: 14px; height: 14px; border: 2px solid var(--cyan); opacity: .7; animation: hud 3s ease-in-out infinite;
}
[data-testid="stMetric"]::before { top: 10px; left: 10px; border-right: none; border-bottom: none; border-top-left-radius: 6px; }
[data-testid="stMetric"]::after  { bottom: 10px; right: 10px; border-left: none; border-top: none; border-bottom-right-radius: 6px; }
[data-testid="stMetric"]:hover { transform: translateY(-6px); border-color: rgba(142,162,255,.55) !important; box-shadow: 0 16px 40px rgba(59,59,255,.28); }
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.02em; }
[data-testid="stMetricValue"] > div { transition: transform .3s var(--ease), text-shadow .3s ease; }
[data-testid="stMetric"]:hover [data-testid="stMetricValue"] > div { transform: scale(1.07); text-shadow: 0 0 20px rgba(142,162,255,.8); }
[data-testid="stMetricLabel"] { opacity: .7; text-transform: uppercase; font-size: .7rem; letter-spacing: .12em; }

.stPlotlyChart { padding: 10px; animation: pop .7s var(--ease) both; transition: transform .35s var(--ease), box-shadow .35s ease; }
.stPlotlyChart:hover { transform: translateY(-3px); box-shadow: 0 16px 40px rgba(0,0,0,.45); }
[data-testid="stDataFrame"] { overflow: hidden; transition: box-shadow .3s ease; }
[data-testid="stDataFrame"]:hover { box-shadow: 0 12px 32px rgba(59,59,255,.2); }
[data-testid="stExpander"] { transition: border-color .3s ease, box-shadow .3s ease; }
[data-testid="stExpander"]:hover { border-color: rgba(142,162,255,.45) !important; box-shadow: 0 10px 28px rgba(0,0,0,.35); }
[data-testid="stExpander"] details[open] > div { animation: fade-up .4s var(--ease) both; }

/* ---------------------------------------------------------------- alerts */
[data-testid="stAlert"] { border-radius: 18px; border: 1px solid var(--line); backdrop-filter: blur(8px); animation: slide-in .55s var(--ease) both; }
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]), [data-testid="stAlert"][kind="error"] { animation: slide-in .5s var(--ease) both, shake .6s .4s ease, glow-red 2.6s 1s ease-in-out infinite; }
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]), [data-testid="stAlert"][kind="success"] { animation: slide-in .5s var(--ease) both, glow-blue 2.8s 1s ease-in-out infinite; }

/* ---------------------------------------------------------------- buttons */
.stButton > button, [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
  position: relative; overflow: hidden; border-radius: 999px; font-weight: 600; letter-spacing: .02em;
  transition: transform .25s var(--ease), box-shadow .25s ease;
}
[data-testid="stBaseButton-primary"] { background: linear-gradient(135deg, var(--blue), var(--violet)) !important; border: none !important; color: #fff !important; box-shadow: 0 8px 24px rgba(59,59,255,.35); }
.stButton > button::after { content: ""; position: absolute; top: 0; left: -60%; width: 40%; height: 100%; transform: skewX(-20deg);
  background: linear-gradient(100deg, transparent, rgba(255,255,255,.35), transparent); }
.stButton > button:hover { transform: translateY(-3px) scale(1.03); box-shadow: 0 12px 30px rgba(59,59,255,.5); }
.stButton > button:hover::after { animation: sweep .9s ease; }
.stButton > button:active { transform: scale(.96); }

/* ---------------------------------------------------------------- tabs / inputs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { border-radius: 14px 14px 0 0; padding: 10px 16px; transition: background .25s ease, transform .25s var(--ease); }
.stTabs [data-baseweb="tab"]:hover { background: rgba(79,70,229,.14); transform: translateY(-2px); }
.stTabs [aria-selected="true"] { color: var(--blue-soft) !important; }
.stTabs [data-baseweb="tab-highlight"] { height: 3px; border-radius: 3px; transition: all .4s var(--ease);
  background: linear-gradient(90deg, var(--blue), var(--cyan), var(--violet), var(--blue)) !important; background-size: 300% 100%; animation: border-run 3s linear infinite; }
.stTabs [data-baseweb="tab-panel"] { animation: fade-up .5s var(--ease) both; }

textarea, input, [data-baseweb="select"] > div, [data-baseweb="base-input"] {
  border-radius: 16px !important; background: rgba(255,255,255,.04) !important; transition: box-shadow .3s ease, border-color .3s ease;
}
textarea:focus, input:focus { box-shadow: 0 0 0 3px rgba(59,59,255,.4) !important; border-color: var(--blue-soft) !important; }
[data-testid="stFileUploaderDropzone"] { border: 2px dashed rgba(142,162,255,.4); border-radius: 22px; background: rgba(79,70,229,.06); transition: background .3s ease, transform .3s var(--ease), border-color .3s ease; }
[data-testid="stFileUploaderDropzone"]:hover { background: rgba(79,70,229,.16); transform: scale(1.01); border-color: var(--blue-soft); }
[data-testid="stSlider"] [role="slider"] { transition: transform .2s var(--ease), box-shadow .2s ease; }
[data-testid="stSlider"] [role="slider"]:hover { transform: scale(1.25); box-shadow: 0 0 0 8px rgba(59,59,255,.28); }
[data-testid="stCode"], pre { border-radius: 16px; border: 1px solid var(--line); animation: fade-in .6s ease both; }
hr { border: none !important; height: 1px !important; background: linear-gradient(90deg, transparent, var(--blue-soft), transparent) !important; opacity: .5; }
::selection { background: rgba(79,70,229,.6); color: #fff; }
::-webkit-scrollbar { width: 10px; height: 10px; } ::-webkit-scrollbar-track { background: rgba(8,7,30,.8); }
::-webkit-scrollbar-thumb { background: linear-gradient(var(--blue), var(--violet)); border-radius: 8px; }

[data-testid="stSpinner"] { animation: fade-in .3s ease both; }
[data-testid="stSpinner"] > div::after { content: ""; display: block; height: 5px; margin-top: 8px; border-radius: 6px;
  background: linear-gradient(90deg, rgba(255,255,255,.05), var(--blue-soft), rgba(255,255,255,.05)); background-size: 200% 100%; animation: skeleton 1.2s linear infinite; }

/* ================================================================ COMMAND CENTER (pipeline panel) */
.cc-panel {
  position: relative; overflow: hidden; border-radius: 30px; padding: 22px 26px 26px; margin-bottom: 20px;
  background: linear-gradient(160deg, rgba(49,46,129,.55), rgba(10,9,36,.85) 55%, rgba(4,4,13,.9));
  border: 1px solid rgba(255,255,255,.10); box-shadow: inset 0 1px 0 rgba(255,255,255,.12), 0 30px 80px rgba(20,16,90,.45);
  animation: pop .8s var(--ease) both;
}
.cc-panel::before { content: ""; position: absolute; inset: 0; border-radius: inherit; padding: 1.5px; pointer-events: none;
  background: linear-gradient(120deg, var(--blue), transparent 30%, var(--cyan) 55%, transparent 75%, var(--violet)); background-size: 300% 300%; animation: border-run 7s linear infinite;
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0); -webkit-mask-composite: xor; mask-composite: exclude; opacity: .8; }
.cc-head { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.cc-title { font-size: clamp(1.2rem, 2.2vw, 1.7rem); font-weight: 600; letter-spacing: -.02em; margin: 0; padding: 0; }
.cc-sub { font-family: 'JetBrains Mono', monospace; font-size: .74rem; color: var(--cyan); letter-spacing: .08em; margin-bottom: 4px; animation: flicker 5s infinite; }
.cc-sub::after { content: "▍"; margin-left: 3px; animation: blink 1s steps(1) infinite; }
.cc-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.cc-chip { font-size: .74rem; padding: 6px 13px; border-radius: 999px; background: rgba(255,255,255,.05); border: 1px solid var(--line); letter-spacing: .04em;
  transition: transform .25s var(--ease), border-color .25s ease; animation: fade-up .6s var(--ease) both; }
.cc-chip:hover { transform: translateY(-3px); border-color: var(--blue-soft); }
.cc-chip.live::before, .cc-chip.off::before { content: ""; display: inline-block; width: 7px; height: 7px; margin-right: 7px; border-radius: 50%; }
.cc-chip.live::before { background: var(--teal); box-shadow: 0 0 8px var(--teal); animation: hud 1.4s infinite; }
.cc-chip.off::before { background: var(--red); }

.cc-ticker { overflow: hidden; margin: 14px 0 6px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: 7px 0;
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent); mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent); }
.cc-ticker > div { display: inline-flex; white-space: nowrap; animation: ticker 32s linear infinite; font-family: 'JetBrains Mono', monospace; font-size: .7rem; letter-spacing: .18em; color: rgba(199,210,254,.7); }
.cc-ticker span { padding-right: 44px; } .cc-ticker span::before { content: "◆ "; color: var(--blue-soft); }

.cc-scroll { overflow-x: auto; overflow-y: hidden; }
.cc { position: relative; min-width: 780px; aspect-ratio: 1000 / 440; container-type: inline-size; }
.cc > img.cc-svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.cc-src { position: absolute; left: 0; width: 25.5%; text-align: right; translate: 0 -50%; white-space: nowrap;
  font-size: 15px; font-size: clamp(10px, 1.45cqw, 17px); color: rgba(226,232,255,.88); animation: slide-in .7s var(--ease) both; transition: color .25s ease, transform .25s var(--ease); }
.cc-src:hover { color: #fff; transform: translateX(-6px); text-shadow: 0 0 14px rgba(142,162,255,.9); }
.cc-core { position: absolute; left: 56%; top: 50%; translate: -50% -50%; text-align: center; width: 24%; animation: pop 1s .2s var(--ease) both; }
.cc-num { display: block; font-weight: 300; letter-spacing: -.04em; line-height: 1; color: #fff; text-shadow: 0 0 30px rgba(120,130,255,.8);
  font-size: 56px; font-size: clamp(30px, 6.6cqw, 78px); }
.cc-num small { font-size: .38em; font-weight: 500; margin-left: .2em; vertical-align: super; color: var(--teal); letter-spacing: 0; }
.cc-num.spam small { color: var(--red); }
.cc-count { counter-reset: n var(--n); animation: count 1.6s var(--ease) both; }
.cc-count::before { content: counter(n); }
.cc-cap { display: block; margin-top: 8px; font-size: 14px; font-size: clamp(9px, 1.35cqw, 16px); color: rgba(226,232,255,.8); letter-spacing: .02em; }
.cc-verdict { display: inline-block; margin-top: 10px; padding: 4px 14px; border-radius: 999px; font-weight: 700; letter-spacing: .2em; font-size: 12px; font-size: clamp(9px, 1.2cqw, 14px);
  border: 1px solid currentColor; animation: pop .7s .5s var(--ease) both; }
.cc-verdict.spam { color: #FB7185; background: rgba(244,63,94,.12); box-shadow: 0 0 22px rgba(244,63,94,.45); }
.cc-verdict.ham  { color: #93A4FF; background: rgba(59,59,255,.14); box-shadow: 0 0 22px rgba(59,59,255,.5); }
.cc-verdict.idle { color: rgba(199,210,254,.55); }

.cc-out { position: absolute; left: 82.5%; width: 16.5%; translate: 0 -50%; padding: 12px 16px; border-radius: 20px; overflow: hidden;
  background: linear-gradient(145deg, rgba(255,255,255,.08), rgba(255,255,255,.015)); border: 1px solid rgba(255,255,255,.10);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.10); animation: slide-left .8s .3s var(--ease) both; transition: transform .3s var(--ease), box-shadow .3s ease, opacity .4s ease; }
.cc-out:hover { transform: scale(1.04); }
.cc-out.spam { top: 25%; } .cc-out.ham { top: 75%; }
.cc-out b { display: block; font-weight: 400; line-height: 1; font-size: 30px; font-size: clamp(18px, 3.1cqw, 38px); letter-spacing: -.02em; }
.cc-out em { display: block; font-style: normal; margin-top: 6px; opacity: .75; font-size: 13px; font-size: clamp(9px, 1.3cqw, 15px); }
.cc-out.spam b { color: #FDA4AF; } .cc-out.ham b { color: #A5B4FF; }
.cc-out.dim { opacity: .45; }
.cc-out.on.spam { border-color: rgba(244,63,94,.6); animation: slide-left .8s .3s var(--ease) both, glow-red 2.2s 1.2s ease-in-out infinite; }
.cc-out.on.ham  { border-color: rgba(59,59,255,.7);  animation: slide-left .8s .3s var(--ease) both, glow-blue 2.2s 1.2s ease-in-out infinite; }
.cc-out::after { content: ""; position: absolute; top: 0; left: 0; width: 35%; height: 100%; background: linear-gradient(90deg, transparent, rgba(255,255,255,.10), transparent); animation: sweep 5s ease-in-out infinite; }

/* ---------------------------------------------------------------- probability meter + stamp */
.mg-meter { height: 12px; border-radius: 999px; background: rgba(255,255,255,.08); overflow: hidden; margin: 6px 0 2px; }
.mg-meter > span { display: block; height: 100%; border-radius: inherit; transform-origin: left; animation: bar-grow 1.1s var(--ease) both; }
.mg-meter.spam > span { background: linear-gradient(90deg, #F59E0B, var(--red)); box-shadow: 0 0 16px rgba(244,63,94,.6); }
.mg-meter.ham  > span { background: linear-gradient(90deg, var(--cyan), var(--blue)); box-shadow: 0 0 16px rgba(59,59,255,.6); }
.mg-stamp-wrap { display: flex; justify-content: flex-end; margin: -6px 12px 6px 0; }
.mg-stamp { display: inline-block; padding: 6px 22px; border: 4px solid currentColor; border-radius: 12px; font-weight: 800; font-size: 2rem; letter-spacing: .18em; text-transform: uppercase;
  transform: rotate(-10deg); animation: stamp .9s var(--ease) both; mix-blend-mode: screen; box-shadow: inset 0 0 0 2px rgba(255,255,255,.08), 0 0 26px currentColor; }
.mg-stamp.spam { color: #FB7185; } .mg-stamp.ham { color: #93A4FF; }

@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
@media (max-width: 768px) { .cc-panel { padding: 16px; } }
</style>
"""


# =========================================================================== SVG helpers
def _uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.strip().encode()).decode()


def _ticks(cx: float, cy: float, r1: float, r2: float, n: int, *, major: int = 5, stroke: str = "#C7D2FE", opacity: float = 0.30) -> str:
    """Radial tick marks (the dial look from the reference)."""
    parts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        long = i % major == 0
        ra = r1 - (5 if long else 0)
        parts.append(
            f'<line x1="{cx + ra * math.cos(a):.1f}" y1="{cy + ra * math.sin(a):.1f}" '
            f'x2="{cx + r2 * math.cos(a):.1f}" y2="{cy + r2 * math.sin(a):.1f}" '
            f'stroke="{stroke}" stroke-opacity="{opacity * (1.6 if long else 1):.2f}" stroke-width="{1.6 if long else 1}"/>'
        )
    return "".join(parts)


_SVG_DOTS = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='36' height='36'>"
    "<circle cx='1' cy='1' r='1.1' fill='#8EA2FF' fill-opacity='.35'/></svg>"
)


def _ring_svg() -> str:
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1000 1000' fill='none'>"
        "<circle cx='500' cy='500' r='490' stroke='#8EA2FF' stroke-opacity='.10'/>"
        "<circle cx='500' cy='500' r='330' stroke='#8EA2FF' stroke-opacity='.08'/>"
        + _ticks(500, 500, 455, 478, 180, major=6, opacity=0.22)
        + "</svg>"
    )


# --------------------------------------------------------------------------- command-center artwork
_SOURCES = [
    "Header Signals", "URLs & Links", "Body Words", "Word Bigrams",
    "Character n-grams", "HTML & MIME", "Sender / Reply-To", "Attachments",
]
_SRC_Y = [48 + i * 49 for i in range(8)]
_CX, _CY = 560, 220


def _pipeline_svg(state: str) -> str:
    """Animated artwork: source lines -> glowing dial -> red (spam) / blue (ham) ribbons.

    ``state`` is 'idle', 'spam' or 'ham'. Delivered as an <img> data-URI so CSS animations
    embedded in the SVG run without needing HTML sanitiser support.
    """
    fast = state != "idle"
    pulse_dur = "1.8s" if fast else "3.6s"
    red_op = {"idle": 0.55, "spam": 1.0, "ham": 0.22}[state]
    blue_op = {"idle": 0.55, "spam": 0.22, "ham": 1.0}[state]

    css = (
        "@keyframes flow{from{stroke-dashoffset:70}to{stroke-dashoffset:-930}}"
        "@keyframes march{to{stroke-dashoffset:-48}}"
        "@keyframes spin{to{transform:rotate(360deg)}}"
        "@keyframes breathe{0%,100%{opacity:.45}50%{opacity:1}}"
        "@keyframes rim{0%,100%{stroke-opacity:.55}50%{stroke-opacity:1}}"
        f".r1{{transform-origin:{_CX}px {_CY}px;animation:spin 80s linear infinite}}"
        f".r2{{transform-origin:{_CX}px {_CY}px;animation:spin 130s linear infinite reverse}}"
        ".pulse{fill:none;stroke-linecap:round;stroke-dasharray:70 930}"
        ".dash{fill:none;stroke-dasharray:4 12;animation:march 2.2s linear infinite}"
        ".dot{animation:breathe 2.4s ease-in-out infinite}"
        ".rim{animation:rim 3s ease-in-out infinite}"
    )

    # sources -> core
    lines = []
    for i, y in enumerate(_SRC_Y):
        d = f"M274 {y} C 385 {y}, 385 {_CY}, {_CX - 124} {_CY}"
        lines.append(
            f'<path d="{d}" fill="none" stroke="#3B3BFF" stroke-opacity=".38" stroke-width="1.5"/>'
            f'<path d="{d}" class="dash" stroke="#8EA2FF" stroke-opacity=".35" stroke-width="1.2" style="animation-delay:{i * .15:.2f}s"/>'
            f'<path d="{d}" class="pulse" pathLength="1000" stroke="url(#gp)" stroke-width="2.4" filter="url(#glow)" '
            f'style="animation:flow {pulse_dur} linear {i * .33:.2f}s infinite"/>'
            f'<circle cx="274" cy="{y}" r="6.5" fill="#0B0B3A" stroke="#6C7BFF" stroke-width="2" class="dot" style="animation-delay:{i * .25:.2f}s"/>'
            f'<circle cx="274" cy="{y}" r="2.4" fill="#C7D2FF" class="dot" style="animation-delay:{i * .25:.2f}s"/>'
        )

    # core -> verdict ribbons
    rx = _CX + 124
    spam_d = f"M{rx - 4} {_CY - 6} C 760 {_CY - 6}, 740 110, 826 110"
    ham_d = f"M{rx - 4} {_CY + 6} C 760 {_CY + 6}, 740 330, 826 330"
    ribbons = (
        f'<g opacity="{red_op}"><path d="{spam_d}" fill="none" stroke="url(#gr)" stroke-width="24" stroke-linecap="round" stroke-opacity=".85"/>'
        f'<path d="{spam_d}" class="pulse" pathLength="1000" stroke="#FFD1DA" stroke-width="5" filter="url(#glow)" style="animation:flow {pulse_dur} linear infinite"/></g>'
        f'<g opacity="{blue_op}"><path d="{ham_d}" fill="none" stroke="url(#gb)" stroke-width="24" stroke-linecap="round" stroke-opacity=".85"/>'
        f'<path d="{ham_d}" class="pulse" pathLength="1000" stroke="#D5DCFF" stroke-width="5" filter="url(#glow)" style="animation:flow {pulse_dur} linear .6s infinite"/></g>'
    )

    core = (
        f'<ellipse cx="{_CX}" cy="{_CY + 70}" rx="95" ry="60" fill="#2A2AFF" opacity=".35" filter="url(#blur)"/>'
        f'<circle cx="{_CX}" cy="{_CY}" r="196" fill="none" stroke="#8EA2FF" stroke-opacity=".08"/>'
        f'<g class="r1">{_ticks(_CX, _CY, 168, 186, 120, major=5, opacity=0.30)}</g>'
        f'<g class="r2">{_ticks(_CX, _CY, 150, 160, 72, major=6, stroke="#8EA2FF", opacity=0.22)}</g>'
        f'<circle cx="{_CX}" cy="{_CY}" r="124" fill="url(#core)"/>'
        f'<circle cx="{_CX}" cy="{_CY}" r="124" fill="none" stroke="url(#rim)" stroke-width="3.5" class="rim"/>'
        f'<circle cx="{_CX}" cy="{_CY}" r="100" fill="none" stroke="#fff" stroke-opacity=".10"/>'
    )

    defs = (
        "<defs>"
        '<linearGradient id="gp" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#6C7BFF" stop-opacity="0"/><stop offset=".5" stop-color="#C7D2FF"/><stop offset="1" stop-color="#6C7BFF" stop-opacity="0"/></linearGradient>'
        '<linearGradient id="gr" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#5B0A1E" stop-opacity=".2"/><stop offset=".6" stop-color="#D81B3F"/><stop offset="1" stop-color="#F43F5E"/></linearGradient>'
        '<linearGradient id="gb" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#14125E" stop-opacity=".2"/><stop offset=".6" stop-color="#2A2AE8"/><stop offset="1" stop-color="#4F5BFF"/></linearGradient>'
        '<radialGradient id="core" cx="50%" cy="38%" r="70%"><stop offset="0" stop-color="#0B0B30"/><stop offset=".7" stop-color="#06061C"/><stop offset="1" stop-color="#14127A"/></radialGradient>'
        '<linearGradient id="rim" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#5B6CFF"/><stop offset=".55" stop-color="#2A2AFF" stop-opacity=".25"/><stop offset="1" stop-color="#22D3EE" stop-opacity=".7"/></linearGradient>'
        '<filter id="glow" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
        '<filter id="blur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="22"/></filter>'
        "</defs>"
    )
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1000 440' fill='none'>"
        f"<style>{css}</style>{defs}{core}{''.join(lines)}{ribbons}</svg>"
    )


# --------------------------------------------------------------------------- injected page chrome
_PARTICLES = [  # (left %, size px, duration s, delay s, colour, drift px)
    (4, 3, 22, 0, "#8EA2FF", 40), (9, 2, 28, 5, "#22D3EE", -30), (15, 4, 25, 9, "#8B5CF6", 50), (21, 2, 31, 2, "#8EA2FF", -40),
    (27, 3, 24, 13, "#22D3EE", 30), (33, 2, 29, 7, "#8EA2FF", -20), (39, 4, 27, 16, "#8B5CF6", 45), (45, 2, 23, 3, "#22D3EE", -35),
    (51, 3, 30, 11, "#8EA2FF", 25), (57, 2, 26, 19, "#8B5CF6", -45), (63, 4, 32, 6, "#22D3EE", 35), (69, 2, 21, 14, "#8EA2FF", -25),
    (75, 3, 28, 1, "#8B5CF6", 40), (81, 2, 25, 10, "#22D3EE", -30), (87, 4, 30, 18, "#8EA2FF", 20), (93, 2, 24, 4, "#8B5CF6", -50),
]


def _background_html() -> str:
    root = f"<style>:root{{--img-dots:url({_uri(_SVG_DOTS)});--img-ring:url({_uri(_ring_svg())});}}</style>"
    parts = "".join(
        f'<span class="mg-particle" style="left:{l}%;--s:{s}px;--d:{d}s;--delay:{dl}s;--c:{c};--x:{x}px"></span>'
        for l, s, d, dl, c, x in _PARTICLES
    )
    return (
        root
        + '<div class="mg-topbar"></div><div class="mg-bg"><div class="mg-dots"></div>'
        + '<div class="mg-bigring"></div><div class="mg-bigring l"></div><div class="mg-vscan"></div>'
        + parts
        + "</div>"
    )


def inject_styles() -> None:
    """Inject the global theme. Call once, near the top of ``app.py``."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(_background_html(), unsafe_allow_html=True)


# =========================================================================== components
def command_center(
    model_label: str = "MailGuard",
    version: str = "",
    prob: float | None = None,
    is_spam: bool | None = None,
    threshold: float | None = None,
    online: bool = True,
) -> None:
    """Animated 'sources -> core -> verdict' panel (mirrors the CyberX command-center look)."""
    state = "idle" if prob is None or is_spam is None else ("spam" if is_spam else "ham")
    svg = _uri(_pipeline_svg(state))

    src_html = "".join(
        f'<div class="cc-src" style="top:{y / 440 * 100:.2f}%;animation-delay:{.1 + i * .08:.2f}s">{name}</div>'
        for i, (name, y) in enumerate(zip(_SOURCES, _SRC_Y))
    )

    if state == "idle":
        num = '<span class="cc-num" style="font-size:clamp(24px,4.6cqw,56px);letter-spacing:.04em">READY</span>'
        cap = '<span class="cc-cap">Awaiting email</span><span class="cc-verdict idle">STANDBY</span>'
        spam_val, ham_val = "—", "—"
    else:
        pct = max(0.0, min(100.0, prob * 100))  # type: ignore[operator]
        cls = "spam" if state == "spam" else ""
        num = f'<span class="cc-num {cls}"><span class="cc-count" style="--t:{round(pct)}"></span><small>%</small></span>'
        cap = '<span class="cc-cap">Spam probability</span>' f'<span class="cc-verdict {state}">{state.upper()}</span>'
        spam_val, ham_val = f"{pct:.1f}%", f"{100 - pct:.1f}%"

    on_spam = " on" if state == "spam" else (" dim" if state == "ham" else "")
    on_ham = " on" if state == "ham" else (" dim" if state == "spam" else "")
    thr = f"threshold {threshold:.2f}" if threshold is not None else "threshold —"

    status = '<span class="cc-chip live">THREAT ENGINE ONLINE</span>' if online else '<span class="cc-chip off">MODEL OFFLINE</span>'
    chips = status + f'<span class="cc-chip">{model_label}</span>' + (f'<span class="cc-chip">v{version}</span>' if version else "") + f'<span class="cc-chip">{thr}</span>'
    steps = ["PARSING HEADERS", "EXTRACTING URLS", "DECODING MIME", "VECTORISING TEXT", "SCORING EMAIL", "EXPLAINING WITH SHAP", "APPLYING THRESHOLD", "ZERO EMAILS STORED"]
    ticker = "".join(f"<span>{s}</span>" for s in steps) * 2

    html = (
        '<div class="cc-panel">'
        '<div class="cc-head"><div><div class="cc-sub">&gt; mailguard --scan --explain</div>'
        '<h2 class="cc-title" style="animation:none">Email Threat Command Center</h2></div>'
        f'<div class="cc-chips">{chips}</div></div>'
        f'<div class="cc-ticker"><div>{ticker}</div></div>'
        '<div class="cc-scroll"><div class="cc">'
        f'<img class="cc-svg" alt="" src="{svg}"/>'
        f"{src_html}"
        f'<div class="cc-core">{num}{cap}</div>'
        f'<div class="cc-out spam{on_spam}"><b>{spam_val}</b><em>Spam</em></div>'
        f'<div class="cc-out ham{on_ham}"><b>{ham_val}</b><em>Ham (legitimate)</em></div>'
        "</div></div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def meter(probability: float, is_spam: bool) -> None:
    """Animated horizontal probability meter (pure CSS)."""
    width = max(2.0, min(100.0, probability * 100))
    cls = "spam" if is_spam else "ham"
    st.markdown(f'<div class="mg-meter {cls}"><span style="width:{width:.1f}%"></span></div>', unsafe_allow_html=True)


def stamp(is_spam: bool) -> None:
    """Rubber-stamp style SPAM / HAM badge that slams in."""
    cls, text = ("spam", "Spam") if is_spam else ("ham", "Ham")
    st.markdown(f'<div class="mg-stamp-wrap"><span class="mg-stamp {cls}">{text}</span></div>', unsafe_allow_html=True)
