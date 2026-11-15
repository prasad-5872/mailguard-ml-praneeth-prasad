"""Plotly figure builders used by the Streamlit pages.

Colors follow a validated categorical palette (fixed slot order, separate light/dark
steps); ham/spam always use slots 1/2 so identity is consistent across pages.
"""

from __future__ import annotations

from typing import Any, Sequence

import plotly.graph_objects as go

LIGHT = {
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    "seq": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    "grid": "rgba(82,81,78,0.15)",
    "muted": "#8a8984",
    "ink": "#0b0b0b",
}
DARK = {
    "series": ["#5B6CFF", "#F43F5E", "#22D3EE", "#C98500"],
    "seq": ["#14125E", "#2A2AE8", "#4F5BFF", "#8EA2FF", "#C7D2FF"],
    "grid": "rgba(195,194,183,0.15)",
    "muted": "#9a998f",
    "ink": "#f0efec",
}
STATUS = {"good": "#0ca30c", "critical": "#d03b3b"}


def palette(dark: bool) -> dict[str, Any]:
    return DARK if dark else LIGHT


def class_colors(dark: bool) -> dict[str, str]:
    s = palette(dark)["series"]
    return {"ham": s[0], "spam": s[1]}


def _layout(fig: go.Figure, dark: bool, *, height: int = 360, xtitle: str = "", ytitle: str = "", legend: bool = True) -> go.Figure:
    p = palette(dark)
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(namelength=-1),
        transition=dict(duration=600, easing="cubic-in-out"),
        font=dict(family="Inter, sans-serif"),
    )
    fig.update_xaxes(title_text=xtitle, gridcolor=p["grid"], zeroline=False, linecolor=p["grid"])
    fig.update_yaxes(title_text=ytitle, gridcolor=p["grid"], zeroline=False, linecolor=p["grid"])
    return fig


def probability_bar(prob: float, threshold: float, dark: bool) -> go.Figure:
    """Single horizontal bar of spam probability with the decision threshold marked."""
    colors = class_colors(dark)
    label = "spam" if prob >= threshold else "ham"
    fig = go.Figure(
        go.Bar(
            x=[prob], y=[""], orientation="h", marker=dict(color=colors[label], cornerradius=4),
            width=0.5, hovertemplate="Spam probability %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=threshold, line=dict(color=palette(dark)["ink"], width=2, dash="dot"))
    fig.add_annotation(x=threshold, y=1, yref="paper", text=f"threshold {threshold:.2f}", showarrow=False,
                       yanchor="bottom", font=dict(size=12))
    _layout(fig, dark, height=140, legend=False)
    fig.update_xaxes(range=[0, 1], tickformat=".1f", title_text="Spam probability")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=8, r=8, t=28, b=8))
    return fig


def contribution_chart(contributions: Sequence[dict[str, Any]], dark: bool) -> go.Figure:
    colors = class_colors(dark)
    items = list(contributions)[::-1]  # largest at top
    labels = [
        f"{c['feature']}  ·  {c['kind']}" + (" (absent)" if c["kind"] != "structured" and not c.get("value") else "")
        for c in items
    ]
    fig = go.Figure(
        go.Bar(
            x=[c["contribution"] for c in items],
            y=labels,
            orientation="h",
            marker=dict(color=[colors[c["direction"]] for c in items], cornerradius=4),
            customdata=[[c["direction"], c.get("description", "")] for c in items],
            hovertemplate="%{y}<br>contribution %{x:+.3f} (towards %{customdata[0]})<br>%{customdata[1]}<extra></extra>",
        )
    )
    _layout(fig, dark, height=max(260, 28 * len(items) + 60), xtitle="Contribution to spam score (− ham · + spam)", legend=False)
    fig.add_vline(x=0, line=dict(color=palette(dark)["muted"], width=1))
    return fig


def confusion_matrix_chart(cm: dict[str, int], dark: bool) -> go.Figure:
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    seq = palette(dark)["seq"]
    text = [[f"TN<br>{cm['tn']}", f"FP<br>{cm['fp']}"], [f"FN<br>{cm['fn']}", f"TP<br>{cm['tp']}"]]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=["Predicted ham", "Predicted spam"], y=["Actual ham", "Actual spam"],
            colorscale=[[i / (len(seq) - 1), c] for i, c in enumerate(seq)], showscale=False,
            text=text, texttemplate="%{text}", textfont=dict(size=16), xgap=2, ygap=2,
            hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
        )
    )
    _layout(fig, dark, height=320, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return fig


def curve_chart(curves: list[dict[str, Any]], kind: str, dark: bool) -> go.Figure:
    """Overlay ROC ('roc') or precision-recall ('pr') curves; ``curves`` items: name, x, y, auc."""
    series = palette(dark)["series"]
    fig = go.Figure()
    for i, c in enumerate(curves):
        fig.add_trace(
            go.Scatter(
                x=c["x"], y=c["y"], mode="lines", name=f"{c['name']} (AUC {c['auc']:.4f})",
                line=dict(color=series[i % len(series)], width=2),
                hovertemplate=f"{c['name']}<br>%{{x:.3f}}, %{{y:.3f}}<extra></extra>",
            )
        )
    if kind == "roc":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                                 line=dict(color=palette(dark)["muted"], width=1, dash="dot"), hoverinfo="skip"))
        _layout(fig, dark, xtitle="False positive rate", ytitle="True positive rate")
    else:
        _layout(fig, dark, xtitle="Recall", ytitle="Precision")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def threshold_chart(rows: Sequence[dict[str, float]], selected_threshold: float, dark: bool) -> go.Figure:
    series = palette(dark)["series"]
    fig = go.Figure()
    for i, (key, name) in enumerate((("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("f0_5", "F0.5"))):
        fig.add_trace(
            go.Scatter(
                x=[r["threshold"] for r in rows], y=[r[key] for r in rows], mode="lines+markers", name=name,
                line=dict(color=series[i], width=2), marker=dict(size=8),
                hovertemplate=f"{name} %{{y:.4f}} at threshold %{{x:.2f}}<extra></extra>",
            )
        )
    fig.add_vline(x=selected_threshold, line=dict(color=palette(dark)["ink"], width=1, dash="dot"))
    fig.add_annotation(x=selected_threshold, y=1, yref="paper", text=f"tuned {selected_threshold:.2f}",
                       showarrow=False, yanchor="bottom", xanchor="left")
    _layout(fig, dark, xtitle="Decision threshold", ytitle="Score")
    fig.update_xaxes(range=[0, 1])
    return fig


def calibration_chart(curves: list[dict[str, Any]], dark: bool, min_count: int = 5) -> go.Figure:
    """Reliability diagram. Bins with fewer than ``min_count`` emails are hidden (too noisy to read)."""
    series = palette(dark)["series"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfectly calibrated",
                             line=dict(color=palette(dark)["muted"], width=1, dash="dot"), hoverinfo="skip"))
    for i, c in enumerate(curves):
        keep = [j for j, n in enumerate(c.get("count") or []) if n >= min_count]
        fig.add_trace(
            go.Scatter(
                x=[c["x"][j] for j in keep], y=[c["y"][j] for j in keep], mode="lines+markers", name=c["name"],
                line=dict(color=series[i % len(series)], width=2), marker=dict(size=8),
                customdata=[c["count"][j] for j in keep],
                hovertemplate=f"{c['name']}<br>predicted %{{x:.2f}} → observed %{{y:.2f}}<br>n=%{{customdata}}<extra></extra>",
            )
        )
    _layout(fig, dark, xtitle="Mean predicted spam probability", ytitle="Observed spam fraction")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def probability_histogram(hist: dict[str, list], dark: bool) -> go.Figure:
    colors = class_colors(dark)
    edges = hist["edges"]
    centers = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    fig = go.Figure()
    for cls in ("ham", "spam"):
        fig.add_trace(
            go.Bar(x=centers, y=hist[cls], name=f"Actual {cls}", marker=dict(color=colors[cls], cornerradius=4),
                   hovertemplate=f"Actual {cls}: %{{y}} emails<br>p≈%{{x:.3f}}<extra></extra>")
        )
    fig.update_layout(barmode="group", bargap=0.1)
    _layout(fig, dark, xtitle="Predicted spam probability (test set)", ytitle="Emails")
    fig.update_yaxes(type="log")
    return fig


def class_distribution_chart(spam: int, ham: int, dark: bool) -> go.Figure:
    colors = class_colors(dark)
    fig = go.Figure(
        go.Bar(
            x=[ham, spam], y=["Ham", "Spam"], orientation="h",
            marker=dict(color=[colors["ham"], colors["spam"]], cornerradius=4),
            text=[ham, spam], textposition="outside",
            hovertemplate="%{y}: %{x} predictions<extra></extra>",
        )
    )
    _layout(fig, dark, height=180, legend=False, xtitle="Predictions")
    return fig


def latency_chart(records: Sequence[dict[str, Any]], dark: bool) -> go.Figure:
    fig = go.Figure()
    for status, color, name in (("success", palette(dark)["series"][0], "Success"), ("error", STATUS["critical"], "Error")):
        pts = [r for r in records if r["status"] == status]
        if not pts:
            continue
        fig.add_trace(
            go.Scatter(
                x=[r["timestamp"] for r in pts], y=[r["latency_ms"] for r in pts], mode="markers", name=name,
                marker=dict(color=color, size=9, symbol="circle" if status == "success" else "x",
                            line=dict(width=2, color="rgba(0,0,0,0)")),
                customdata=[[r["request_id"], r.get("prediction") or r.get("error_type") or ""] for r in pts],
                hovertemplate="%{customdata[0]}<br>%{y:.1f} ms · %{customdata[1]}<extra></extra>",
            )
        )
    _layout(fig, dark, height=300, xtitle="Request time (UTC)", ytitle="Latency (ms)")
    return fig
