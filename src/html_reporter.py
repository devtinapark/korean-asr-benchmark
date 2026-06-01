"""
HTML Report Generator
Creates a self-contained HTML comparison report for ASR benchmark results
"""
from pathlib import Path
from typing import Dict, List
import pandas as pd


MODEL_COLORS = {
    0: {"primary": "#4F46E5", "light": "#EEF2FF", "border": "#C7D2FE"},  # indigo
    1: {"primary": "#059669", "light": "#ECFDF5", "border": "#A7F3D0"},  # emerald
    2: {"primary": "#DC2626", "light": "#FEF2F2", "border": "#FECACA"},  # red
}


def _cer_color(cer: float) -> str:
    if cer <= 0.05:   return "#059669"
    if cer <= 0.10:   return "#D97706"
    return "#DC2626"


def _wer_color(wer: float) -> str:
    if wer <= 0.10:   return "#059669"
    if wer <= 0.20:   return "#D97706"
    return "#DC2626"


def _bar(value: float, max_val: float, color: str, width: int = 120) -> str:
    pct = min(value / max_val, 1.0) if max_val > 0 else 0
    px = int(pct * width)
    return (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="width:{px}px;height:10px;background:{color};border-radius:3px;min-width:2px"></div>'
        f'<span style="font-size:13px;color:#374151">{value:.4f}</span>'
        f'</div>'
    )


def generate_html_report(
    ranked_df: pd.DataFrame,
    top_models: List[str],
    per_model_samples: Dict[str, List[dict]] = None,
    cost_data: Dict[str, dict] = None,
    metadata: Dict = None,
) -> str:

    per_model_samples = per_model_samples or {}
    cost_data = cost_data or {}
    model_colors = {name: MODEL_COLORS[i % len(MODEL_COLORS)] for i, name in enumerate(top_models)}

    # ── Summary cards ──────────────────────────────────────────────────────────
    cards_html = ""
    for model_name in top_models:
        if ranked_df.empty or model_name not in ranked_df['model'].values:
            continue
        row = ranked_df[ranked_df['model'] == model_name].iloc[0]
        cost = cost_data.get(model_name, {})
        c = model_colors[model_name]

        cards_html += f"""
        <div style="border:2px solid {c['border']};border-radius:12px;padding:24px;background:{c['light']};flex:1;min-width:280px">
            <div style="font-size:11px;font-weight:700;letter-spacing:1px;color:{c['primary']};text-transform:uppercase;margin-bottom:8px">
                #{int(row['rank'])}
            </div>
            <div style="font-size:18px;font-weight:700;color:#111827;margin-bottom:20px;word-break:break-all">
                {model_name}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
                <div>
                    <div style="font-size:11px;color:#6B7280;margin-bottom:3px">CER</div>
                    <div style="font-size:24px;font-weight:700;color:{_cer_color(row['cer'])}">{row['cer']:.4f}</div>
                </div>
                <div>
                    <div style="font-size:11px;color:#6B7280;margin-bottom:3px">WER</div>
                    <div style="font-size:24px;font-weight:700;color:{_wer_color(row['wer'])}">{row['wer']:.4f}</div>
                </div>
                <div>
                    <div style="font-size:11px;color:#6B7280;margin-bottom:3px">Loanword Acc</div>
                    <div style="font-size:20px;font-weight:600;color:#111827">{row['loanword_accuracy']:.4f}</div>
                </div>
                <div>
                    <div style="font-size:11px;color:#6B7280;margin-bottom:3px">Composite Score</div>
                    <div style="font-size:20px;font-weight:600;color:#111827">{row['composite_score']:.4f}</div>
                </div>
            </div>
            {"" if not cost else f'''
            <div style="margin-top:16px;padding-top:16px;border-top:1px solid {c['border']}">
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                    <div>
                        <div style="font-size:11px;color:#6B7280;margin-bottom:2px">Avg Latency</div>
                        <div style="font-size:16px;font-weight:600;color:#111827">{cost.get('avg_latency', 0):.2f}s</div>
                    </div>
                    <div>
                        <div style="font-size:11px;color:#6B7280;margin-bottom:2px">Est. Cost</div>
                        <div style="font-size:16px;font-weight:600;color:#111827">${cost.get('estimated_cost', 0):.5f}</div>
                    </div>
                </div>
            </div>'''}
        </div>
        """

    # ── Per-sample table ───────────────────────────────────────────────────────
    # Gather all sample IDs across models
    all_ids = []
    for name in top_models:
        for s in per_model_samples.get(name, []):
            if s['id'] not in all_ids:
                all_ids.append(s['id'])

    # Build lookup: model → sample_id → metrics
    lookup = {}
    for name in top_models:
        lookup[name] = {s['id']: s for s in per_model_samples.get(name, [])}

    max_cer = 0.25

    # Table header
    header_cols = "".join(
        f'<th colspan="2" style="background:{model_colors[m]["primary"]};color:white;padding:10px 16px;text-align:center;font-size:13px">{m}</th>'
        for m in top_models
    )
    sub_header = "".join(
        '<th style="background:#374151;color:#D1D5DB;padding:8px 12px;font-size:12px;text-align:center">CER</th>'
        '<th style="background:#374151;color:#D1D5DB;padding:8px 12px;font-size:12px;text-align:center">WER</th>'
        for _ in top_models
    )

    rows_html = ""
    for sid in all_ids:
        is_noisy = "noise" in sid
        row_bg = "#FFF7ED" if is_noisy else "#FFFFFF"
        noise_badge = '<span style="font-size:10px;background:#FEF3C7;color:#92400E;padding:2px 6px;border-radius:9px;margin-left:6px">noisy</span>' if is_noisy else ""

        cells = ""
        for model_name in top_models:
            s = lookup[model_name].get(sid)
            if s:
                cer_c = _cer_color(s['cer'])
                wer_c = _wer_color(s['wer'])
                cells += f"""
                    <td style="padding:10px 12px;text-align:center;border-right:1px solid #E5E7EB">
                        <span style="font-weight:600;color:{cer_c}">{s['cer']:.4f}</span>
                    </td>
                    <td style="padding:10px 12px;text-align:center;border-right:1px solid #E5E7EB">
                        <span style="font-weight:600;color:{wer_c}">{s['wer']:.4f}</span>
                    </td>
                """
            else:
                cells += '<td colspan="2" style="text-align:center;color:#9CA3AF">—</td>'

        rows_html += f"""
        <tr style="background:{row_bg};border-bottom:1px solid #E5E7EB">
            <td style="padding:10px 16px;font-size:13px;font-weight:500;color:#111827;white-space:nowrap;border-right:1px solid #E5E7EB">
                {sid}{noise_badge}
            </td>
            {cells}
        </tr>
        """

    # ── Noise impact section ───────────────────────────────────────────────────
    noise_rows = ""
    for model_name in top_models:
        samples = per_model_samples.get(model_name, [])
        clean = [s for s in samples if not s.get("noise")]
        noisy = [s for s in samples if s.get("noise")]
        if not clean or not noisy:
            continue
        clean_cer = sum(s["cer"] for s in clean) / len(clean)
        noisy_cer = sum(s["cer"] for s in noisy) / len(noisy)
        delta = noisy_cer - clean_cer
        delta_color = "#DC2626" if delta > 0.02 else "#D97706" if delta > 0.01 else "#059669"
        c = model_colors[model_name]
        noise_rows += f"""
        <tr style="border-bottom:1px solid #E5E7EB">
            <td style="padding:12px 16px;font-weight:600;color:{c['primary']}">{model_name}</td>
            <td style="padding:12px 16px;text-align:center;color:#059669;font-weight:600">{clean_cer:.4f}</td>
            <td style="padding:12px 16px;text-align:center;color:#DC2626;font-weight:600">{noisy_cer:.4f}</td>
            <td style="padding:12px 16px;text-align:center;font-weight:700;color:{delta_color}">+{delta:.4f}</td>
        </tr>
        """

    # ── Cost & latency table ───────────────────────────────────────────────────
    cost_rows = ""
    for model_name in top_models:
        cost = cost_data.get(model_name, {})
        if not cost:
            continue
        c = model_colors[model_name]
        cost_rows += f"""
        <tr style="border-bottom:1px solid #E5E7EB">
            <td style="padding:12px 16px;font-weight:600;color:{c['primary']}">{model_name}</td>
            <td style="padding:12px 16px;text-align:center">${cost.get('cost_per_minute', 0):.4f}</td>
            <td style="padding:12px 16px;text-align:center">{cost.get('audio_minutes', 0):.2f} min</td>
            <td style="padding:12px 16px;text-align:center;font-weight:600">${cost.get('estimated_cost', 0):.6f}</td>
            <td style="padding:12px 16px;text-align:center">{cost.get('avg_latency', 0):.2f}s</td>
            <td style="padding:12px 16px;text-align:center">{cost.get('total_latency', 0):.2f}s</td>
        </tr>
        """

    # ── Dataset info ───────────────────────────────────────────────────────────
    num_samples = metadata.get("num_samples", "—") if metadata else "—"
    dataset = metadata.get("dataset", "Kitchen Audio Samples") if metadata else "Kitchen Audio Samples"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multilingual ASR Benchmark — OpenAI vs Deepgram</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #F9FAFB; color: #111827; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px; }}
  h1 {{ font-size: 28px; font-weight: 800; color: #111827; }}
  h2 {{ font-size: 18px; font-weight: 700; color: #111827; margin-bottom: 16px; }}
  .subtitle {{ color: #6B7280; margin-top: 6px; font-size: 15px; }}
  .section {{ background: white; border-radius: 12px; padding: 28px; margin-top: 28px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align:left; font-size:13px; font-weight:600; padding:10px 16px; }}
  .legend {{ display:flex; gap:20px; margin-top:12px; font-size:12px; color:#6B7280; }}
  .legend span {{ display:flex; align-items:center; gap:5px; }}
  .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div style="margin-bottom:8px">
    <h1>Multilingual ASR Benchmark</h1>
    <p class="subtitle">OpenAI gpt-4o-transcribe vs Deepgram Nova-3 &nbsp;·&nbsp; {dataset} &nbsp;·&nbsp; {num_samples} clips &nbsp;·&nbsp; EN / KO mixed language</p>
  </div>

  <!-- Model summary cards -->
  <div class="section">
    <h2>Overall Results</h2>
    <div style="display:flex;gap:20px;flex-wrap:wrap">
      {cards_html}
    </div>
    <div class="legend" style="margin-top:20px">
      <span><span class="dot" style="background:#059669"></span> Excellent (CER ≤ 0.05)</span>
      <span><span class="dot" style="background:#D97706"></span> Good (CER ≤ 0.10)</span>
      <span><span class="dot" style="background:#DC2626"></span> Needs improvement (CER > 0.10)</span>
    </div>
  </div>

  <!-- Per-sample breakdown -->
  <div class="section">
    <h2>Per-Sample Breakdown</h2>
    <p style="font-size:13px;color:#6B7280;margin-bottom:16px">
      Clean clips have white background &nbsp;·&nbsp; <span style="background:#FFF7ED;padding:2px 8px;border-radius:4px">Noisy clips</span> have orange tint
    </p>
    <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th style="background:#1F2937;color:white;border-radius:8px 0 0 0;padding:10px 16px">Sample</th>
          {header_cols}
        </tr>
        <tr>
          <th style="background:#374151;color:#9CA3AF;font-size:11px;padding:6px 16px">ID</th>
          {sub_header}
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
    </div>
  </div>

  <!-- Noise impact -->
  <div class="section">
    <h2>Noise Impact</h2>
    <p style="font-size:13px;color:#6B7280;margin-bottom:16px">Average CER on clean vs noisy clips — lower delta means more noise-robust</p>
    <table>
      <thead>
        <tr style="border-bottom:2px solid #E5E7EB">
          <th>Model</th>
          <th style="text-align:center">Clean avg CER</th>
          <th style="text-align:center">Noisy avg CER</th>
          <th style="text-align:center">Degradation Δ</th>
        </tr>
      </thead>
      <tbody>{noise_rows}</tbody>
    </table>
  </div>

  <!-- Cost & latency -->
  <div class="section">
    <h2>Cost &amp; Latency</h2>
    <p style="font-size:13px;color:#6B7280;margin-bottom:16px">
      Cost based on audio duration × price per minute. Latency = API response time only (rate-limit pauses excluded).
    </p>
    <table>
      <thead>
        <tr style="border-bottom:2px solid #E5E7EB">
          <th>Model</th>
          <th style="text-align:center">$/min</th>
          <th style="text-align:center">Audio duration</th>
          <th style="text-align:center">Est. cost (this run)</th>
          <th style="text-align:center">Avg latency/clip</th>
          <th style="text-align:center">Total latency</th>
        </tr>
      </thead>
      <tbody>{cost_rows}</tbody>
    </table>
  </div>

  <!-- Metrics explanation -->
  <div class="section">
    <h2>How Metrics Are Calculated</h2>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:4px">
      <div>
        <p style="font-weight:600;margin-bottom:4px">CER (Character Error Rate)</p>
        <p style="font-size:13px;color:#6B7280">Primary metric for Korean — spaces stripped before comparison since Korean spacing is inconsistent across models. Lower is better.</p>
      </div>
      <div>
        <p style="font-weight:600;margin-bottom:4px">WER (Word Error Rate)</p>
        <p style="font-size:13px;color:#6B7280">Word-level accuracy. Less reliable for Korean due to ambiguous word boundaries — use CER as primary. Lower is better.</p>
      </div>
      <div>
        <p style="font-weight:600;margin-bottom:4px">Loanword / Code-switching Accuracy</p>
        <p style="font-size:13px;color:#6B7280">Accuracy on English loanwords in Korean context (오븐, 레시피, 간 맞추기). Higher is better.</p>
      </div>
      <div>
        <p style="font-weight:600;margin-bottom:4px">Composite Score</p>
        <p style="font-size:13px;color:#6B7280">Weighted: CER 55% + WER 30% + Loanword 15%. Relative between models — only meaningful for comparison.</p>
      </div>
    </div>
  </div>

  <p style="text-align:center;color:#9CA3AF;font-size:12px;margin-top:32px">
    Generated by korean-asr-benchmark &nbsp;·&nbsp; github.com/your-username/korean-asr-benchmark
  </p>

</div>
</body>
</html>"""

    return html


def save_html_report(
    output_dir: str,
    ranked_df: pd.DataFrame,
    top_models: List[str],
    per_model_samples: Dict[str, List[dict]] = None,
    cost_data: Dict[str, dict] = None,
    metadata: Dict = None,
    filename: str = "benchmark_report.html",
):
    html = generate_html_report(ranked_df, top_models, per_model_samples, cost_data, metadata)
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"HTML report saved to {path}")
    return path
