import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import av
import torch
import transformers
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


# -----------------------------
# Config
# -----------------------------


ROOT = Path(__file__).parents[1]
PIXELS_PER_TOKEN = 32**2


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "nvidia/Cosmos-Reason2-2B"
    # model_name: str = "nvidia/Cosmos-Reason2-8B"
    dtype: torch.dtype = torch.float16
    attn_implementation: str = "sdpa"
    seed: int = 0
    fps_for_prompt: int = 10
    max_new_tokens: int = 4096
    min_vision_tokens: int = 256
    max_vision_tokens: int = 8192


# -----------------------------
# Prompt builder
# -----------------------------

SYSTEM_PROMPT = "You are a helpful assistant."

# Common prompt elements
RULES_AND_FORMAT = """
Rules:
- Base findings ONLY on what is clearly visible. If unsure, say "uncertain".
- Provide timestamps (start/end) for each observation in mm:ss format.
- If you report a failure, request evidence frames at specific timestamps in mm:ss format.
- Prefer structured output and be concise.

Answer the question using the following format: <think> Your reasoning. </think> Write your final answer immediately after the </think> tag and include the timestamps in mm:ss format.
"""

BARRIER_PROMPT = f"""You are a UK worksite safety compliance agent.

Your job is to produce a compliance report for Barrier Continuity.
Check: Verify that barriers are continuously placed and locked with no gaps. Any gap between barriers is a safety violation.
{RULES_AND_FORMAT}
Return JSON exactly in this schema:
{{
  "check_type": "BARRIER_CONTINUITY",
  "status": "Compliant/Not Compliant",
  "findings": [
    {{
      "timestamp_range": ["start mm:ss", "end mm:ss"],
      "observation": "string",
      "confidence": "low|medium|high",
      "evidence_timestamps": ["t1 mm:ss", "t2 mm:ss"]
    }}
  ],
  "recommendations": ["string", "..."]
}}
"""

PPE_PROMPT = f"""You are a UK worksite safety compliance agent.

Your job is to produce a compliance report for PPE Compliance.
Check: Verify that all persons visible on site are wearing high-visibility PPE (vest and hard hat).
{RULES_AND_FORMAT}
Return JSON exactly in this schema:
{{
  "check_type": "PPE",
  "status": "Compliant/Not Compliant",
  "findings": [
    {{
      "timestamp_range": ["start mm:ss", "end mm:ss"],
      "observation": "string",
      "confidence": "low|medium|high",
      "evidence_timestamps": ["t1 mm:ss", "t2 mm:ss"]
    }}
  ],
  "recommendations": ["string", "..."]
}}
"""

SIGNAGE_PROMPT = f"""You are a UK worksite safety compliance agent.

Your job is to produce a compliance report for Chapter 8 Signage Compliance.
Check: Verify that proper warning signs are placed in accordance with UK Chapter 8 rules.
{RULES_AND_FORMAT}
Return JSON exactly in this schema:
{{
  "check_type": "CHAPTER_8_SIGNAGE",
  "status": "Compliant/Not Compliant",
  "findings": [
    {{
      "timestamp_range": ["start mm:ss", "end mm:ss"],
      "observation": "string",
      "confidence": "low|medium|high",
      "evidence_timestamps": ["t1 mm:ss", "t2 mm:ss"]
    }}
  ],
  "recommendations": ["string", "..."]
}}
"""


# -----------------------------
# Model runner
# -----------------------------

def load_model(cfg: ModelConfig):
    transformers.set_seed(cfg.seed)

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model_name,
        dtype=cfg.dtype,
        device_map="auto",
        attn_implementation=cfg.attn_implementation,
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(cfg.model_name)

    processor.image_processor.size = {
        "shortest_edge": cfg.min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": cfg.max_vision_tokens * PIXELS_PER_TOKEN,
    }
    processor.video_processor.size = {
        "shortest_edge": cfg.min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": cfg.max_vision_tokens * PIXELS_PER_TOKEN,
    }

    return model, processor


def run_video_json_report(
    model,
    processor,
    video_path: Path,
    prompt: str,
    cfg: ModelConfig,
) -> Dict[str, Any]:
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=cfg.fps_for_prompt,
    ).to(model.device)

    # generated_ids = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,  # deterministic
        temperature=0.0,  # safe with do_sample=False; keeps it stable
        top_p=1.0,
        repetition_penalty=1.05,  # small nudge to avoid looping
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(f'Output: {output_text}')
    return coerce_json(output_text)


# -----------------------------
# Robust JSON parsing
# -----------------------------

def coerce_json(text: str) -> Dict[str, Any]:
    """
    Cosmos/Qwen outputs might include extra text; try to extract the first JSON object.
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # First try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract JSON block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON object in model output:\n{text}")

    json_str = match.group(0)
    # Remove trailing junk after the last brace (rare, but safe)
    json_str = json_str[: json_str.rfind("}") + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse extracted JSON. Error={e}\nExtracted:\n{json_str}\n\nFull:\n{text}") from e


# -----------------------------
# Evidence frame extraction
# -----------------------------

def get_video_duration_s(video_path: Path) -> float:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
    # Fallback: decode frames and estimate (slower)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        last_pts = None
        for frame in container.decode(video=0):
            last_pts = frame.pts
        if last_pts is None or stream.time_base is None:
            return 0.0
        return float(last_pts * stream.time_base)


def extract_frame_at_second(
    video_path: Path,
    t_s: int,
    out_path: Path,
) -> Path:
    """
    Seek and decode closest frame to t_s (seconds) and save as PNG.
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base

        # Clamp seek target
        duration = get_video_duration_s(video_path)
        t_s = int(max(0, min(t_s, math.floor(duration))) if duration > 0 else max(0, t_s))

        # Convert seconds to PTS units (microseconds-based seek)
        seek_ts = int(t_s / time_base) if time_base is not None else t_s
        container.seek(seek_ts, stream=stream)

        for frame in container.decode(video=0):
            # Find first frame at or after t_s (best-effort)
            if frame.pts is None or time_base is None:
                img = frame.to_image()
                img.save(out_path)
                return out_path

            frame_time = float(frame.pts * time_base)
            if frame_time >= t_s:
                img = frame.to_image()
                img.save(out_path)
                return out_path

        # Fallback: if nothing found after seek, decode last frame
        container.seek(0, stream=stream)
        last = None
        for frame in container.decode(video=0):
            last = frame
        if last is None:
            raise RuntimeError("Could not decode any frames from the video.")
        last.to_image().save(out_path)
        return out_path


def _normalize_checks(raw_checks: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_checks, dict):
        # If it already looks like a single check object, wrap it.
        if any(k in raw_checks for k in ("check_type", "status", "findings")):
            return [raw_checks]
        return [c for c in raw_checks.values() if isinstance(c, dict)]
    if isinstance(raw_checks, list):
        return [c for c in raw_checks if isinstance(c, dict)]
    return []


def _normalize_findings(raw_findings: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_findings, dict):
        # If it already looks like a single finding object, wrap it.
        if any(k in raw_findings for k in ("timestamp_range", "observation", "confidence", "evidence_timestamps")):
            return [raw_findings]
        return [f for f in raw_findings.values() if isinstance(f, dict)]
    if isinstance(raw_findings, list):
        return [f for f in raw_findings if isinstance(f, dict)]
    return []


def collect_evidence_timestamps_by_check(report: Dict[str, Any]) -> Dict[str, List[int]]:
    result: Dict[str, List[int]] = {}
    for chk in _normalize_checks(report.get("checks", [])):
        check_type = chk.get("check_type", "UNKNOWN")
        ts: List[int] = []
        for finding in _normalize_findings(chk.get("findings", [])):
            for t in finding.get("evidence_timestamps", []) or []:
                if isinstance(t, str):
                    try:
                        # Parse "mm:ss" or "mm:ss.ff" format
                        parts = t.strip().split(":")
                        if len(parts) == 2:
                            m = int(parts[0])
                            # handle potential .ff (milliseconds/frames)
                            s_parts = parts[1].split(".")
                            s = int(s_parts[0])
                            ts.append(m * 60 + s)
                    except ValueError:
                        pass
                elif isinstance(t, (int, float)):
                    ts.append(int(t))
        if ts:
            if check_type not in result:
                result[check_type] = []
            result[check_type].extend(ts)
    
    # Deduplicate and sort
    for k in result:
        result[k] = sorted(set(result[k]))
    return result


# -----------------------------
# PDF report generator
# -----------------------------

def build_pdf_report(
    report: Dict[str, Any],
    evidence_images_by_check: Dict[str, List[Tuple[int, Path]]],
    out_pdf: Path,
) -> Path:
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    width, height = A4

    def write_line(y, txt, size=11, x=2*cm):
        c.setFont("Helvetica", size)
        
        # Wrap text to fit within page width margins
        max_text_width = width - (x + 2*cm) # Leaves 2cm margin on right
        
        text_obj = c.beginText(x, y)
        text_obj.setFont("Helvetica", size)
        
        # Simple word wrap loop finding break points
        words = txt.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            # Check width of tentative line
            line_width = c.stringWidth(" ".join(current_line), "Helvetica", size)
            if line_width > max_text_width:
                # Remove word that caused overflow
                if len(current_line) > 1:
                    current_line.pop()
                    lines.append(" ".join(current_line))
                    current_line = [word] # Start new line with the overflowing word
                else:
                    # Single very long word, just force it and accept some overflow
                    lines.append(" ".join(current_line))
                    current_line = []
        if current_line:
            lines.append(" ".join(current_line))
        
        # Draw the wrapped lines
        for text_line in lines:
            text_obj.textLine(text_line)
        
        c.drawText(text_obj)
        
        # Calculate new Y offset based on number of wrapped lines drawn. TextLine adds trailing space, approx 1.2 * size.
        # But ReportLab textLine standard leading is roughly 1.2 * fontsize.
        # Original simple drop was 0.6*cm. Let's return new Y.
        lines_drawn = len(lines) if lines else 1
        return y - (0.5 * cm * lines_drawn) - (0.1 * cm)

    y = height - 2 * cm

    video_id = report.get("video_id", "unknown")
    y = write_line(y, f"Streetwork compliance report (Video ID: {video_id})", size=16)

    summary = report.get("summary", {})


    y -= 0.3 * cm
    y = write_line(y, "Checks:", size=12)

    for chk in _normalize_checks(report.get("checks", [])):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y, f"{chk.get('check_type', 'UNKNOWN')}  -  {chk.get('status', 'unknown')}")
        y -= 0.7 * cm

        c.setFont("Helvetica", 11)
        for f in _normalize_findings(chk.get("findings", []))[:6]:
            tr = f.get("timestamp_range", ["?", "?"])
            obs = f.get("observation", "")
            conf = f.get("confidence", "unknown")
            y = write_line(y, f"- [{tr[0]} - {tr[1]}] ({conf}) {obs}", size=10)

            if y < 4 * cm:
                c.showPage()
                y = height - 2 * cm

        y -= 0.2 * cm

    y = write_line(y, "Recommendations:", size=12)
    for rec in report.get("recommendations", [])[:10]:
        y = write_line(y, f"- {rec}", size=11)

    notes = report.get("notes", [])
    if notes:
        y -= 0.2 * cm
        y = write_line(y, "Notes:", size=12)
        for n in notes[:8]:
            y = write_line(y, f"- {n}", size=11)

    # Evidence pages
    if evidence_images_by_check:
        c.showPage()
        y = height - 2 * cm
        y = write_line(y, "Evidence Frames:", size=16)
        y = y - (0.5 * cm)

        for check_type, img_list in evidence_images_by_check.items():
            if not img_list:
                continue

            if y < 4 * cm:
                c.showPage()
                y = height - 2 * cm

            c.setFont("Helvetica-Bold", 14)
            c.drawString(2*cm, y, f"{check_type} evidence frames")
            y = y - (0.8 * cm)

            for t_s, img_path in img_list:
                if y < 8 * cm:
                    c.showPage()
                    y = height - 2 * cm

                y = write_line(y, f"Timestamp: {t_s}s", size=12)

                # Fit image
                max_w = width - 4 * cm
                max_h = 9 * cm
                with Image.open(img_path) as im:
                    iw, ih = im.size
                scale = min(max_w / iw, max_h / ih)
                draw_w, draw_h = iw * scale, ih * scale

                c.drawImage(str(img_path), 2*cm, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
                y = y - (draw_h + 1.0 * cm)
            
            y = y - (0.5 * cm)

    c.save()
    return out_pdf


# -----------------------------
# Main entry
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate UK streetworks compliance report from video")
    ap.add_argument("-i", "--input", required=True, help="Input video path")
    ap.add_argument("-o", "--output", default="outputs", help="Output directory path")
    args = ap.parse_args()

    cfg = ModelConfig()

    video_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_id = video_path.stem

    model, processor = load_model(cfg)

    print("Running check 1: Barrier Continuity...")
    barrier_report = run_video_json_report(
        model=model,
        processor=processor,
        video_path=video_path,
        prompt=BARRIER_PROMPT,
        cfg=cfg,
    )

    print("Running check 2: PPE Compliance...")
    ppe_report = run_video_json_report(
        model=model,
        processor=processor,
        video_path=video_path,
        prompt=PPE_PROMPT,
        cfg=cfg,
    )

    print("Running check 3: Chapter 8 Signage...")
    signage_report = run_video_json_report(
        model=model,
        processor=processor,
        video_path=video_path,
        prompt=SIGNAGE_PROMPT,
        cfg=cfg,
    )

    # Combine into a single report
    checks = [barrier_report, ppe_report, signage_report]
    all_recommendations = []
    statuses = []
    
    for c in checks:
        if "recommendations" in c:
            all_recommendations.extend(c["recommendations"])
        if "status" in c:
            statuses.append(c["status"])

    if "fail" in statuses:
        overall_risk = "high"
    elif "partial" in statuses:
        overall_risk = "medium"
    elif "pass" in statuses:
        overall_risk = "low"
    else:
        overall_risk = "unknown"

    report = {
        "video_id": video_id,
        "summary": {
            "overall_risk_level": overall_risk,
            "key_findings": [f"Based on 3 sub-checks, the site is classified as {overall_risk} risk."]
        },
        "checks": checks,
        "recommendations": list(set(all_recommendations)), # deduplicate identical recommendations
        "notes": ["Report aggregated from 3 sequential LVLM calls."]
    }

    # Save raw JSON
    json_path = out_dir / f"{video_id}_safety_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Evidence frames
    evidence_ts_by_check = collect_evidence_timestamps_by_check(report)
    evidence_images_by_check: Dict[str, List[Tuple[int, Path]]] = {}
    
    total_extracted = 0
    for check_type, ts_list in evidence_ts_by_check.items():
        evidence_images_by_check[check_type] = []
        for t_s in ts_list:
            if total_extracted >= 30:  # cap to avoid huge PDFs
                break
            img_path = out_dir / f"{video_id}_evidence_{check_type}_{t_s:06d}s.png"
            extract_frame_at_second(video_path, t_s, img_path)
            evidence_images_by_check[check_type].append((t_s, img_path))
            total_extracted = total_extracted + 1

    # PDF
    pdf_path = out_dir / f"{video_id}_safety_report.pdf"
    build_pdf_report(report, evidence_images_by_check, pdf_path)

    print(f"Saved JSON: {json_path}")
    print(f"Saved PDF : {pdf_path}")
    if total_extracted > 0:
        print(f"Saved {total_extracted} evidence frames in {out_dir}/")


if __name__ == "__main__":
    main()
