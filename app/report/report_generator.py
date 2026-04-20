import os
import tempfile
import cv2
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image as RLImage, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

from app.report.ai_coach import get_yoga_wisdom

from google import genai
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# 🔐 Load API Key (NEW SDK)
# ─────────────────────────────────────────────
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ─────────────────────────────────────────────
# 🎨 Colors
# ─────────────────────────────────────────────
PRIMARY   = HexColor("#0f3460")
GREEN     = HexColor("#28a745")
RED       = HexColor("#dc3545")
LIGHT_BG  = HexColor("#f8f9fa")
BORDER    = HexColor("#dee2e6")
DARK      = HexColor("#343a40")
GRAY      = HexColor("#6c757d")

# ─────────────────────────────────────────────
# ✍️ Style helper
# ─────────────────────────────────────────────
def S(name, **kw):
    base = getSampleStyleSheet()["Normal"]
    return ParagraphStyle(name, parent=base, **kw)

# ─────────────────────────────────────────────
# 📊 Score calculation
# ─────────────────────────────────────────────
def calculate_score(all_violations, total_frames):
    if total_frames == 0:
        return 0.0

    total_violations = sum(len(v) for v in all_violations)
    max_violations   = total_frames * 5

    score = max(0, 100 - (total_violations / max_violations * 100))
    return round(score, 1)

# ─────────────────────────────────────────────
# 📄 Report generation
# ─────────────────────────────────────────────
def generate_report(pose_name, analysis_results, output_path):

    if not analysis_results:
        raise ValueError("No analysis results provided")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    story = []

    # ── Header ──
    header_table = Table([[
        Paragraph("AI ASANA ANALYST",
                  S("h", fontSize=22, fontName="Helvetica-Bold",
                    textColor=white, alignment=TA_CENTER))
    ], [
        Paragraph("Posture Diagnostic Report",
                  S("s", fontSize=12, textColor=HexColor("#adb5bd"),
                    alignment=TA_CENTER))
    ]], colWidths=[7.0*inch])

    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), PRIMARY),
        ("TOPPADDING", (0,0), (-1,-1), 14),
        ("BOTTOMPADDING", (0,0), (-1,-1), 14),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ]))

    story.append(header_table)
    story.append(Spacer(1, 10))

    # ── Meta ──
    story.append(Paragraph(
        f"<b>Pose:</b> {pose_name} &nbsp;&nbsp;"
        f"<b>Date:</b> {datetime.now().strftime('%d %b %Y %H:%M')} &nbsp;&nbsp;"
        f"<b>Frames:</b> {len(analysis_results)}",
        S("meta", fontSize=10, textColor=GRAY)
    ))

    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 10))

    # ── Score ──
    all_violations = [r[1] for r in analysis_results if r[3] == "ok"]
    undetectable   = sum(1 for r in analysis_results if r[3] == "undetectable")

    score = calculate_score(all_violations, len(analysis_results))

    score_color = GREEN if score >= 85 else (
        HexColor("#fd7e14") if score >= 60 else RED
    )

    score_table = Table([[
        Paragraph(f"{score}%",
                  S("sc", fontSize=36, fontName="Helvetica-Bold",
                    textColor=score_color)),
        Paragraph("Overall Posture Score",
                  S("sl", fontSize=12, textColor=DARK))
    ]])

    story.append(score_table)
    story.append(Spacer(1, 10))

    if undetectable:
        story.append(Paragraph(
            f"{undetectable} frame(s) undetectable (excluded).",
            S("note", fontSize=9, textColor=RED)
        ))

    # ── Violations ──
    seen = set()
    unique_v = []

    for frame, violations, angles, status in analysis_results:
        if status != "ok":
            continue
        for v in violations:
            if v["joint"] not in seen:
                seen.add(v["joint"])
                unique_v.append(v)

    if not unique_v:
        story.append(Paragraph("No violations detected ✔",
                               S("ok", textColor=GREEN)))
    else:
        for v in unique_v:
            detected = (
                f"{v['detected']}°"
                if v['detected'] is not None else "—"
            )
            target = (
                f"{v['min']}-{v['max']}°"
                if v['min'] is not None else "—"
            )

            story.append(Paragraph(
                f"<b>{v['joint']}:</b> {v['mistake']} "
                f"(Detected: {detected}, Target: {target})",
                S("v", fontSize=10)
            ))

    story.append(Spacer(1, 12))

    # ── Image ──
    best_frame = None
    for frame, violations, _, status in analysis_results:
        if status == "ok":
            best_frame = frame
            break

    tmp = None
    if best_frame is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()

        cv2.imwrite(tmp.name, best_frame)

        img_w = 5 * inch
        img_h = img_w * best_frame.shape[0] / best_frame.shape[1]

        story.append(RLImage(tmp.name, width=img_w, height=img_h))

    # ── AI Coaching (NEW SDK USED HERE) ──
    story.append(Spacer(1, 12))
    story.append(Paragraph("AI Coaching",
                           S("h2", fontSize=14, textColor=PRIMARY)))

    top_mistakes = unique_v[:2]

    try:
        prompt = f"""
        You are a Yoga AI Coach.

        Pose: {pose_name}
        Score: {score}%
        Issues: {", ".join([v['mistake'] for v in top_mistakes])}

        Give short corrections and tips.
        """

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        wisdom = response.text

    except Exception:
        wisdom = "AI coaching unavailable."

    story.append(Paragraph(wisdom,
                           S("ai", fontSize=10, alignment=TA_JUSTIFY)))

    # ── Build PDF ──
    doc.build(story)

    # Cleanup
    if tmp and os.path.exists(tmp.name):
        os.unlink(tmp.name)

    return output_path


# ─────────────────────────────────────────────
# 🤖 Standalone AI helper (NEW SDK)
# ─────────────────────────────────────────────
def generate_ai_assistance(pose_data):

    if not pose_data or not pose_data.get('violations'):
        return "Maintain your posture and breathing."

    prompt = f"""
    You are a professional Yoga AI Coach.

    Pose: {pose_data['name']}
    Accuracy: {pose_data['accuracy']}%
    Issues: {", ".join(pose_data['violations'])}

    Provide:
    - Key corrections
    - Helpful tips
    Keep it concise and motivating.
    """

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text

    except Exception as e:
        return f"AI coaching unavailable: {str(e)}"