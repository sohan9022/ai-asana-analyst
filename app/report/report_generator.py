import os
import json
import tempfile
import cv2
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from app.report.ai_coach import get_yoga_wisdom

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

PRIMARY   = HexColor("#0f3460")
GOLD      = HexColor("#e94560")
GREEN     = HexColor("#28a745")
RED       = HexColor("#dc3545")
LIGHT_BG  = HexColor("#f8f9fa")
BORDER    = HexColor("#dee2e6")
DARK      = HexColor("#343a40")
GRAY      = HexColor("#6c757d")

def S(name, **kw):
    base = getSampleStyleSheet()["Normal"]
    return ParagraphStyle(name, parent=base, **kw)

def calculate_score(all_violations, total_frames):
    """Calculate score — penalise per violation found."""
    if total_frames == 0:
        return 0.0
    total_violations = sum(len(v) for v in all_violations)
    max_violations   = total_frames * 5  # Assume max 5 violations possible
    score = max(0, 100 - (total_violations / max_violations * 100))
    return round(score, 1)

def generate_report(pose_name, analysis_results, output_path):
    """
    Generate a PDF report from analysis results.

    Args:
        pose_name        : str
        analysis_results : list of (annotated_frame, violations, joint_angles, status)
        output_path      : str — where to save the PDF
    """
    doc   = SimpleDocTemplate(output_path, pagesize=letter,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=0.75*inch,  bottomMargin=0.75*inch)
    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    header_table = Table([[
        Paragraph("AI ASANA ANALYST", S("h", fontSize=22, fontName="Helvetica-Bold",
                                         textColor=white, alignment=TA_CENTER)),
    ], [
        Paragraph("Posture Diagnostic Report", S("s", fontSize=12, fontName="Helvetica",
                                                   textColor=HexColor("#adb5bd"), alignment=TA_CENTER)),
    ]], colWidths=[7.0*inch])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), PRIMARY),
        ("TOPPADDING",    (0,0), (-1,-1), 14),
        ("BOTTOMPADDING", (0,0), (-1,-1), 14),
        ("LEFTPADDING",   (0,0), (-1,-1), 20),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
    ]))

    # ── Meta info ─────────────────────────────────────────────────────────────
    story.append(Paragraph(
        f"<b>Pose:</b> {pose_name}  &nbsp;&nbsp;  "
        f"<b>Date:</b> {datetime.now().strftime('%d %b %Y %H:%M')}  &nbsp;&nbsp;  "
        f"<b>Frames Analysed:</b> {len(analysis_results)}",
        S("meta", fontSize=10, textColor=GRAY, spaceAfter=4)
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 10))

    # ── Score ─────────────────────────────────────────────────────────────────
    all_violations   = [r[1] for r in analysis_results if r[3] == "ok"]
    undetectable     = sum(1 for r in analysis_results if r[3] == "undetectable")
    score            = calculate_score(all_violations, len(all_violations))
    score_color      = GREEN if score >= 85 else (HexColor("#fd7e14") if score >= 60 else RED)

    score_table = Table([[
        Paragraph(f"{score}%", S("sc", fontSize=36, fontName="Helvetica-Bold",
                                   textColor=score_color, alignment=TA_CENTER)),
        Paragraph("Overall Posture Score", S("sl", fontSize=12, textColor=DARK)),
    ]], colWidths=[1.5*inch, 5.5*inch])
    score_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), LIGHT_BG),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 20),
        ("BOX",           (0,0), (-1,-1), 1, BORDER),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 12))

    if undetectable > 0:
        story.append(Paragraph(
            f"⚑  {undetectable} frame(s) were undetectable and excluded from scoring.",
            S("note", fontSize=9, textColor=HexColor("#c96a00"), fontName="Helvetica-Oblique")
        ))
        story.append(Spacer(1, 8))

    # ── Violations found ──────────────────────────────────────────────────────
    # ── Violations found ──────────────────────────────────────────────────────
    story.append(Paragraph("Violations Detected", S("vh", fontSize=14,
                fontName="Helvetica-Bold", textColor=PRIMARY, spaceAfter=6)))

    # Collect unique violations
    seen       = set()
    unique_v   = []
    for _, violations, angles, status in analysis_results:
        if status != "ok":
            continue
        for v in violations:
            key = v["joint"]
            if key not in seen:
                seen.add(key)
                unique_v.append((v, angles))

    if not unique_v:
        story.append(Paragraph("✓  No violations detected — excellent form!",
                                S("ok", fontSize=11, textColor=GREEN)))
    else:
        v_header = [
            # Center align the numerical column headers
            Paragraph(c, S("th", fontName="Helvetica-Bold", fontSize=9, textColor=white, 
                           alignment=TA_CENTER if c in ["Detected", "Target"] else 0))
            for c in ["Joint", "Detected", "Target", "Mistake", "Correction"]
        ]
        v_data = [v_header]
        for v, angles in unique_v:
            # FIX 1: Use a standard hyphen and safe HTML entity for the degree symbol
            detected_str = f"{v['detected']}&deg;" if v['detected'] else "—"
            target_str   = f"{v['min']}-{v['max']}&deg;" if v['min'] else "—"
            
            v_data.append([
                Paragraph(v["joint"].replace("_", " ").title(),
                          S("td", fontSize=9, textColor=DARK)),
                # Center align the numerical data
                Paragraph(detected_str, S("td2", fontSize=9, textColor=RED, fontName="Helvetica-Bold", alignment=TA_CENTER)),
                Paragraph(target_str,   S("td3", fontSize=9, textColor=GREEN, alignment=TA_CENTER)),
                Paragraph(v["mistake"],    S("td4", fontSize=9, textColor=DARK)),
                Paragraph(v["correction"], S("td5", fontSize=9, textColor=DARK)),
            ])

        # FIX 2: Slightly adjusted colWidths to give the Correction column more breathing room
        v_table = Table(v_data, colWidths=[1.0*inch, 0.7*inch, 0.8*inch, 1.9*inch, 2.6*inch])
        v_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  PRIMARY),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [white, LIGHT_BG]),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
            # FIX 3: Force all text to align to the TOP of the cell
            ("VALIGN",        (0,0), (-1,-1), "TOP"), 
            ("LINEBELOW",     (0,0), (-1,-1), 0.5, BORDER),
            ("BOX",           (0,0), (-1,-1), 1, BORDER),
        ]))
        story.append(v_table)

    story.append(Spacer(1, 14))

    # ── Annotated image ───────────────────────────────────────────────────────
    story.append(Paragraph("Annotated Frame", S("ah", fontSize=14,
                fontName="Helvetica-Bold", textColor=PRIMARY, spaceAfter=6)))

    # Find first frame with violations to show
    best_frame = None
    for frame, violations, _, status in analysis_results:
        if status == "ok" and len(violations) > 0:
            best_frame = frame
            break
    if best_frame is None:
        for frame, _, _, status in analysis_results:
            if status == "ok":
                best_frame = frame
                break

    if best_frame is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()  
        cv2.imwrite(tmp.name, best_frame)
        img_w = 5.0 * inch
        img_h = img_w * best_frame.shape[0] / best_frame.shape[1]
        story.append(RLImage(tmp.name, width=img_w, height=img_h))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "AI Asana Analyst  ·  Automated Posture Report  ·  University Project",
        S("footer", fontSize=8, textColor=GRAY, alignment=TA_CENTER)
    ))

    # ── Phase 6: Generative AI Yoga Wisdom ────────────────────────────────────
    story.append(Spacer(1, 14))
    story.append(Paragraph("Yoga Wisdom & AI Coaching", S("gh", fontSize=14, 
                fontName="Helvetica-Bold", textColor=PRIMARY, spaceAfter=8)))
    
    # Generate the AI summary
    top_mistakes = unique_v[:2] # Pass top mistakes found
    wisdom_text = get_yoga_wisdom(pose_name, score, top_mistakes)
    
    # Create a highlighted box for the AI feedback
    wisdom_table = Table([[
        Paragraph(f"<i>{wisdom_text}</i>", S("gt", fontSize=10, 
                  textColor=DARK, leading=14, alignment=TA_JUSTIFY))
    ]], colWidths=[7.0*inch])
    
    wisdom_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), HexColor("#e8f4f8")),
        ("BOX", (0,0), (-1,-1), 0.5, PRIMARY),
        ("LEFTPADDING", (0,0), (-1,-1), 15),
        ("RIGHTPADDING", (0,0), (-1,-1), 15),
        ("TOPPADDING", (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),
    ]))
    story.append(wisdom_table)

    doc.build(story)

    # <-- ADD THIS CLEANUP BLOCK -->
    if best_frame is not None and os.path.exists(tmp.name):
        os.unlink(tmp.name)
        
    return output_path

def generate_ai_assistance(pose_data):
    """
    pose_data should be a dictionary like:
    {'name': 'Warrior II', 'accuracy': 72, 'violations': ['Right knee not bent', 'Arms not level']}
    """
    
    # 1. Check if we actually have data, otherwise it will feel "hardcoded"
    if not pose_data or not pose_data.get('violations'):
        return "Maintain your focus and keep your core engaged."

    # 2. Dynamic Prompting (The key to fixing the 'same line' issue)
    prompt = f"""
    You are a professional Yoga AI Coach. 
    The user is performing {pose_data['name']} with {pose_data['accuracy']}% accuracy.
    Specific skeletal errors detected: {", ".join(pose_data['violations'])}.

    Provide a VISUALLY IMPRESSIVE report using Markdown:
    1. A section titled "🚨 CRITICAL CORRECTIONS" with bullet points.
    2. A section titled "💡 PRO TIPS" explaining the bio-mechanics of {pose_data['name']}.
    Keep it concise and encouraging.
    """

    response = model.generate_content(prompt)
    return response.text

