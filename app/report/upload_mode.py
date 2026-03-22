import os
import tempfile
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

from app.report.analyzer import analyze_image, analyze_video
from app.report.report_generator import generate_report

# Load your Gemini API Key securely
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_ai_coach_feedback(pose_name, errors_text):
    """Generates dynamic, strictly formatted AI feedback using Gemini."""
    if not errors_text or errors_text == "None":
        prompt = f"The user just performed {pose_name} with perfect skeletal alignment. Give a short, 2-sentence encouraging expert yoga tip to maintain this excellent form."
    else:
        prompt = f"""
        Act as an elite Yoga Instructor. The user is performing {pose_name}.
        The computer vision model detected these specific biomechanical errors: {errors_text}.
        
        Format your response EXACTLY like this using Markdown:
        ### 🚨 Priority Corrections
        * **[Target Body Part]:** [Direct, actionable step to fix the error]
        
        ### 💡 Pro Tip
        * [One advanced breathing or alignment tip for {pose_name}]
        """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"*(AI Coach is currently offline. Please check API Key.)* Error: {e}"

def run_upload_mode():
    st.title("AI Asana Analyst — Upload & Report")
    st.markdown("Upload a photo or video of your pose and get a full diagnostic report.")

    pose_name = st.selectbox(
        "Select Pose",
        ["Warrior II", "Tree Pose", "Triangle Pose"]
    )

    guidance = {
        "Warrior II":    "📷 Required camera angle: Side profile view",
        "Tree Pose":     "📷 Required camera angle: Front facing view",
        "Triangle Pose": "📷 Required camera angle: Side profile view",
    }
    st.info(guidance[pose_name])

    uploaded_file = st.file_uploader(
        "Upload your pose image or video",
        type=["jpg", "jpeg", "png", "mp4"]
    )

    if uploaded_file is None:
        return

    # Save uploaded file to temp location
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    tmp_input = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_input.write(uploaded_file.read())
    tmp_input.close()

    st.success(f"File uploaded: {uploaded_file.name}")

    if st.button("Analyse Pose"):
        with st.spinner("Analysing your pose..."):

            # ── Run analysis ──────────────────────────────────────────────────
            if suffix in [".jpg", ".jpeg", ".png"]:
                frame, violations, angles, status = analyze_image(
                    tmp_input.name, pose_name
                )
                results = [(frame, violations, angles, status)]
            else:
                results = analyze_video(tmp_input.name, pose_name)

            # ── Generate PDF report ───────────────────────────────────────────
            tmp_report = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            tmp_report.close()
            
            report_path = generate_report(pose_name, results, tmp_report.name)
            
            # ── Show summary on screen ────────────────────────────────────────
            ok_results   = [r for r in results if r[3] == "ok"]
            undetectable = len(results) - len(ok_results)

            if undetectable == len(results):
                st.error("No pose detected in the uploaded file. "
                         "Please ensure you are clearly visible and try again.")
            else:
                # Extract unique errors
                all_v = [v for _, violations, _, _ in ok_results for v in violations]
                unique_joints = {v["joint"] for v in all_v}
                
                # Create a clean string of errors to feed to the AI
                unique_mistakes_list = list({v['mistake'] for v in all_v})
                errors_str = ", ".join(unique_mistakes_list) if unique_mistakes_list else "None"

                # ── The Visually Impressive AI Dashboard ──────────────────────
                st.markdown("---")
                st.subheader("🤖 Smart AI Consultation")
                
                # Calculate a mock visual score based on error count
                score = max(0, 100 - (len(unique_mistakes_list) * 15))
                
                with st.spinner("Consulting AI Yoga Coach..."):
                    ai_advice = get_ai_coach_feedback(pose_name, errors_str)

                # The Premium UI Container
                with st.container(border=True):
                    cols = st.columns([1, 3])
                    
                    with cols[0]:
                        st.metric(label="Form Accuracy", value=f"{score}%")
                        if score >= 85:
                            st.success("Excellent Alignment!")
                        elif score >= 70:
                            st.warning("Good, but needs work.")
                        else:
                            st.error("Correction Required")
                            
                    with cols[1]:
                        # Render the perfectly formatted Gemini Markdown
                        st.markdown(ai_advice)

                # Show the raw technical data below the AI summary
                st.markdown("---")
                with st.expander("Show Raw Biomechanical Data"):
                    if not unique_joints:
                        st.success("✅ Excellent form! No violations detected.")
                    else:
                        st.warning(f"⚠️ {len(unique_joints)} joint violation(s) detected:")
                        for v in all_v:
                            if v["joint"] in unique_joints:
                                unique_joints.discard(v["joint"])
                                st.error(f"**{v['mistake']}**")
                                st.info(f"💡 {v['correction']}")

                if undetectable > 0:
                    st.warning(f"{undetectable} frame(s) were undetectable and excluded.")

                # ── Download button ───────────────────────────────────────────
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Full PDF Report",
                        data=f,
                        file_name=f"asana_report_{pose_name.replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
            
            # Clean up the generated files
            os.unlink(report_path) 
        os.unlink(tmp_input.name)