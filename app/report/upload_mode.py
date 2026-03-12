import os
import tempfile
import streamlit as st
from app.report.analyzer import analyze_image, analyze_video
from app.report.report_generator import generate_report

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

            # ── Generate report ───────────────────────────────────────────────
            tmp_report = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            tmp_report.close() # <-- ADD THIS LINE to prevent Windows file lock crashes
            
            report_path = generate_report(pose_name, results, tmp_report.name)
            # ── Show summary on screen ────────────────────────────────────────
            ok_results   = [r for r in results if r[3] == "ok"]
            undetectable = len(results) - len(ok_results)

            if undetectable == len(results):
                st.error("No pose detected in the uploaded file. "
                         "Please ensure you are clearly visible and try again.")
            else:
                all_v = [v for _, violations, _, _ in ok_results for v in violations]
                unique_joints = {v["joint"] for v in all_v}

                if not unique_joints:
                    st.success("✅ Excellent form! No violations detected.")
                else:
                    st.warning(f"⚠️ {len(unique_joints)} violation(s) detected:")
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
            
            # Clean up the generated PDF from the server so your hard drive doesn't fill up!
            os.unlink(report_path) 
        
        # Clean up the uploaded video/image
        os.unlink(tmp_input.name)