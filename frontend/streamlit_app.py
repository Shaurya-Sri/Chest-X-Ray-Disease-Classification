import io
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# CONFIG 
st.set_page_config(
    page_title="🫁 Chest X-Ray Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# Severity & Color Mapping
SEVERITY = {
    "Pneumothorax": "critical", "Pneumonia": "critical",
    "Edema": "high", "Consolidation": "high", "Effusion": "high", "Lung Opacity": "high",
    "Cardiomegaly": "moderate", "Atelectasis": "moderate", "Mass": "moderate",
    "Nodule": "moderate", "Emphysema": "moderate", "Fracture": "moderate",
    "Lung Lesion": "moderate", "Enlarged Cardiomediastinum": "moderate",
    "Fibrosis": "low", "Hernia": "low", "Infiltration": "low",
    "Pleural_Thickening": "low", "Support Devices": "low",
    "No Finding": "none",
}

COLOR = {
    "critical": "#e74c3c",
    "high": "#e67e22",
    "moderate": "#f39c12",
    "low": "#2ecc71",
    "none": "#3498db",
}

#  SIDEBAR 
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
        caption="Sample Chest X-Ray ",
        width='stretch'
    )
    
    st.markdown("---")
    st.markdown("### 🧠 About the Project")
    st.markdown("""
    Chest X-Ray Pathology Classifier is an AI-powered web application designed to assist in the early detection and classification of thoracic diseases from chest radiographs.The application can detect up to 20 different thoracic pathologies simultaneously
    - Trained with asymmetric loss (False Negatives penalized 5×)
    - Focuses on clinical priority & severity
    - Designed to assist radiologists and medical professionals
    """)
    
    st.markdown("---")
    st.markdown("### Supported Pathologies")
    pathologies = sorted(SEVERITY.keys())
    for p in pathologies:
        sev = SEVERITY[p]
        col = COLOR[sev]
        st.markdown(f"• <span style='color:{col}'>**{p}**</span> ({sev})", unsafe_allow_html=True)

    st.markdown("---")
    api_status = st.empty()

#  MAIN UI 
st.title("🫁 Chest X-Ray Pathology Classifier")
st.markdown("**Tool for detecting 20 thoracic diseases from chest radiographs**")

# API Status
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

with api_status:
    if check_api():
        st.success("🟢 Backend API Connected", icon="✅")
    else:
        st.error("🔴 Backend API Offline — Start the FastAPI server", icon="❌")

tab1, tab2 = st.tabs(["🔍 Single Image Analysis", "📦 Batch Processing"])

# TAB 1: SINGLE IMAGE 
with tab1:
    col1, col2 = st.columns([1.1, 2])

    with col1:
        st.subheader("Upload Chest X-Ray")
        uploaded = st.file_uploader(
            "Choose a chest X-ray image (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            help="Best results with frontal (PA/AP) view, high-resolution images"
        )

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image",width='stretch')
            st.caption(f"📏 {img.size[0]}×{img.size[1]} px | Mode: {img.mode}")
            
            analyze_btn = st.button(
                "🔍 Analyze ",
                type="primary",
                width='stretch'
            )
        else:
            analyze_btn = False
            st.info("👆 Upload a chest X-ray to start analysis", icon="📸")

    with col2:
        if uploaded and analyze_btn:
            with st.spinner("Running..."):
                try:
                    uploaded.seek(0)
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    
                    resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)

                    if resp.status_code == 200:
                        data = resp.json()
                        probs = data["probabilities"]
                        preds = data["predictions"]
                        findings = data.get("top_findings", [])

                        st.subheader("📋 Analysis Results")
                        
                        if "No Finding" in findings and len(findings) == 1:
                            st.success("✅ **No significant pathology detected**", icon="🎉")
                        else:
                            st.warning("⚠️ Pathologies Detected", icon="🔬")
                            actual_findings = [f for f in findings if f != "No Finding"]
                            
                            for finding in actual_findings:
                                sev = SEVERITY.get(finding, "low")
                                col = COLOR[sev]
                                st.markdown(
                                    f"""
                                    <div style="background:{col}15; border-left:5px solid {col}; 
                                    padding:12px 16px; margin:8px 0; border-radius:6px;">
                                        <b style="color:{col}">● {finding}</b> 
                                        <span style="color:#666; font-size:0.9em;">({sev.upper()})</span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        st.divider()
                        st.subheader("📊 Prediction Probabilities :")

                        prob_df = pd.DataFrame.from_dict(
                            probs, orient="index", columns=["Probability"]
                        ).sort_values("Probability", ascending=False)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ["#e74c3c" if preds.get(cls, 0) == 1 else "#3498db" 
                                 for cls in prob_df.index[:10]]
                        
                        ax.barh(
                            prob_df.index[:10][::-1], 
                            prob_df["Probability"][:10][::-1], 
                            color=colors[::-1],
                            edgecolor="white"
                        )
                        ax.axvline(data.get("threshold_used", 0.5), color="black", linestyle="--",
                                   label=f"Threshold = {data.get('threshold_used', 0.5):.3f}")
                        ax.set_xlabel("Probability")
                        ax.set_title("Top 10 Predicted Pathologies")
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)

                        with st.expander("Show All Class Probabilities"):
                            st.dataframe(
                                prob_df.style.format({"Probability": "{:.4f}"}),
                                width='stretch'
                            )

                    else:
                        st.error(f"API Error ({resp.status_code})")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Is the API running?", icon="🔌")
                except Exception as e:
                    st.error(f"Error: {str(e)}", icon="❗")

# =TAB 2: BATCH 
with tab2:
    st.subheader("Batch Analysis")
    batch_files = st.file_uploader(
        "Upload multiple chest X-rays",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if batch_files and st.button("🚀 Analyze All ", type="primary", width='stretch'):
        with st.spinner(f"Processing {len(batch_files)} images"):
            try:
                files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in batch_files]
                resp = requests.post(f"{API_URL}/predict_batch", files=files_payload, timeout=180)

                if resp.status_code == 200:
                    results = resp.json()["results"]
                    rows = []
                    for r in results:
                        findings = r.get("top_findings", [])
                        rows.append({
                            "Filename": r["filename"],
                            "Findings": ", ".join(findings) if findings else "No Finding",
                            "Pathology Count": len([f for f in findings if f != "No Finding"])
                        })
                    
                    df = pd.DataFrame(rows)
                    st.success(f"✅ Batch complete — {len(results)} images analyzed", icon="🎯")
                    st.dataframe(df, width='stretch', hide_index=True)
                    
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Results CSV",
                        csv,
                        "chest_xray_batch_results.csv",
                        "text/csv",
                        width='stretch'
                    )
                else:
                    st.error("Batch processing failed")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        • Multi-label Chest X-ray Classification Tool • 
        
    </div>
    """,
    unsafe_allow_html=True
)