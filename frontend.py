import streamlit as st
import requests

API_BASE = "http://localhost:8000"
st.set_page_config(page_title="🩹 Skensure", layout="wide")

if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.image_id = None
    st.session_state.disease = None
    st.session_state.messages = []

def api_post_json(endpoint, data):
    """POST JSON data"""
    try:
        resp = requests.post(f"{API_BASE}/{endpoint}", json=data)
        resp.raise_for_status()
        return resp.json()
    except:
        st.error(f"API Error at {endpoint}")
        return None

def api_post_files(endpoint, files_data, form_data):
    """POST file uploads"""
    try:
        resp = requests.post(f"{API_BASE}/{endpoint}", files=files_data, data=form_data)
        resp.raise_for_status()
        return resp.json()
    except:
        st.error(f"Upload failed")
        return None

# === SIDEBAR ===
with st.sidebar:
    st.title("🩹 Skensure")
    
    if st.session_state.user_id is None:
        tab1, tab2 = st.tabs(["Login", "Signup"])
        with tab1:
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                result = api_post_json("login", {"email": email, "password": password})
                if result:
                    st.session_state.user_id = 1  # Update with real user ID later
                    st.success("Logged in!")
                    st.rerun()
        
        with tab2:
            email = st.text_input("New Email")
            password = st.text_input("New Password", type="password")
            if st.button("Signup"):
                result = api_post_json("signup", {"email": email, "password": password})
                if result:
                    st.session_state.user_id = 1
                    st.success("Signed up!")
                    st.rerun()
    else:
        st.success("✅ Logged in")
        if st.button("Logout"):
            st.session_state.user_id = None
            st.rerun()
        
        st.divider()
        uploaded_file = st.file_uploader("📸 Upload Image", type=["jpg","png","jpeg"])
        
        if uploaded_file and st.button("Analyze Image"):
            files = {"file": uploaded_file}
            form_data = {"user_id": st.session_state.user_id}
            
            # Upload
            upload_result = api_post_files("upload-image", files, form_data)
            if upload_result:
                st.session_state.image_id = upload_result["image_id"]
                
                # Predict
                pred_result = api_post_json(f"predict/{upload_result['image_id']}", {})
                if pred_result:
                    st.session_state.disease = pred_result["prediction"]
                    st.metric("Detected", pred_result["prediction"])
                    st.caption(f"Confidence: {pred_result['confidence']:.1%}")

# === MAIN CHAT ===
if st.session_state.user_id:
    st.title("💬 Dermatology Assistant")
    
    # Show messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your skin condition..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = api_post_json("chat", {
                    "user_id": st.session_state.user_id,
                    "message": prompt,
                    "image_id": st.session_state.image_id
                })
                if result:
                    reply = result["reply"]
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.markdown(reply)
        
        st.rerun()

else:
    st.info("👆 Login first!")
