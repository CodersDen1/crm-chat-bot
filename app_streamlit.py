import streamlit as st
import requests

# URL for your FastAPI backend
BACKEND_URL = "http://localhost:7777"

st.title("Document QA Service")

# Create two tabs: one for file management and one for chat.
tab1, tab2 = st.tabs(["Upload & Manage Documents", "Chat QA"])

# ---------------------------
# Upload & Manage Documents Tab
# ---------------------------
with tab1:
    st.header("Upload & Manage Documents")
    
    # Upload options: Multiple File Upload or URL download.
    upload_method = st.radio("Select upload method", ["Upload Files", "Enter URL"])
    
    if upload_method == "Upload Files":
        # Allow multiple file uploads (including pdf)
        uploaded_files = st.file_uploader("Choose files", type=["txt", "md", "csv", "json", "pdf"], accept_multiple_files=True)
        if st.button("Upload Files"):
            if not uploaded_files:
                st.error("Please select at least one file to upload.")
            else:
                for uploaded_file in uploaded_files:
                    files = {"file": uploaded_file}
                    try:
                        response = requests.post(f"{BACKEND_URL}/upload", files=files)
                        if response.ok:
                            res_data = response.json()
                            st.success(f"File '{uploaded_file.name}' uploaded successfully! (ID: {res_data['file_id']})")
                        else:
                            st.error(f"Error uploading '{uploaded_file.name}': {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Request failed for '{uploaded_file.name}': {e}")
                    
    else:
        url_input = st.text_input("Enter file URL")
        if st.button("Upload URL"):
            if not url_input:
                st.error("Please enter a valid URL.")
            else:
                # Send the URL as a form field.
                data = {"url": url_input}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload", data=data)
                    if response.ok:
                        res_data = response.json()
                        st.success(f"File downloaded and uploaded successfully! (ID: {res_data['file_id']})")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Request failed: {e}")
    
    st.markdown("---")
    st.subheader("Uploaded Files")
    
    # Fetch the list of uploaded files.
    try:
        files_response = requests.get(f"{BACKEND_URL}/files")
        if files_response.ok:
            files_list = files_response.json()
            if not files_list:
                st.info("No files uploaded yet.")
            for file_info in files_list:
                with st.expander(f"File ID: {file_info['file_id']}"):
                    st.write(f"Original Filename: {file_info['original_filename']}")
                    # Provide a link to download the file.
                    download_url = f"{BACKEND_URL}/files/{file_info['file_id']}"
                    st.markdown(f"[Download File]({download_url})", unsafe_allow_html=True)
                    
                    # Button to delete the file.
                    if st.button(f"Delete File {file_info['file_id']}", key=f"delete_{file_info['file_id']}"):
                        try:
                            del_resp = requests.delete(f"{BACKEND_URL}/files/{file_info['file_id']}")
                            if del_resp.ok:
                                st.success("File deleted successfully.")
                            else:
                                st.error(f"Error: {del_resp.json().get('detail', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Deletion failed: {e}")
                    
                    # Option to replace/update file content, including PDF.
                    new_file = st.file_uploader(f"Replace File {file_info['file_id']}", key=f"replace_{file_info['file_id']}", type=["txt", "md", "csv", "json", "pdf"])
                    if new_file is not None:
                        if st.button(f"Update File {file_info['file_id']}", key=f"update_{file_info['file_id']}"):
                            files = {"file": new_file}
                            try:
                                rep_resp = requests.put(f"{BACKEND_URL}/files/{file_info['file_id']}", files=files)
                                if rep_resp.ok:
                                    st.success("File replaced successfully.")
                                else:
                                    st.error(f"Error: {rep_resp.json().get('detail', 'Unknown error')}")
                            except Exception as e:
                                st.error(f"Update failed: {e}")
        else:
            st.error("Failed to retrieve uploaded files.")
    except Exception as e:
        st.error(f"Error: {e}")

# ---------------------------
# Chat QA Tab
# ---------------------------
with tab2:
    st.header("Chat with Your Documents")
    question = st.text_input("Enter your question here:")
    if st.button("Ask"):
        if not question:
            st.error("Please enter a question.")
        else:
            payload = {"question": question}
            try:
                response = requests.post(f"{BACKEND_URL}/ask", json=payload)
                if response.ok:
                    answer = response.json().get("answer", "")
                    st.subheader("Answer")
                    st.write(answer)
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Request failed: {e}")
