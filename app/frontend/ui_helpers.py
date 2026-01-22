import streamlit as st

def render_chat_message(role: str, content: str):
    with st.chat_message(role):
        st.markdown(content)

def render_sidebar(thread_ids: list, current_thread_id: str):
    st.sidebar.title("Chat Threads")
    selected_option = st.sidebar.radio(
        "Select Thread", 
        thread_ids, 
        index=thread_ids.index(current_thread_id) if current_thread_id in thread_ids else 0
    )
    return selected_option

def display_video_player(url: str):
    st.video(url)
