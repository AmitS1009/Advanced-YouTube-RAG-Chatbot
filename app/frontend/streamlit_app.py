import streamlit as st
import os
import time

# Initialize managers
from app.ingestion.youtube_loader import YoutubeTranscriptLoader
from app.ingestion.text_cleaner import TextCleaner
from app.ingestion.chunker import TimeAwareChunker
from app.embeddings.embedder import Embedder
from app.vectorstore.faiss_store import FaissVectorStore
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.sparse_retriever import SparseRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.hybrid_retriever import HybridRetriever
from app.reasoning.query_rewriter import QueryRewriter
from app.reasoning.context_compressor import ContextCompressor
from app.llm.answer_generator import AnswerGenerator
from app.evaluation.confidence_scorer import ConfidenceScorer
from app.frontend.session_state import SessionManager
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def process_video_ingestion(url: str):
    """
    Handles the full ingestion flow: Load -> Clean -> Chunk -> Index.
    """
    loader = YoutubeTranscriptLoader()
    video_id = loader.extract_video_id(url)
    
    # Check Session Cache
    if 'processed_videos' not in st.session_state:
        st.session_state.processed_videos = set()
        
    if video_id in st.session_state.processed_videos:
        st.success(f"Video {video_id} loaded from cache!")
        return True

    with st.status("Ingesting video...", expanded=True) as status:
        transcript = loader.load_transcript(url)
        
        if not transcript:
            st.error("Failed to fetch transcript. Video might not have captions or is restricted.")
            status.update(label="Ingestion Failed", state="error")
            return False

        st.write(f"Fetched {len(transcript)} subtitle items.")
        
        cleaner = TextCleaner()
        # Clean transcript text in place for simplicity
        for item in transcript:
            item['text'] = cleaner.clean_text(item['text'])
            
        st.write("Transcript cleaned.")
        
        chunker = TimeAwareChunker()
        chunks = chunker.create_chunks(transcript, video_id=video_id)
        st.write(f"Created {len(chunks)} semantic chunks.")
        
        # Indexing
        vector_store = st.session_state.components['vector_store']
        vector_store.create_index(chunks) 
        
        # Sparse Indexing
        from langchain_core.documents import Document
        docs = [Document(page_content=c['text'], metadata={k:v for k,v in c.items() if k!='text'}) for c in chunks]
        st.session_state.components['sparse_retriever'].create_index(docs)
        
        # Mark as processed
        st.session_state.processed_videos.add(video_id)
        
        st.write("Indexing completed.")
        status.update(label="Ingestion Complete!", state="complete")
    return True

def main():
    st.set_page_config(page_title="Advanced YouTube RAG", layout="wide")

    # CSS for better aesthetics
    st.markdown("""
    <style>
        .stChatMessage {
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .user-message {
            background-color: #f0f2f6;
        }
        .assistant-message {
            background-color: #e8f0fe;
        }
    </style>
    """, unsafe_allow_html=True)

    # Application State Initialization
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()
        st.session_state.current_thread_id = st.session_state.session_manager.create_thread()

    if 'pipeline_initialized' not in st.session_state:
        # Initialize components
        # Added _v2 suffix to force cache invalidation after method signature change
        @st.cache_resource
        def get_pipeline_components_v2():
            vector_store = FaissVectorStore()
            sparse_retriever = SparseRetriever()
            reranker = Reranker()
            dense_retriever = DenseRetriever(vector_store)
            hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever, reranker)
            
            query_rewriter = QueryRewriter()
            context_compressor = ContextCompressor()
            answer_generator = AnswerGenerator()
            
            return {
                'vector_store': vector_store,
                'sparse_retriever': sparse_retriever,
                'hybrid_retriever': hybrid_retriever,
                'query_rewriter': query_rewriter,
                'context_compressor': context_compressor,
                'answer_generator': answer_generator
            }

        st.session_state.components = get_pipeline_components_v2()
        st.session_state.pipeline_initialized = True

    # --- UI Setup ---

    st.title("🎥 Advanced YouTube RAG Chatbot")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # 1. URL Input (Moved to Top)
        video_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
        if st.button("Process Video"):
            if video_url:
                success = process_video_ingestion(video_url)
                if success:
                    st.session_state.current_video_url = video_url
                    st.success("Ready to chat!")
            else:
                st.warning("Please enter a URL.")
                
        st.divider()

        # 2. Thread Management
        st.subheader("Chat History")
        if st.button("➕ New Chat", use_container_width=True):
            new_id = st.session_state.session_manager.create_thread()
            st.session_state.current_thread_id = new_id
            st.rerun()
            
        st.write("---")
        
        # Display threads as a stack of buttons/selectable items
        threads = st.session_state.session_manager.threads
        # Sort by creation? Dict preserves insertion order in py3.7+, so reverse it for newest first
        thread_ids = list(threads.keys())[::-1]
        
        for tid in thread_ids:
            title = st.session_state.session_manager.get_title(tid)
            # Highlight current
            if tid == st.session_state.current_thread_id:
                st.info(f"📍 {title}")
            else:
                if st.button(title, key=f"btn_{tid}", use_container_width=True):
                    st.session_state.current_thread_id = tid
                    st.rerun()

    # Main Chat Area
    if 'current_video_url' in st.session_state:
        st.video(st.session_state.current_video_url)

    # Display Chat History
    history = st.session_state.session_manager.get_thread(st.session_state.current_thread_id)
    for msg in history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # User Input
    if prompt := st.chat_input("Ask about the video..."):
        # Render user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Save to history
        st.session_state.session_manager.add_message(st.session_state.current_thread_id, "user", prompt)
        
        # Generate Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # 1. Rewrite Query
            rewriter = st.session_state.components['query_rewriter']
            # Fetch history for context awareness
            thread_history = st.session_state.session_manager.get_thread(st.session_state.current_thread_id)
            # Exclude the current prompt from history provided to rewriter (it's the 'Original Query')
            # Actually, session_manager.add_message happens BEFORE this block in standard flow? 
            # Checked code: add_message(user, prompt) happens at line 189. 
            # So thread_history includes the current prompt.
            # We want history *prior* to current prompt.
            prior_history = thread_history[:-1] 
            
            rewritten_query = rewriter.rewrite(prompt, chat_history=prior_history)
            with st.expander("Search Process"):
                st.write(f"**Rewritten Query:** {rewritten_query}")

                # 2. Retrieve
                hybrid_retriever = st.session_state.components['hybrid_retriever']
                raw_docs = hybrid_retriever.search(rewritten_query)
                
                # 3. Check Confidence
                confidence = ConfidenceScorer.calculate_confidence(raw_docs)
                st.write(f"**Confidence Score:** {confidence}")
                
                if confidence == "LOW":
                    st.warning("⚠️ Low confidence in retrieved context.")
                    
                # 4. Compress
                compressor = st.session_state.components['context_compressor']
                compressed_docs = compressor.compress(raw_docs)
                st.write(f"**Context Used:** {len(compressed_docs)} chunks")

            # 5. Generate & Stream
            answer_generator = st.session_state.components['answer_generator']
            
            # Confidence Badge (Subtle)
            if confidence == "HIGH":
                st.caption("✅ High Confidence")
            elif confidence == "MEDIUM":
                st.caption("⚠️ Medium Confidence")
            else:
                 st.caption("🚨 Low Confidence")
            
             # If low confidence, maybe inject a prefix?
            prefix = ""
            if confidence == "LOW":
                prefix = "**[Low Confidence]** "
                
            generator = answer_generator.generate_answer(prompt, compressed_docs)
            
            try:
                for chunk in generator:
                    full_response += chunk
                    message_placeholder.markdown(prefix + full_response + "▌")
                
                message_placeholder.markdown(prefix + full_response)
            except Exception as e:
                st.error(f"Error: {e}")
                full_response = "Sorry, I encountered an error."

        # Save to history
        st.session_state.session_manager.add_message(st.session_state.current_thread_id, "assistant", prefix + full_response)

if __name__ == "__main__":
    main()
