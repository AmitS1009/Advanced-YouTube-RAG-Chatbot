from typing import Dict, List
import uuid

class SessionManager:
    """
    Manages chat sessions and threads.
    In-memory storage for now (as per "session_state.py" name intuition).
    """
    def __init__(self):
        # Structure: {thread_id: [{'role': 'user', 'content': '...'}, ...]}
        self.threads: Dict[str, List[Dict[str, str]]] = {}
        # Structure: {thread_id: "Title"}
        self.titles: Dict[str, str] = {}

    def get_thread(self, thread_id: str) -> List[Dict[str, str]]:
        return self.threads.get(thread_id, [])
        
    def get_title(self, thread_id: str) -> str:
        return self.titles.get(thread_id, "New Chat")

    def add_message(self, thread_id: str, role: str, content: str):
        if thread_id not in self.threads:
            self.threads[thread_id] = []
        self.threads[thread_id].append({"role": role, "content": content})
        
        # Auto-name thread if it's the first user message and title is generic
        if role == "user" and len(self.threads[thread_id]) == 1:
             # Basic truncation for title
             title = content[:30] + "..." if len(content) > 30 else content
             self.titles[thread_id] = title

    def create_thread(self) -> str:
        thread_id = str(uuid.uuid4())
        self.threads[thread_id] = []
        self.titles[thread_id] = "New Chat"
        return thread_id
