import os
from typing import List, Dict
from langsmith import traceable
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_community.document_loaders import YoutubeLoader as LangChainYoutubeLoader
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

import json
from app.config.settings import settings

class YoutubeTranscriptLoader:
    def __init__(self):
        self.transcript_dir = os.path.join(settings.DATA_DIR, "transcripts")
        os.makedirs(self.transcript_dir, exist_ok=True)

    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extracts video ID from a YouTube URL."""
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be" in url:
            return url.split("/")[-1]
        return url

    @traceable(name="load_transcript", run_type="tool")
    def load_transcript(self, video_url: str) -> List[Dict]:
        """
        Tries to load transcript via YouTubeTranscriptApi.
        Falls back to other methods if needed (though we'll start with just API).
        Returns a list of dictionaries with 'text', 'start', 'duration'.
        """
        video_id = self.extract_video_id(video_url)
        
        # Check Cache
        cache_path = os.path.join(self.transcript_dir, f"{video_id}.json")
        if os.path.exists(cache_path):
            logger.info(f"Loading transcript from cache: {cache_path}")
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")

        logger.info(f"Fetching transcript for video: {video_id}")
        transcript_data = None
        
        try:
            # 1. List all available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_transcript(['en', 'en-US'])
            logger.info(f"Found transcript: {transcript.language}")
            transcript_data = transcript.fetch()
            
        except Exception as e:
            logger.warning(f"Primary fetch failed: {e}. Trying fallback methods...")
            
            # Fallback 1: Loose search via list_transcripts (already tried implicitly above but let's be thorough)
            try:
                available = YouTubeTranscriptApi.list_transcripts(video_id)
                for code in ['en', 'en-US', 'en-GB', 'auto']: 
                    try:
                        transcript_data = available.find_transcript([code]).fetch()
                        break
                    except:
                        continue
            except:
                pass

            if not transcript_data:
                # Fallback 2: yt-dlp (Robust for auto-subs)
                logger.info("Attempting yt-dlp fallback...")
                try:
                    import yt_dlp
                    import webvtt
                    import tempfile
                    import uuid
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        out_tmpl = os.path.join(temp_dir, f"{str(uuid.uuid4())}.%(ext)s")
                        ydl_opts = {
                            'skip_download': True,
                            'writesubtitles': True,
                            'writeautomaticsub': True, # Important for auto-generated
                            'subtitleslangs': ['en'],
                            'subtitlesformat': 'json3', # Clean linear format
                            'outtmpl': out_tmpl,
                            'quiet': True,
                        }
                        
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([video_url])
                            
                        # Find the .json3 file
                        json_file = None
                        for f in os.listdir(temp_dir):
                            if f.endswith('.json3'):
                                json_file = os.path.join(temp_dir, f)
                                break
                                
                        if json_file:
                            logger.info(f"Parsing JSON3 file: {json_file}")
                            parsed_transcript = []
                            with open(json_file, 'r', encoding='utf-8') as jf:
                                data = json.load(jf)
                                
                            for event in data.get('events', []):
                                # Skip non-speech events if any
                                if 'segs' not in event:
                                    continue
                                    
                                # Concatenate segments
                                segs = event['segs']
                                # DEBUG: Print segs for first few items
                                if len(parsed_transcript) < 5:
                                    print(f"DEBUG SEGS: {segs}")

                                text = "".join([s.get('utf8', '') for s in segs]).strip()
                                
                                # Skip empty
                                if not text or text == '\n':
                                    continue

                                start = event.get('tStartMs', 0) / 1000.0
                                duration = event.get('dDurationMs', 0) / 1000.0
                                
                                parsed_transcript.append({
                                    'text': text,
                                    'start': start,
                                    'duration': duration
                                })
                                
                            logger.info(f"Successfully loaded {len(parsed_transcript)} items via yt-dlp.")
                            transcript_data = self._deduplicate_rolling_captions(parsed_transcript)
                            logger.info(f"Deduplicated to {len(transcript_data)} items.")
                        else:
                            logger.warning("yt-dlp ran but no JSON3 file found.")
    
                except ImportError:
                    logger.error("yt-dlp or webvtt-py not installed.")
                except Exception as e_dlp:
                    logger.error(f"yt-dlp fallback failed: {e_dlp}")

        if transcript_data:
             # Save to Cache
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(transcript_data, f, indent=2)
                logger.info(f"Saved transcript to cache: {cache_path}")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")
            
            return transcript_data

        logger.error("All transcript fetch methods failed.")
        return None

    def _deduplicate_rolling_captions(self, transcript: List[Dict]) -> List[Dict]:
        """
        Fixes rolling caption duplication (A B, B C, C D -> A B C D).
        Also handles exact repeats (A, A, A -> A).
        """
        if not transcript:
            return []
            
        cleaned = []
        last_text = ""
        
        for item in transcript:
            text = item['text'].strip()
            if not text:
                continue
            
            if len(cleaned) < 5:
                print(f"DEBUG DEDUPE: Processing: '{text}' vs Last: '{last_text}'")

            # 1. Exact Duplicate
            if text == last_text:
                if len(cleaned) < 5: print("  -> Skipped (Exact)")
                continue
                
            # 2. Substring (subset)
            if text in last_text:
                if len(cleaned) < 5: print("  -> Skipped (In Last)")
                continue
            if last_text in text and last_text:
                if len(cleaned) < 5: print("  -> Merging (Last In Text)")
                cleaned.pop()
                cleaned.append(item)
                last_text = text
                continue
                
            # 3. Rolling Overlap (Suffix-Prefix)
            # Check for overlap of 10 chars, then 9... down to 5.
            # "Hello world", "world is" -> "Hello world is"
            # This is expensive for every pair.
            # Simplified: Check word overlap.
            
            # Simple heuristic: Just append. 
            # If the json3 gives "A B", "B C", "C D" -> "A B B C C D".
            # The duplication is mostly "A B", "A B", "A B" (Exact) in my symptoms.
            # Or "A B", "B C".
            
            # Let's try strictly appending IF NOT exact/substring first.
            # The garbage output showed "in general in general in general".
            # This implies "in general", "in general", "in general".
            # Case 1 (Exact) handles this.
            
            cleaned.append(item)
            last_text = text
            
        return cleaned

    def load_as_langchain_documents(self, url: str):
        """Legacy/Alternative method using LangChain's loader if needed."""
        loader = LangChainYoutubeLoader.from_youtube_url(url, add_video_info=True)
        return loader.load()
