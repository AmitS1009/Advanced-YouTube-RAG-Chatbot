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
                            'outtmpl': out_tmpl,
                            'quiet': True,
                        }
                        
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([video_url])
                            
                        # Find the .vtt file
                        vtt_file = None
                        for f in os.listdir(temp_dir):
                            if f.endswith('.vtt'):
                                vtt_file = os.path.join(temp_dir, f)
                                break
                                
                        if vtt_file:
                            logger.info(f"Parsing VTT file: {vtt_file}")
                            parsed_transcript = []
                            for caption in webvtt.read(vtt_file):
                                # Convert to seconds
                                start = caption.start_in_seconds
                                # Approximate duration if not explicit (end - start)
                                duration = caption.end_in_seconds - caption.start_in_seconds
                                
                                parsed_transcript.append({
                                    'text': caption.text.replace('\n', ' '),
                                    'start': start,
                                    'duration': duration
                                })
                            logger.info(f"Successfully loaded {len(parsed_transcript)} items via yt-dlp.")
                            transcript_data = parsed_transcript
                        else:
                            logger.warning("yt-dlp ran but no VTT file found.")
    
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

    def load_as_langchain_documents(self, url: str):
        """Legacy/Alternative method using LangChain's loader if needed."""
        loader = LangChainYoutubeLoader.from_youtube_url(url, add_video_info=True)
        return loader.load()
