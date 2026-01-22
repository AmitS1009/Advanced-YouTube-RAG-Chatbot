import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.document_loaders import YoutubeLoader as LangChainYoutubeLoader
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class YoutubeTranscriptLoader:
    def __init__(self):
        pass

    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extracts video ID from a YouTube URL."""
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be" in url:
            return url.split("/")[-1]
        return url

    def load_transcript(self, url: str):
        """
        Tries to load transcript via YouTubeTranscriptApi.
        Falls back to other methods if needed (though we'll start with just API).
        Returns a list of dictionaries with 'text', 'start', 'duration'.
        """
        video_id = self.extract_video_id(url)
        logger.info(f"Fetching transcript for video: {video_id}")
        
        try:
            # 1. List all available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_transcript(['en', 'en-US'])
            logger.info(f"Found transcript: {transcript.language}")
            return transcript.fetch()
            
        except Exception as e:
            logger.warning(f"Primary fetch failed: {e}. Trying fallback methods...")
            
            # Fallback 1: Loose search via list_transcripts (already tried implicitly above but let's be thorough)
            try:
                available = YouTubeTranscriptApi.list_transcripts(video_id)
                for code in ['en', 'en-US', 'en-GB', 'auto']: 
                    try:
                        return available.find_transcript([code]).fetch()
                    except:
                        continue
            except:
                pass

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
                        ydl.download([url])
                        
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
                        return parsed_transcript
                    else:
                        logger.warning("yt-dlp ran but no VTT file found.")

            except ImportError:
                logger.error("yt-dlp or webvtt-py not installed.")
            except Exception as e_dlp:
                logger.error(f"yt-dlp fallback failed: {e_dlp}")

            logger.error("All transcript fetch methods failed.")
            return None

    def load_as_langchain_documents(self, url: str):
        """Legacy/Alternative method using LangChain's loader if needed."""
        loader = LangChainYoutubeLoader.from_youtube_url(url, add_video_info=True)
        return loader.load()
