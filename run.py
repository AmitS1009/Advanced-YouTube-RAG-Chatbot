import os
import sys
import subprocess

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        # If we are running inside Streamlit, this context will be present
        if get_script_run_ctx():
            from app.frontend.streamlit_app import main
            main()
        else:
            # We are running via 'python run.py', so relaunch with 'streamlit run'
            print("[INFO] Detected direct Python execution. Relaunching with Streamlit...")
            cmd = [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:]
            subprocess.run(cmd)
            
    except (ImportError, ModuleNotFoundError):
        # Fallback for environment issues, attempt to launch regardless
        cmd = [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:]
        subprocess.run(cmd)
