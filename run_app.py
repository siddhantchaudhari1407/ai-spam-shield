import os
import sys
import streamlit.web.cli as stcli

def resolve_path(path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, path)

if __name__ == "__main__":
    print("Initializing AI Spam Shield Desktop Launcher...")
    try:
        # Point to the main streamlit app file
        app_path = resolve_path("app.py")
        print(f"Resolved app path: {app_path}")
        
        if not os.path.exists(app_path):
            print(f"CRITICAL ERROR: {app_path} not found!")
            sys.exit(1)

        # Configure streamlit to run headlessly
        sys.argv = [
            "streamlit",
            "run",
            app_path,
            "--global.developmentMode=false",
            "--server.headless=true",
        ]
        
        print("Launching Streamlit Server...")
        sys.exit(stcli.main())
    except Exception as e:
        print(f"\nFATAL ERROR DURING STARTUP:\n{str(e)}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...") # Keep console open on error
