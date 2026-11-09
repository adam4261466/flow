import os
import subprocess
import sys

def main():
    # This script now launches the GUI version of the pipeline
    project_root = r"C:\Users\kabli\OneDrive\Desktop\projects\flow"
    
    # We launch gui.py which provides the window with buttons/options
    gui_script = os.path.join(project_root, "gui.py")
    
    if not os.path.exists(gui_script):
        print(f"Error: Could not find GUI script at {gui_script}")
        input("Press Enter to exit...")
        sys.exit(1)
        
    try:
        # Launch the GUI using Python 3.13
        # We use 'start' to open it in a way that keeps the GUI responsive
        subprocess.Popen(['cmd', '/c', 'start', 'py', '-3.13', gui_script], cwd=project_root)
    except Exception as e:
        print(f"Failed to launch GUI: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
