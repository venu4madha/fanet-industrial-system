import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    print(f"\n{'='*50}\n[+] {text}\n{'='*50}")

def print_success(text):
    print(f"  [OK] {text}")

def print_error(text):
    print(f"  [ERROR] {text}")

def run_cmd(cmd, check=True):
    try:
        subprocess.run(cmd, shell=True, check=check)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {cmd}")
        return False

def main():
    print_header("FANet Industrial System - Auto Fix Utility")
    
    # 1. Check Python Version
    print("\n--- Checking Python Version ---")
    if sys.version_info < (3, 7):
        print_error("Python 3.7 or higher is required.")
        sys.exit(1)
    print_success(f"Python version: {sys.version.split(' ')[0]}")

    # 2. Check Virtual Environment
    print("\n--- Checking Virtual Environment ---")
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print_error("You are not inside a virtual environment.")
        print("    Recommendation: Run '.\\venv\\Scripts\\activate' before running the server.")
    else:
        print_success("Virtual environment is active.")

    # 3. Create Required Directories
    print("\n--- Creating Required Directories ---")
    dirs = [
        "datasets",
        "models",
        "memory_banks",
        "media/uploads",
        "media/heatmaps",
        "logs"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print_success(f"Directory verified: {d}")

    # 4. Install Dependencies
    print("\n--- Installing/Verifying Dependencies ---")
    if os.path.exists("requirements.txt"):
        run_cmd(f"{sys.executable} -m pip install -r requirements.txt")
        print_success("Dependencies installed.")
    else:
        print_error("requirements.txt not found!")

    # 5. Database Migrations
    print("\n--- Running Database Migrations ---")
    run_cmd(f"{sys.executable} manage.py makemigrations")
    run_cmd(f"{sys.executable} manage.py migrate")
    print_success("Database schema is up to date.")

    # 6. Create Default Users
    print("\n--- Creating Default Users ---")
    run_cmd(f"{sys.executable} manage.py create_default_users", check=False)
    print_success("Default users verified.")

    # 7. Check for AI Models
    print("\n--- Checking AI Models ---")
    models_dir = Path("models")
    if models_dir.exists() and any(models_dir.iterdir()):
        print_success(f"Found saved AI models in {models_dir}")
    else:
        print_error("No AI models found. You will need to train them via the frontend.")

    print_header("Fix Complete!")
    print("If you had issues, they should now be resolved. You can start the server with:")
    print(f"    {sys.executable} manage.py runserver")

if __name__ == "__main__":
    main()
