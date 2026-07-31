#!/usr/bin/env python3
"""
AI Library Assistant Launcher
Run: python stream.py
"""

import os
import sys
import socket
import subprocess

APP_FILE = "app.py"
DEFAULT_PORT = 8501


# ---------------- UTILITIES ----------------

def port_available(port: int) -> bool:
    """Return True if port is free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def get_free_port(start=8501, end=8510) -> int:
    """Find first free port in range."""
    for p in range(start, end):
        if port_available(p):
            return p
    return DEFAULT_PORT


def check_file_exists(file):
    if not os.path.exists(file):
        print(f"❌ ERROR: '{file}' not found in current directory!")
        return False
    return True


def check_env_token():
    if not os.path.exists(".env"):
        print("❌ .env file missing!")
        print("Create .env and add:\nHF_TOKEN=your_huggingface_token")
        return False

    with open(".env") as f:
        content = f.read()

    if "HF_TOKEN=" not in content:
        print("❌ HF_TOKEN not found in .env file!")
        return False

    if "your_token" in content:
        print("⚠️ Replace placeholder token with real Hugging Face token.")
    return True


def check_packages():
    try:
        import streamlit  # noqa
        import transformers  # noqa
        import requests  # noqa
    except ImportError as e:
        print(f"❌ Missing package: {e.name}")
        print("Run: pip install -r requirements.txt")
        return False
    return True


# ---------------- MAIN ----------------

def main():
    print("\n" + "=" * 60)
    print("📚 AI LIBRARY ASSISTANT LAUNCHER")
    print("=" * 60)

    # Checks
    if not all([
        check_file_exists(APP_FILE),
        check_env_token(),
        check_packages()
    ]):
        print("\n❌ Setup check failed. Fix errors above.\n")
        sys.exit(1)

    port = get_free_port()

    print(f"📁 Directory : {os.getcwd()}")
    print(f"🚀 Launching  : {APP_FILE}")
    print(f"🌐 URL       : http://localhost:{port}")
    print("⏳ Loading Streamlit...\n")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        APP_FILE,
        f"--server.port={port}",
        "--browser.gatherUsageStats=false"
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"\n❌ Failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
