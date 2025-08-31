#!/usr/bin/env python3
"""Run the Hartford AI Agent with ADK web interface."""

import os
import sys
from pathlib import Path

# Set up environment
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables if .env exists
env_file = project_root / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv()

# Verify API key is set
if not os.getenv("GOOGLE_GENAI_API_KEY"):
    print("⚠️  Warning: GOOGLE_GENAI_API_KEY not set in environment")
    print("Please set it in .env file or export it:")
    print("export GOOGLE_GENAI_API_KEY='your-api-key'")
    sys.exit(1)

# Import and run the agent
from google.genai.adk import start_server
from hartford_agent import agent

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           Hartford Insurance AI Development Assistant            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Starting ADK Web Interface...                                   ║
║                                                                   ║
║  Once started, open your browser to:                           ║
║  http://localhost:8080                                          ║
║                                                                   ║
║  Available Commands in Chat:                                    ║
║  • "Analyze this requirement: [your requirement]"               ║
║  • "Analyze repository [github_url]"                           ║
║  • "Create a user story for [requirement]"                     ║
║  • "Show workflow status"                                      ║
║  • "Plan implementation for [requirement]"                     ║
║                                                                   ║
║  Press Ctrl+C to stop the server                               ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Start the ADK web server
    start_server(agent)