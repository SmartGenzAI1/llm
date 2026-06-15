import os
from src.ui import build_interface

if __name__ == "__main__":
    # Build the Gradio interface
    demo = build_interface()
    
    # Check for private passcode environment variable
    password = os.environ.get("APP_PASSWORD")
    
    # Hugging Face Spaces runs on port 7860 by default.
    # We bind to 0.0.0.0 to make the service reachable within the HF container.
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "show_api": False,
        "concurrency_limit": 10
    }
    
    # Enable login window if password secret is configured
    if password:
        launch_kwargs["auth"] = ("admin", password)
        print("Secure authentication enabled via APP_PASSWORD secret.")
        
    demo.launch(**launch_kwargs)

