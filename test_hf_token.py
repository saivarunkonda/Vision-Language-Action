"""
Test script to verify HuggingFace token is working.
"""

from huggingface_hub import whoami
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get token
token = os.environ.get('HF_TOKEN')

if not token:
    print("❌ HF_TOKEN not found in environment variables")
    print("Please set it in .env file or as environment variable")
else:
    print(f"✓ HF_TOKEN found (length: {len(token)})")
    
    try:
        # Test authentication
        user_info = whoami(token=token)
        print(f"✓ Authentication successful!")
        print(f"  Username: {user_info.get('name', 'N/A')}")
        print(f"  Type: {user_info.get('type', 'N/A')}")
        print(f"  Can pay: {user_info.get('canPay', False)}")
        print("\n✓ Token is working correctly!")
        print("✓ You should now have faster download speeds from HuggingFace")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("Please check your token is correct")
