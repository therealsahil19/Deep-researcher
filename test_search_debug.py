"""
Debug script to test if Exa and Tavily search functions work correctly.
Run this with: python test_search_debug.py
"""
import os
import sys

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import exa_search, tavily_search
import re

# Test regex patterns
test_outputs = [
    '''Thought: I need to search for LLM releases in November 2025.
Action: search_discovery
Action Input: "November 2025 large language model released"''',
    
    '''Thought: Performing a targeted search.
Action: search_discovery
Action Input: LLM November 2025''',
    
    '''Thought: Let's try a different query.
Action: search_fact
Action Input: "Meta Llama 3 November 2025 release"''',
]

print("=" * 60)
print("TESTING REGEX PATTERNS")
print("=" * 60)

for i, output in enumerate(test_outputs):
    print(f"\n--- Test {i+1} ---")
    print(f"Output: {output[:80]}...")
    
    action_match = re.search(r"Action:\s*(search_discovery|search_fact)", output, re.IGNORECASE)
    input_match = re.search(r"Action Input:\s*(.+)", output, re.IGNORECASE)
    
    if action_match and input_match:
        action_name = action_match.group(1).lower()
        tool_input = input_match.group(1).strip()
        # Remove quotes if present
        tool_input = tool_input.strip('"\'')
        tool_input = tool_input.split("Observation:")[0].strip()
        
        print(f"[OK] Match found!")
        print(f"   Action: {action_name}")
        print(f"   Input: {tool_input}")
    else:
        print(f"[FAIL] No match!")
        print(f"   action_match: {action_match}")
        print(f"   input_match: {input_match}")

print("\n" + "=" * 60)
print("TESTING SEARCH FUNCTIONS")
print("=" * 60)

# Test with environment variables or placeholder keys
exa_key = os.environ.get("EXA_API_KEY", "")
tavily_key = os.environ.get("TAVILY_API_KEY", "")

if exa_key:
    print("\n--- Testing Exa Search ---")
    try:
        result = exa_search("AI news December 2025", exa_key)
        print(f"Result length: {len(result)} chars")
        if result.startswith("Error"):
            print(f"ERROR: {result}")
        else:
            print("Search returned results successfully!")
    except Exception as e:
        print(f"Exception: {e}")
else:
    print("\n[WARN] EXA_API_KEY not set, skipping Exa test")

if tavily_key:
    print("\n--- Testing Tavily Search ---")
    try:
        result = tavily_search("latest AI announcements", tavily_key)
        print(f"Result length: {len(result)} chars")
        if result.startswith("Error"):
            print(f"ERROR: {result}")
        else:
            print("Search returned results successfully!")
    except Exception as e:
        print(f"Exception: {e}")
else:
    print("\n[WARN] TAVILY_API_KEY not set, skipping Tavily test")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
