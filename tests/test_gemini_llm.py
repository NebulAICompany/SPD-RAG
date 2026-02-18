"""
Test the GEMINI_LLM model from constants.py.

This script tests the RobustChatGoogleGenerativeAI wrapper configured
as GEMINI_LLM to ensure it's working correctly.
"""

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_gemini_llm_sync():
    """Test synchronous invocation of GEMINI_LLM."""
    from backend.shared.constants import GEMINI_LLM
    
    print("\n" + "="*60)
    print("  Testing GEMINI_LLM (gemini-2.5-flash)")
    print("="*60)
    
    # Simple test message
    test_message = "Hello! Please respond with a brief greeting and confirm you're working."
    
    print(f"\n[SEND] Sending test message: {test_message}")
    print("[WAIT] Waiting for response...\n")
    
    try:
        response = GEMINI_LLM.invoke(test_message)
        
        # Extract content from response
        if hasattr(response, 'content'):
            content = response.content
        elif isinstance(response, str):
            content = response
        else:
            content = str(response)
        
        print("[OK] Response received successfully!")
        print(f"\n[RESPONSE] Response content:\n{'─'*60}")
        print(content)
        print(f"{'─'*60}\n")
        
        assert len(content) > 0, "Response should not be empty"
        print("[OK] Test passed: GEMINI_LLM is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_gemini_llm_async():
    """Test asynchronous invocation of GEMINI_LLM."""
    from backend.shared.constants import GEMINI_LLM
    
    print("\n" + "="*60)
    print("  Testing GEMINI_LLM (async)")
    print("="*60)
    
    test_message = "Count from 1 to 5, then say 'Done!'"
    
    print(f"\n[SEND] Sending async test message: {test_message}")
    print("[WAIT] Waiting for response...\n")
    
    try:
        response = await GEMINI_LLM.ainvoke(test_message)
        
        # Extract content from response
        if hasattr(response, 'content'):
            content = response.content
        elif isinstance(response, str):
            content = response
        else:
            content = str(response)
        
        print("[OK] Async response received successfully!")
        print(f"\n[RESPONSE] Response content:\n{'─'*60}")
        print(content)
        print(f"{'─'*60}\n")
        
        assert len(content) > 0, "Response should not be empty"
        print("[OK] Async test passed: GEMINI_LLM async is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  GEMINI_LLM Test Suite")
    print("="*60)
    
    # Test synchronous call
    print("\n[1/2] Testing synchronous invocation...")
    sync_result = test_gemini_llm_sync()
    
    # Test asynchronous call
    print("\n[2/2] Testing asynchronous invocation...")
    import asyncio
    async_result = asyncio.run(test_gemini_llm_async())
    
    # Summary
    print("\n" + "="*60)
    if sync_result and async_result:
        print("  All tests passed! [OK]")
    else:
        print("  Some tests failed! [ERROR]")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
