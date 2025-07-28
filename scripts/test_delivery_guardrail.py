#!/usr/bin/env python3
"""
Test output guardrail integration in delivery agent
"""

import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent))
from shared.guardrails import OutputGuardrail


async def test_delivery_guardrail():
    """Test that output guardrail properly validates code."""
    
    print("🧪 Testing Delivery Agent Guardrail Integration...\n")
    
    # Test code samples
    test_cases = [
        {
            "name": "Clean code",
            "code": '''def calculate_area(radius: float) -> float:
    """Calculate circle area."""
    import math
    return math.pi * radius ** 2''',
            "expect_sanitized": False
        },
        {
            "name": "Code with hardcoded secret",
            "code": '''def connect_to_api():
    api_key = "sk-1234567890abcdef"
    return requests.get(f"https://api.example.com?key={api_key}")''',
            "expect_sanitized": True
        },
        {
            "name": "Code with dangerous operation",
            "code": '''import subprocess
subprocess.call("rm -rf /tmp/*", shell=True)''',
            "expect_sanitized": True
        }
    ]
    
    # Test the guardrail
    guardrail = OutputGuardrail()
    
    for test in test_cases:
        print(f"📋 Testing: {test['name']}")
        print(f"   Original code:")
        print("   " + "-"*40)
        for line in test['code'].split('\n'):
            print(f"   {line}")
        print("   " + "-"*40)
        
        # Apply guardrail
        context = {"final_code": test['code']}
        result = await guardrail.validate(context)
        
        final_code = result.get("final_code", test['code'])
        was_sanitized = final_code != test['code']
        quality_score = result.get("quality_score", 0)
        
        print(f"\n   Results:")
        print(f"   - Quality Score: {quality_score:.2f}")
        print(f"   - Was Sanitized: {was_sanitized}")
        print(f"   - Expected Sanitized: {test['expect_sanitized']}")
        
        if was_sanitized:
            print(f"\n   Sanitized code:")
            print("   " + "-"*40)
            for line in final_code.split('\n'):
                print(f"   {line}")
            print("   " + "-"*40)
        
        # Check if result matches expectation
        status = "✅" if was_sanitized == test['expect_sanitized'] else "❌"
        print(f"\n   {status} Test {'PASSED' if was_sanitized == test['expect_sanitized'] else 'FAILED'}")
        print("\n" + "="*60 + "\n")
    
    print("✅ Delivery guardrail integration test complete!")
    print("\nTo test with the delivery agent:")
    print("1. Run: adk run agents/delivery_agent")
    print("2. Give it code with security issues")
    print("3. Verify it reports sanitization")


if __name__ == "__main__":
    asyncio.run(test_delivery_guardrail())