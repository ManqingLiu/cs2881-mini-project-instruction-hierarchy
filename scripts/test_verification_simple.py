"""
Simple test to verify the code structure without loading models
Tests that the methods exist and have correct signatures
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_method_signatures():
    """Test that all new methods are properly defined"""
    print("=" * 70)
    print("Testing Method Signatures (No Model Loading)")
    print("=" * 70)

    # Import the class
    from generate_data import InstructionHierarchyDataGenerator

    # Check methods exist
    methods_to_check = [
        'verify_ground_truth_with_ai',
        'generate_ground_truth_with_rejection_sampling',
        'generate_fallback_response'
    ]

    print("\n[1/3] Checking method existence...")
    for method_name in methods_to_check:
        if hasattr(InstructionHierarchyDataGenerator, method_name):
            method = getattr(InstructionHierarchyDataGenerator, method_name)
            print(f"  ✓ {method_name} exists")

            # Check if it's callable
            if callable(method):
                print(f"    - Callable: Yes")

                # Get signature info
                import inspect
                sig = inspect.signature(method)
                params = list(sig.parameters.keys())
                print(f"    - Parameters: {', '.join(params)}")
        else:
            print(f"  ✗ {method_name} NOT FOUND")

    print("\n[2/3] Checking method documentation...")
    for method_name in methods_to_check:
        if hasattr(InstructionHierarchyDataGenerator, method_name):
            method = getattr(InstructionHierarchyDataGenerator, method_name)
            if method.__doc__:
                doc_lines = method.__doc__.strip().split('\n')
                print(f"  ✓ {method_name}: {doc_lines[0]}")
            else:
                print(f"  - {method_name}: No documentation")

    print("\n[3/3] Checking updated closed-domain generation...")
    # Check the generate_closed_domain_injections method
    method = InstructionHierarchyDataGenerator.generate_closed_domain_injections

    # Read the source to check if it uses rejection sampling
    import inspect
    source = inspect.getsource(method)

    checks = [
        ('generate_ground_truth_with_rejection_sampling', 'Uses rejection sampling'),
        ('attempt_stats', 'Tracks attempt statistics'),
        ('use_ai_verification=True', 'Uses AI verification'),
        ('STEP 4', 'Has Step 4 section')
    ]

    print("  Checking for key features:")
    for check_str, description in checks:
        if check_str in source:
            print(f"    ✓ {description}")
        else:
            print(f"    ✗ Missing: {description}")

    print("\n" + "=" * 70)
    print("Structure Validation Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run: python scripts/test_verification.py")
    print("     (This will actually load the model and test functionality)")
    print("  2. Or run the full data generation with small dataset:")
    print("     python scripts/generate_data.py")


if __name__ == "__main__":
    test_method_signatures()
