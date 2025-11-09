"""
Installation and Functionality Test for SCA Project
Run this to verify everything is working correctly
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)
    
    required_packages = {
        'numpy': 'NumPy',
        'tensorflow': 'TensorFlow',
        'sklearn': 'scikit-learn',
        'h5py': 'h5py',
    }
    
    optional_packages = {
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
    }
    
    all_good = True
    
    # Test required packages
    print("\nRequired Packages:")
    for package, name in required_packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name:20s} - Version {version}")
        except ImportError:
            print(f"  ✗ {name:20s} - NOT FOUND")
            all_good = False
    
    # Test optional packages
    print("\nOptional Packages:")
    for package, name in optional_packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name:20s} - Version {version}")
        except ImportError:
            print(f"  ⚠ {name:20s} - Not installed (optional)")
    
    return all_good


def test_files():
    """Test if all project files exist"""
    print("\n" + "=" * 60)
    print("Testing Project Files")
    print("=" * 60)
    
    required_files = [
        'side_channel_cnn.py',
        'sca_agent.py',
        'sca_dataset_loader.py',
        'sca_visualizer.py',
        'run_sca_demo.py',
        'integrated_sca_agent.py',
        'requirements_sca.txt',
        'README_SCA.md',
        'QUICKSTART_SCA.md',
        'SCA_PROJECT_SUMMARY.md'
    ]
    
    all_good = True
    
    for filename in required_files:
        filepath = Path(filename)
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✓ {filename:30s} - {size:,} bytes")
        else:
            print(f"  ✗ {filename:30s} - NOT FOUND")
            all_good = False
    
    return all_good


def test_module_imports():
    """Test if project modules can be imported"""
    print("\n" + "=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    modules = [
        'side_channel_cnn',
        'sca_agent',
        'sca_dataset_loader',
        'sca_visualizer',
    ]
    
    all_good = True
    
    for module_name in modules:
        try:
            mod = importlib.import_module(module_name)
            print(f"  ✓ {module_name:30s} - OK")
        except Exception as e:
            print(f"  ✗ {module_name:30s} - ERROR: {str(e)[:40]}")
            all_good = False
    
    return all_good


def test_basic_functionality():
    """Test basic functionality"""
    print("\n" + "=" * 60)
    print("Testing Basic Functionality")
    print("=" * 60)
    
    try:
        import numpy as np
        from side_channel_cnn import SCAConfig, SideChannelCNN
        
        # Test configuration
        config = SCAConfig(trace_length=1000, num_classes=256, epochs=1)
        print(f"  ✓ Configuration created")
        
        # Test model creation
        model = SideChannelCNN(config)
        model.build_model()
        print(f"  ✓ CNN model built")
        
        # Test data generation
        traces = np.random.normal(0, 1, (100, 1000))
        labels = np.random.randint(0, 256, 100)
        print(f"  ✓ Test data generated")
        
        # Test preprocessing
        processed = model.preprocess_traces(traces)
        print(f"  ✓ Data preprocessing works")
        
        print(f"\n  All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        return False


def test_agent_creation():
    """Test agent creation"""
    print("\n" + "=" * 60)
    print("Testing Agent Creation")
    print("=" * 60)
    
    try:
        from sca_agent import SCAAgent, ThreatLevel
        
        # Create agent
        agent = SCAAgent(agent_id="TEST-AGENT")
        print(f"  ✓ SCA Agent created")
        
        # Test threat levels
        levels = list(ThreatLevel)
        print(f"  ✓ Threat levels: {[l.value for l in levels]}")
        
        # Cleanup
        agent.shutdown()
        print(f"  ✓ Agent shutdown successful")
        
        print(f"\n  All agent tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Agent test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("SCA PROJECT INSTALLATION TEST")
    print("=" * 60)
    
    results = {
        'Package Imports': test_imports(),
        'Project Files': test_files(),
        'Module Imports': test_module_imports(),
        'Basic Functionality': test_basic_functionality(),
        'Agent Creation': test_agent_creation(),
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:25s} - {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Installation is complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run: python run_sca_demo.py")
        print("  2. Check: QUICKSTART_SCA.md for usage examples")
        print("  3. Read: README_SCA.md for full documentation")
    else:
        print("✗ SOME TESTS FAILED - Please check the errors above")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("  1. Install missing packages: pip install -r requirements_sca.txt")
        print("  2. Check Python version: python --version (need 3.7+)")
        print("  3. Verify file integrity")
    
    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
