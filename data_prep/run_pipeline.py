#!/usr/bin/env python3
"""
Main pipeline runner for cross-city restaurant recommendation data preparation
"""

import sys
import time
from pathlib import Path
import subprocess
import argparse

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEP_SCRIPTS = [
    "step1_filter_and_split.py",
    "step2_bert_encoding.py", 
    "step3_geo_features.py",
    "step4_graph_construction.py",
    "step5_aspect_assignment.py"
]

def run_step(script_name, step_num, total_steps, resume_from=None):
    """Run a single step of the pipeline"""
    print(f"\n{'='*60}")
    print(f"Step {step_num}/{total_steps}: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script as a subprocess
        result = subprocess.run([
            sys.executable, script_name
        ], cwd=Path(__file__).parent, capture_output=True, text=True)
        
        if result.returncode == 0:
            elapsed_time = time.time() - start_time
            print(f"✓ Step {step_num} completed in {elapsed_time:.1f} seconds")
            print("Output:")
            print(result.stdout)
            return True
        else:
            print(f"✗ Step {step_num} failed with error code {result.returncode}")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Step {step_num} failed with exception: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'transformers', 
        'torch', 'tqdm', 'pyarrow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    print("✓ All dependencies are available")
    return True

def main():
    """Main pipeline runner"""
    parser = argparse.ArgumentParser(description='Cross-city restaurant recommendation data pipeline')
    parser.add_argument('--start-step', type=int, default=1, help='Step to start from (1-5)')
    parser.add_argument('--end-step', type=int, default=5, help='Step to end at (1-5)')
    parser.add_argument('--skip-deps-check', action='store_true', help='Skip dependency check')
    parser.add_argument('--skip-step1', action='store_true', help='Skip step 1')
    parser.add_argument('--skip-step2', action='store_true', help='Skip step 2')
    parser.add_argument('--skip-step3', action='store_true', help='Skip step 3')
    parser.add_argument('--skip-step4', action='store_true', help='Skip step 4')
    parser.add_argument('--skip-step5', action='store_true', help='Skip step 5')
    
    args = parser.parse_args()
    
    print("Cross-City Restaurant Recommendation Data Preparation Pipeline")
    print("=" * 70)
    
    # Validate step range
    if args.start_step < 1 or args.start_step > 5 or args.end_step < 1 or args.end_step > 5:
        print("Error: Step numbers must be between 1 and 5")
        return 1
    
    if args.start_step > args.end_step:
        print("Error: Start step cannot be greater than end step")
        return 1
    
    # Check dependencies
    if not args.skip_deps_check and not check_dependencies():
        return 1
    
    # Create outputs directory
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Run pipeline steps
    total_steps = args.end_step - args.start_step + 1
    completed_steps = 0
    
    print(f"\nRunning pipeline from step {args.start_step} to {args.end_step}")
    
    skip_flags = {
        1: args.skip_step1,
        2: args.skip_step2,
        3: args.skip_step3,
        4: args.skip_step4,
        5: args.skip_step5,
    }

    for step_num in range(args.start_step, args.end_step + 1):
        if skip_flags.get(step_num, False):
            print(f"\n{'='*60}")
            print(f"Step {step_num}/{total_steps}: skipped")
            print(f"{'='*60}")
            completed_steps += 1
            continue
        script_name = STEP_SCRIPTS[step_num - 1]
        
        if run_step(script_name, step_num, total_steps):
            completed_steps += 1
        else:
            print(f"\nPipeline failed at step {step_num}")
            print(f"Completed {completed_steps}/{total_steps} steps")
            return 1
    
    print(f"\n{'='*70}")
    print(f"Pipeline completed successfully!")
    print(f"All {completed_steps} steps completed")
    print(f"Output files saved to: {output_dir}")
    print(f"{'='*70}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
