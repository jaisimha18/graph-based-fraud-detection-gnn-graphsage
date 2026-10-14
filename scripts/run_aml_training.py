#!/usr/bin/env python3
# ============================================================
# run_aml_training.py
#
# Top-level runner for the AML GraphSAGE baseline model.
# Executes training and evaluation sequentially.
# ============================================================

import sys
import os
import time

# Ensure src/ is in the PYTHONPATH
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.training.train_aml import train_aml
from src.training.evaluate_aml import evaluate_aml

def main():
    print("============================================================")
    print("  AML GraphSAGE Baseline Execution")
    print("============================================================")
    
    start_time = time.time()
    
    try:
        # Step 1: Train the model
        train_aml()
        
        # Step 2: Evaluate the model
        evaluate_aml()
        
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.")
    except Exception as e:
        print(f"\n[!] Execution failed: {e}")
        import traceback
        traceback.print_exc()
        
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time) / 60:.1f} minutes.")
    print("============================================================")

if __name__ == "__main__":
    main()
