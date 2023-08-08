#!/bin/bash
python3 run_tf_training.py --label composite --frozen 0 --cv_split 3 && 
python3 run_tf_training.py --label composite --frozen 0 --cv_split 4 && 
python3 run_tf_training.py --label composite --frozen 1 --cv_split 2 && 
python3 run_tf_training.py --label composite --frozen 1 --cv_split 3 && 
python3 run_tf_training.py --label composite --frozen 1 --cv_split 4 && 
python3 run_tf_baseline_training.py --label composite --frozen 0 --cv_split 0 && 
python3 run_tf_baseline_training.py --label composite --frozen 1 --cv_split 0 && 
python3 run_tf_baseline_training.py --label composite --frozen 0 --cv_split 1 && 
python3 run_tf_baseline_training.py --label composite --frozen 1 --cv_split 1 && 
python3 run_tf_baseline_training.py --label composite --frozen 0 --cv_split 2 && 
python3 run_tf_baseline_training.py --label composite --frozen 1 --cv_split 2 && 
python3 run_tf_baseline_training.py --label composite --frozen 0 --cv_split 3 && 
python3 run_tf_baseline_training.py --label composite --frozen 1 --cv_split 3 && 
python3 run_tf_baseline_training.py --label composite --frozen 0 --cv_split 4 && 
python3 run_tf_baseline_training.py --label composite --frozen 1 --cv_split 4 