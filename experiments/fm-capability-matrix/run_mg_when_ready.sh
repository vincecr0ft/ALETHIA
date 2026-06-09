#!/bin/bash
cd /home/vince/ALETHIA/experiments/fm-capability-matrix
nsm() { ls output_mg_cache/sm_*.npz 2>/dev/null | wc -l; }
ntr() { ls output_mg_cache/train_*.npz 2>/dev/null | wc -l; }
nte() { ls output_mg_cache/test_*.npz 2>/dev/null | wc -l; }
until [ "$(nsm)" -ge 22 ] && [ "$(ntr)" -ge 120 ] && [ "$(nte)" -ge 40 ]; do sleep 120; done
echo "data ready: sm=$(nsm) train=$(ntr) test=$(nte)"
../../.venv/bin/python -u mg_c_recovery.py
