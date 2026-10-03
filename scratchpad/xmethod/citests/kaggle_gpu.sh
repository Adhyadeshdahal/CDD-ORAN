# Kaggle GPU job (R-26; Colab unavailable): cmi_knn torch backend.
# (1) F2-GPU: exact equality vs CPU tigramite; (2) R-9 cost per dataset at n 1000 / 4000 / 8000 (E5 R1, eq, kappa .25);
# (3) F4 cmi_knn on the GPU backend (synthetic, arm native, primary candidates).
set -x
OUT=${JOB_OUT:-out}; mkdir -p "$OUT/gpu"
python -m pip install -q tigramite==5.2.10.1 dcor==0.7 momentchi2==0.1.8 gymnasium==1.2.3 pyyaml==6.0.3 numba==0.67.0 llvmlite==0.49.0 > "$OUT/pip_extra.log" 2>&1
python -m pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/pip_extra.log" 2>&1   # R-35: torch pinned to uv.lock (2.10.0+cu128)
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))" | tee "$OUT/gpu/torch.txt"
python scratchpad/xmethod/citests/gpu_cmi.py f2 "$OUT/gpu"
python scratchpad/xmethod/citests/gpu_cmi.py cost "$OUT/gpu" E5 R1 1000 --cap-min 60
python scratchpad/xmethod/citests/gpu_cmi.py cost "$OUT/gpu" E5 R1 4000 --cap-min 40
python scratchpad/xmethod/citests/gpu_cmi.py cost "$OUT/gpu" E5 R1 8000 --cap-min 40
C='{"arm": "native", "backend": "torch", "device": "cuda"}'
F="python scratchpad/xmethod/citests/f4.py run $OUT/f4_cmi_gpu.jsonl"
$F cmi_knn null 500 100 --jobs 2 --primary-only --cfg "$C"
$F cmi_knn planted 500 20 --jobs 2 --primary-only --cfg "$C"
$F cmi_knn null 1000 60 --jobs 2 --primary-only --cfg "$C"
$F cmi_knn planted 1000 8 --jobs 2 --primary-only --cfg "$C"
python scratchpad/xmethod/citests/f4.py summary "$OUT/f4_cmi_gpu.jsonl" --out "$OUT/F4_cmi_gpu_summary.json"
