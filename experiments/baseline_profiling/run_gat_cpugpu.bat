@echo off
REM ========================================================================
REM GAT - CPU + GPU Testing
REM ========================================================================
setlocal EnableDelayedExpansion

echo ========================================================================
echo Baseline Profiling - GAT (CPU + GPU)
echo ========================================================================
echo.

CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

if not exist "results" mkdir results

echo [Step 1/2] Exporting GAT models...
python -c "
import sys
sys.path.insert(0, '.')
from profile_baseline import *

MODELS_DIR.mkdir(parents=True, exist_ok=True)
dummy_input = generate_input(5000, 50000)
dynamic_axes = {'x': {0: 'num_nodes'}, 'edge_index': {1: 'num_edges'}, 'output': {0: 'num_nodes'}}

model = GAT1Layer(FEATURE_DIM, OUT_DIM)
model.eval()

# Export ONNX
onnx_path = MODELS_DIR / 'gat_dynamic.onnx'
with torch.no_grad():
    torch.onnx.export(model, dummy_input, str(onnx_path), input_names=['x', 'edge_index'], output_names=['output'], dynamic_axes=dynamic_axes, opset_version=17, do_constant_folding=True)

# Convert to IR
for device in ['CPU', 'GPU']:
    ir_path = MODELS_DIR / f'gat_{device.lower()}.xml'
    convert_to_ir(onnx_path, ir_path)
    print(f'Exported: {ir_path.name}')
"

echo.
echo [Step 2/2] Measuring GAT latencies...
python -c "
import json
import sys
sys.path.insert(0, '.')
from profile_baseline import *

config = load_config()
test_cases = config['test_cases']
num_warmup = config['config']['num_warmup']
num_iterations = config['config']['num_iterations']

results = {}
for device in ['CPU', 'GPU']:
    print(f'\n[{device}]')
    ir_path = MODELS_DIR / f'gat_{device.lower()}.xml'
    if not ir_path.exists():
        print(f'  IR not found: {ir_path}')
        continue
    for case in test_cases:
        nodes, edges = case['nodes'], case['edges']
        print(f'  [{nodes}n, {edges}e]... ', end='', flush=True)
        dummy_input = generate_input(nodes, edges)
        result = measure_latency_openvino(ir_path, device, dummy_input, num_warmup, num_iterations)
        key = f'gat,{nodes},{edges},{device}'
        results[key] = result
        if result['failed']:
            print(f'FAILED')
        else:
            print(f'{result[\"mean\"]:.2f}ms')

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_DIR / 'gat_cpugpu.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to: results/gat_cpugpu.json')
"

echo.
echo GAT CPU+GPU testing complete!
pause
