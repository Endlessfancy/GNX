"""
分离测量 OpenVINO 输入/输出带宽 (Async API)

测量目标:
- CPU -> CPU (基准)
- CPU -> GPU / GPU -> CPU
- CPU -> NPU / NPU -> CPU

方法:
1. Slice 模型: input[N] -> output[1], 测输入带宽 (O(1) 计算)
2. Identity 模型: input[N] -> output[N], 测组合带宽
3. 反推输出带宽: bw_out = 1 / (k - 1/bw_in)

关键区别:
- CPU/GPU: 动态形状，编译一次，测试多个大小
- NPU: 静态形状，每个大小单独编译
"""

import numpy as np
import time
import json
from pathlib import Path
import openvino as ov
import openvino.runtime.opset14 as opset
from sklearn.linear_model import LinearRegression

# 测试数据大小 (元素数量)
TEST_SIZES = [
    1 * 1024 * 1024,      # 1M elements = 4 MB
    4 * 1024 * 1024,      # 4M elements = 16 MB
    16 * 1024 * 1024,     # 16M elements = 64 MB
    32 * 1024 * 1024,     # 32M elements = 128 MB
    64 * 1024 * 1024,     # 64M elements = 256 MB
    128 * 1024 * 1024,    # 128M elements = 512 MB
]

# NPU 可能需要更小的测试范围
NPU_TEST_SIZES = [
    512 * 1024,           # 0.5M elements = 2 MB
    1 * 1024 * 1024,      # 1M elements = 4 MB
    2 * 1024 * 1024,      # 2M elements = 8 MB
    4 * 1024 * 1024,      # 4M elements = 16 MB
    8 * 1024 * 1024,      # 8M elements = 32 MB
    16 * 1024 * 1024,     # 16M elements = 64 MB
]

WARMUP_ITERATIONS = 10
MEASURE_ITERATIONS = 50


# =============================================================================
# 动态形状模型 (CPU/GPU)
# =============================================================================

def create_dynamic_slice_model():
    """
    动态 Slice 模型: input[?] -> output[1]
    用于 CPU/GPU
    """
    # -1 表示动态维度
    param = opset.parameter([-1], dtype=np.float32, name="input")

    # Slice: 只取第一个元素 [0:1]
    start = opset.constant(np.array([0], dtype=np.int64))
    stop = opset.constant(np.array([1], dtype=np.int64))
    step = opset.constant(np.array([1], dtype=np.int64))
    axes = opset.constant(np.array([0], dtype=np.int64))
    sliced = opset.slice(param, start, stop, step, axes)

    result = opset.result(sliced, name="output")
    return ov.Model([result], [param], "dynamic_slice_model")


def create_dynamic_identity_model():
    """
    动态 Identity 模型: input[?] -> output[?]
    用于 CPU/GPU
    """
    param = opset.parameter([-1], dtype=np.float32, name="input")

    # 加极小常数防止被优化掉
    const = opset.constant(np.array([0.0], dtype=np.float32))
    add = opset.add(param, const)

    result = opset.result(add, name="output")
    return ov.Model([result], [param], "dynamic_identity_model")


# =============================================================================
# 静态形状模型 (NPU)
# =============================================================================

def create_static_slice_model(input_size):
    """
    静态 Slice 模型: input[N] -> output[1]
    用于 NPU
    """
    param = opset.parameter([input_size], dtype=np.float32, name="input")

    start = opset.constant(np.array([0], dtype=np.int64))
    stop = opset.constant(np.array([1], dtype=np.int64))
    step = opset.constant(np.array([1], dtype=np.int64))
    axes = opset.constant(np.array([0], dtype=np.int64))
    sliced = opset.slice(param, start, stop, step, axes)

    result = opset.result(sliced, name="output")
    return ov.Model([result], [param], f"static_slice_model_{input_size}")


def create_static_identity_model(size):
    """
    静态 Identity 模型: input[N] -> output[N]
    用于 NPU
    """
    param = opset.parameter([size], dtype=np.float32, name="input")

    const = opset.constant(np.array([0.0], dtype=np.float32))
    add = opset.add(param, const)

    result = opset.result(add, name="output")
    return ov.Model([result], [param], f"static_identity_model_{size}")


# =============================================================================
# 测量函数
# =============================================================================

def measure_async(infer_request, input_data, iterations=MEASURE_ITERATIONS):
    """使用 async API 测量推理时间"""
    input_tensor = ov.Tensor(input_data)

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        infer_request.set_input_tensor(input_tensor)
        infer_request.start_async()
        infer_request.wait()
        _ = infer_request.get_output_tensor().data.copy()

    # Measure
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        infer_request.set_input_tensor(input_tensor)
        infer_request.start_async()
        infer_request.wait()
        output = infer_request.get_output_tensor().data.copy()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    return {
        'mean': float(np.mean(latencies)),
        'std': float(np.std(latencies)),
        'min': float(np.min(latencies)),
        'max': float(np.max(latencies)),
        'median': float(np.median(latencies)),
    }


# =============================================================================
# CPU/GPU 测量 (动态形状)
# =============================================================================

def measure_dynamic_device(device, core, sizes=TEST_SIZES):
    """
    测量 CPU/GPU (动态形状，编译一次)
    """
    print(f"\n{'='*70}")
    print(f"测试设备: {device} (动态形状)")
    print(f"{'='*70}")

    # 编译动态模型 (只编译一次)
    print(f"\n  编译动态模型...")

    slice_model = create_dynamic_slice_model()
    slice_compiled = core.compile_model(slice_model, device)
    slice_request = slice_compiled.create_infer_request()

    identity_model = create_dynamic_identity_model()
    identity_compiled = core.compile_model(identity_model, device)
    identity_request = identity_compiled.create_infer_request()

    print(f"  编译完成")

    # 测量 Slice 模型
    print(f"\n  Slice 模型 (input[N] -> output[1]):")
    slice_results = []

    for num_elements in sizes:
        input_size_mb = num_elements * 4 / (1024 * 1024)

        try:
            input_data = np.random.randn(num_elements).astype(np.float32)
            result = measure_async(slice_request, input_data)
            result['num_elements'] = num_elements
            result['input_MB'] = input_size_mb
            result['output_MB'] = 4 / (1024 * 1024)
            slice_results.append(result)

            print(f"    N={num_elements:>10,} ({input_size_mb:>6.1f} MB): {result['mean']:>8.3f} ± {result['std']:.3f} ms")

        except Exception as e:
            print(f"    N={num_elements:>10,}: 错误 - {e}")

    # 测量 Identity 模型
    print(f"\n  Identity 模型 (input[N] -> output[N]):")
    identity_results = []

    for num_elements in sizes:
        data_size_mb = num_elements * 4 / (1024 * 1024)

        try:
            input_data = np.random.randn(num_elements).astype(np.float32)
            result = measure_async(identity_request, input_data)
            result['num_elements'] = num_elements
            result['input_MB'] = data_size_mb
            result['output_MB'] = data_size_mb
            result['total_MB'] = data_size_mb * 2
            identity_results.append(result)

            print(f"    N={num_elements:>10,} ({data_size_mb:>6.1f} MB): {result['mean']:>8.3f} ± {result['std']:.3f} ms")

        except Exception as e:
            print(f"    N={num_elements:>10,}: 错误 - {e}")

    return estimate_bandwidths(slice_results, identity_results, device)


# =============================================================================
# NPU 测量 (静态形状)
# =============================================================================

def measure_static_device(device, core, sizes=NPU_TEST_SIZES):
    """
    测量 NPU (静态形状，每个大小单独编译)
    """
    print(f"\n{'='*70}")
    print(f"测试设备: {device} (静态形状)")
    print(f"{'='*70}")

    # 测量 Slice 模型
    print(f"\n  Slice 模型 (input[N] -> output[1]):")
    slice_results = []

    for num_elements in sizes:
        input_size_mb = num_elements * 4 / (1024 * 1024)

        try:
            print(f"    编译 N={num_elements:,}...", end=" ")

            model = create_static_slice_model(num_elements)
            compiled = core.compile_model(model, device)
            request = compiled.create_infer_request()

            input_data = np.random.randn(num_elements).astype(np.float32)
            result = measure_async(request, input_data)
            result['num_elements'] = num_elements
            result['input_MB'] = input_size_mb
            result['output_MB'] = 4 / (1024 * 1024)
            slice_results.append(result)

            print(f"{result['mean']:>8.3f} ± {result['std']:.3f} ms")

        except Exception as e:
            print(f"错误 - {e}")

    # 测量 Identity 模型
    print(f"\n  Identity 模型 (input[N] -> output[N]):")
    identity_results = []

    for num_elements in sizes:
        data_size_mb = num_elements * 4 / (1024 * 1024)

        try:
            print(f"    编译 N={num_elements:,}...", end=" ")

            model = create_static_identity_model(num_elements)
            compiled = core.compile_model(model, device)
            request = compiled.create_infer_request()

            input_data = np.random.randn(num_elements).astype(np.float32)
            result = measure_async(request, input_data)
            result['num_elements'] = num_elements
            result['input_MB'] = data_size_mb
            result['output_MB'] = data_size_mb
            result['total_MB'] = data_size_mb * 2
            identity_results.append(result)

            print(f"{result['mean']:>8.3f} ± {result['std']:.3f} ms")

        except Exception as e:
            print(f"错误 - {e}")

    return estimate_bandwidths(slice_results, identity_results, device)


# =============================================================================
# 带宽估算
# =============================================================================

def estimate_bandwidths(slice_results, identity_results, device):
    """估算输入/输出带宽"""
    print(f"\n  带宽估算:")
    print(f"  {'-'*50}")

    # 估算输入带宽 (从 Slice 模型)
    input_bw = None
    if slice_results and len(slice_results) >= 3:
        X = np.array([r['input_MB'] for r in slice_results]).reshape(-1, 1)
        y = np.array([r['mean'] for r in slice_results])

        reg = LinearRegression().fit(X, y)
        coef = reg.coef_[0]

        if coef > 0:
            bandwidth = 1 / coef
        else:
            bandwidth = float('inf')

        input_bw = {
            'bandwidth_GBps': float(bandwidth),
            'overhead_ms': float(reg.intercept_),
            'coef_ms_per_MB': float(coef),
            'r_squared': float(reg.score(X, y)),
        }
        print(f"  输入带宽 (Slice): {input_bw['bandwidth_GBps']:.2f} GB/s (R²={input_bw['r_squared']:.4f})")
        print(f"    斜率: {input_bw['coef_ms_per_MB']:.4f} ms/MB, 截距: {input_bw['overhead_ms']:.3f} ms")

    # 估算组合带宽 (从 Identity 模型)
    combined_bw = None
    if identity_results and len(identity_results) >= 3:
        X = np.array([r['input_MB'] for r in identity_results]).reshape(-1, 1)
        y = np.array([r['mean'] for r in identity_results])

        reg = LinearRegression().fit(X, y)
        k = reg.coef_[0]

        combined_bw = {
            'k_combined': float(k),
            'overhead_ms': float(reg.intercept_),
            'r_squared': float(reg.score(X, y)),
        }
        print(f"  组合系数 (Identity): k = {combined_bw['k_combined']:.4f} ms/MB (R²={combined_bw['r_squared']:.4f})")
        print(f"    截距: {combined_bw['overhead_ms']:.3f} ms")

    # 计算输出带宽
    output_bw = None
    if input_bw and combined_bw:
        bw_in = input_bw['bandwidth_GBps']
        k = combined_bw['k_combined']

        inv_bw_out = k - 1/bw_in

        if inv_bw_out > 0:
            bw_out = 1 / inv_bw_out
            output_bw = {
                'bandwidth_GBps': float(bw_out),
                'calculation': f"1/bw_out = {k:.4f} - 1/{bw_in:.2f} = {inv_bw_out:.4f}",
            }
            print(f"  输出带宽 (计算): {output_bw['bandwidth_GBps']:.2f} GB/s")
            print(f"    {output_bw['calculation']}")
        else:
            print(f"  输出带宽: 无法计算 (k - 1/bw_in <= 0)")

    return {
        'slice_results': slice_results,
        'identity_results': identity_results,
        'input_bandwidth': input_bw,
        'combined_bandwidth': combined_bw,
        'output_bandwidth': output_bw,
    }


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 70)
    print("OpenVINO 分离带宽测量 (Async API)")
    print("=" * 70)
    print()
    print("方法:")
    print("  1. Slice 模型 [N] -> [1]: 测输入带宽 (O(1) 计算)")
    print("  2. Identity 模型 [N] -> [N]: 测组合带宽")
    print("  3. 反推: bw_out = 1 / (k - 1/bw_in)")
    print()
    print("模型类型:")
    print("  CPU/GPU: 动态形状 (编译一次)")
    print("  NPU: 静态形状 (每个大小单独编译)")
    print()
    print(f"CPU/GPU 测试: {len(TEST_SIZES)} 种大小 ({TEST_SIZES[0]//1024//1024}M - {TEST_SIZES[-1]//1024//1024}M 元素)")
    print(f"NPU 测试: {len(NPU_TEST_SIZES)} 种大小 ({NPU_TEST_SIZES[0]//1024//1024}M - {NPU_TEST_SIZES[-1]//1024//1024}M 元素)")
    print(f"Warmup: {WARMUP_ITERATIONS} 次, 测量: {MEASURE_ITERATIONS} 次")

    core = ov.Core()
    print(f"\n可用设备: {core.available_devices}")

    all_results = {}

    # 测试 CPU 和 GPU (动态形状)
    for device in ['CPU', 'GPU']:
        if device in core.available_devices:
            try:
                all_results[device] = measure_dynamic_device(device, core, TEST_SIZES)
            except Exception as e:
                print(f"\n{device} 测试失败: {e}")
        else:
            print(f"\n跳过 {device} (不可用)")

    # 测试 NPU (静态形状)
    if 'NPU' in core.available_devices:
        try:
            all_results['NPU'] = measure_static_device('NPU', core, NPU_TEST_SIZES)
        except Exception as e:
            print(f"\nNPU 测试失败: {e}")
    else:
        print(f"\n跳过 NPU (不可用)")

    # 保存结果
    output_dir = Path(__file__).parent / 'results' / 'bandwidth'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'bandwidth_async.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # 打印摘要
    print("\n" + "=" * 70)
    print("带宽测量摘要")
    print("=" * 70)
    print()
    print(f"{'设备':<8} {'输入带宽':<18} {'输出带宽':<18} {'组合系数 k':<15}")
    print(f"{'':8} {'(CPU->Dev)':<18} {'(Dev->CPU)':<18} {'(ms/MB)':<15}")
    print("-" * 62)

    for device in ['CPU', 'GPU', 'NPU']:
        if device in all_results:
            r = all_results[device]

            in_bw = r.get('input_bandwidth')
            out_bw = r.get('output_bandwidth')
            comb = r.get('combined_bandwidth')

            in_str = f"{in_bw['bandwidth_GBps']:.2f} GB/s" if in_bw else "N/A"
            out_str = f"{out_bw['bandwidth_GBps']:.2f} GB/s" if out_bw else "N/A"
            k_str = f"{comb['k_combined']:.4f}" if comb else "N/A"

            print(f"{device:<8} {in_str:<18} {out_str:<18} {k_str:<15}")

    print("-" * 62)
    print()
    print(f"结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
