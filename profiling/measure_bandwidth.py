"""
分离测量 OpenVINO 输入/输出带宽 (直接测量)

方法:
- 使用 Identity 模型，分离计时
- 输入带宽: 从 start_async + wait 测量
- 输出带宽: 直接从 get_output_tensor().data 测量

测量目标:
- CPU -> CPU (基准)
- CPU -> GPU / GPU -> CPU
- CPU -> NPU / NPU -> CPU

模型类型:
- CPU/GPU: 动态形状 (编译一次)
- NPU: 静态形状 (每个大小单独编译)
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
MEASURE_ITERATIONS = 30


# =============================================================================
# 模型创建
# =============================================================================

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
# 测量函数 (分离输入/输出时间)
# =============================================================================

def measure_separated(infer_request, input_data, iterations=MEASURE_ITERATIONS):
    """
    分离测量输入传输+计算 和 输出传输

    Returns:
        dict:
        - input_compute_ms: start_async + wait (输入传输 + 计算)
        - output_ms: get_output_tensor().data (输出传输)
        - total_ms: 端到端总时间
    """
    input_tensor = ov.Tensor(input_data)

    # Warmup (每次都 set tensor)
    for _ in range(WARMUP_ITERATIONS):
        infer_request.set_input_tensor(input_tensor)
        infer_request.start_async()
        infer_request.wait()
        _ = infer_request.get_output_tensor().data.copy()

    # Measure
    input_compute_times = []
    output_times = []
    total_times = []

    for _ in range(iterations):
        # 每次都生成新数据，确保真实传输
        new_data = np.random.randn(len(input_data)).astype(np.float32)
        input_tensor = ov.Tensor(new_data)
        infer_request.set_input_tensor(input_tensor)

        t_start = time.perf_counter()
        infer_request.start_async()
        infer_request.wait()
        t_compute = time.perf_counter()

        # 直接测量输出传输时间
        output = infer_request.get_output_tensor().data.copy()
        t_output = time.perf_counter()

        input_compute_times.append((t_compute - t_start) * 1000)
        output_times.append((t_output - t_compute) * 1000)
        total_times.append((t_output - t_start) * 1000)

    return {
        'input_compute_ms': float(np.mean(input_compute_times)),
        'input_compute_std': float(np.std(input_compute_times)),
        'output_ms': float(np.mean(output_times)),
        'output_std': float(np.std(output_times)),
        'total_ms': float(np.mean(total_times)),
        'total_std': float(np.std(total_times)),
    }


# =============================================================================
# CPU/GPU 测量 (动态形状)
# =============================================================================

def measure_dynamic_device(device, core, sizes=TEST_SIZES):
    """
    测量 CPU/GPU (动态形状，编译一次)
    """
    print(f"\n{'='*70}")
    print(f"测试设备: {device} (动态形状, 分离测量)")
    print(f"{'='*70}")

    # 编译动态模型 (只编译一次)
    print(f"\n  编译 Identity 模型...")

    identity_model = create_dynamic_identity_model()
    identity_compiled = core.compile_model(identity_model, device)
    identity_request = identity_compiled.create_infer_request()

    print(f"  编译完成")

    # 测量
    print(f"\n  Identity 模型 (input[N] -> output[N], 分离计时):")
    print(f"  {'N':>12}  {'Size':>8}  {'In+Compute':>12}  {'Output':>10}  {'Total':>10}")
    print(f"  {'-'*60}")

    results = []

    for num_elements in sizes:
        data_size_mb = num_elements * 4 / (1024 * 1024)

        try:
            input_data = np.random.randn(num_elements).astype(np.float32)
            result = measure_separated(identity_request, input_data)
            result['num_elements'] = num_elements
            result['data_MB'] = data_size_mb
            results.append(result)

            print(f"  {num_elements:>12,}  {data_size_mb:>6.1f}MB  "
                  f"{result['input_compute_ms']:>8.3f}ms   "
                  f"{result['output_ms']:>7.3f}ms   "
                  f"{result['total_ms']:>7.3f}ms")

        except Exception as e:
            print(f"  {num_elements:>12,}: 错误 - {e}")

    return estimate_bandwidths_v2(results, device)


# =============================================================================
# NPU 测量 (静态形状)
# =============================================================================

def measure_static_device(device, core, sizes=NPU_TEST_SIZES):
    """
    测量 NPU (静态形状，每个大小单独编译)
    """
    print(f"\n{'='*70}")
    print(f"测试设备: {device} (静态形状, 分离测量)")
    print(f"{'='*70}")

    print(f"\n  Identity 模型 (input[N] -> output[N], 分离计时):")
    print(f"  {'N':>12}  {'Size':>8}  {'In+Compute':>12}  {'Output':>10}  {'Total':>10}")
    print(f"  {'-'*60}")

    results = []

    for num_elements in sizes:
        data_size_mb = num_elements * 4 / (1024 * 1024)

        try:
            print(f"  {num_elements:>12,}  编译中...", end="\r")

            model = create_static_identity_model(num_elements)
            compiled = core.compile_model(model, device)
            request = compiled.create_infer_request()

            input_data = np.random.randn(num_elements).astype(np.float32)
            result = measure_separated(request, input_data)
            result['num_elements'] = num_elements
            result['data_MB'] = data_size_mb
            results.append(result)

            print(f"  {num_elements:>12,}  {data_size_mb:>6.1f}MB  "
                  f"{result['input_compute_ms']:>8.3f}ms   "
                  f"{result['output_ms']:>7.3f}ms   "
                  f"{result['total_ms']:>7.3f}ms")

        except Exception as e:
            print(f"  {num_elements:>12,}: 错误 - {e}")

    return estimate_bandwidths_v2(results, device)


# =============================================================================
# 带宽估算 (分离测量)
# =============================================================================

def estimate_bandwidths_v2(results, device):
    """
    估算输入/输出带宽 (分离测量)

    输入带宽: 从 input_compute_ms 估算 (Identity 计算极小，可忽略)
    输出带宽: 直接从 output_ms 测量
    """
    print(f"\n  带宽估算:")
    print(f"  {'-'*50}")

    if len(results) < 3:
        print(f"  数据点不足，需要至少 3 个")
        return {'results': results}

    # 输入带宽 (从 input_compute_ms)
    X = np.array([r['data_MB'] for r in results]).reshape(-1, 1)
    y_input = np.array([r['input_compute_ms'] for r in results])

    reg_input = LinearRegression().fit(X, y_input)
    coef_input = reg_input.coef_[0]

    if coef_input > 0:
        input_bw = 1 / coef_input  # GB/s (因为 ms/MB = 1/(GB/s))
    else:
        input_bw = float('inf')

    input_bandwidth = {
        'bandwidth_GBps': float(input_bw),
        'overhead_ms': float(reg_input.intercept_),
        'coef_ms_per_MB': float(coef_input),
        'r_squared': float(reg_input.score(X, y_input)),
    }

    print(f"  输入带宽 (In+Compute): {input_bandwidth['bandwidth_GBps']:.2f} GB/s "
          f"(R²={input_bandwidth['r_squared']:.4f})")
    print(f"    斜率: {input_bandwidth['coef_ms_per_MB']:.4f} ms/MB, "
          f"截距: {input_bandwidth['overhead_ms']:.3f} ms")

    # 输出带宽 (直接从 output_ms)
    y_output = np.array([r['output_ms'] for r in results])

    reg_output = LinearRegression().fit(X, y_output)
    coef_output = reg_output.coef_[0]

    if coef_output > 0:
        output_bw = 1 / coef_output
    else:
        output_bw = float('inf')

    output_bandwidth = {
        'bandwidth_GBps': float(output_bw),
        'overhead_ms': float(reg_output.intercept_),
        'coef_ms_per_MB': float(coef_output),
        'r_squared': float(reg_output.score(X, y_output)),
    }

    print(f"  输出带宽 (Direct):    {output_bandwidth['bandwidth_GBps']:.2f} GB/s "
          f"(R²={output_bandwidth['r_squared']:.4f})")
    print(f"    斜率: {output_bandwidth['coef_ms_per_MB']:.4f} ms/MB, "
          f"截距: {output_bandwidth['overhead_ms']:.3f} ms")

    # 组合带宽 (从 total_ms)
    y_total = np.array([r['total_ms'] for r in results])

    reg_total = LinearRegression().fit(X, y_total)
    coef_total = reg_total.coef_[0]

    combined_bandwidth = {
        'coef_ms_per_MB': float(coef_total),
        'overhead_ms': float(reg_total.intercept_),
        'r_squared': float(reg_total.score(X, y_total)),
    }

    print(f"  组合系数 (Total):     k = {combined_bandwidth['coef_ms_per_MB']:.4f} ms/MB "
          f"(R²={combined_bandwidth['r_squared']:.4f})")

    # 验证: input + output ≈ total
    expected_total = coef_input + coef_output
    print(f"\n  验证: In+Out = {coef_input:.4f} + {coef_output:.4f} = {expected_total:.4f} ms/MB")
    print(f"        Total  = {coef_total:.4f} ms/MB")
    diff_pct = abs(expected_total - coef_total) / coef_total * 100 if coef_total > 0 else 0
    print(f"        差异   = {diff_pct:.1f}%")

    return {
        'results': results,
        'input_bandwidth': input_bandwidth,
        'output_bandwidth': output_bandwidth,
        'combined_bandwidth': combined_bandwidth,
    }


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 70)
    print("OpenVINO 分离带宽测量 (直接测量输入/输出)")
    print("=" * 70)
    print()
    print("方法:")
    print("  1. Identity 模型 [N] -> [N]")
    print("  2. 分离计时:")
    print("     - input_compute = start_async() + wait()")
    print("     - output        = get_output_tensor().data")
    print("  3. 直接测量两个带宽，不需要间接计算")
    print()
    print("与 V1 对比:")
    print("  V1: Slice 测输入 → Identity 测组合 → 反推输出")
    print("  Identity 分离计时 → 直接测输入/输出")
    print()
    print(f"CPU/GPU 测试: {len(TEST_SIZES)} 种大小 "
          f"({TEST_SIZES[0]//1024//1024}M - {TEST_SIZES[-1]//1024//1024}M 元素)")
    print(f"NPU 测试: {len(NPU_TEST_SIZES)} 种大小 "
          f"({NPU_TEST_SIZES[0]//1024//1024}M - {NPU_TEST_SIZES[-1]//1024//1024}M 元素)")
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

    output_file = output_dir / 'bandwidth.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # 打印摘要
    print("\n" + "=" * 70)
    print("带宽测量摘要")
    print("=" * 70)
    print()
    print(f"{'设备':<8} {'输入带宽':<18} {'输出带宽':<18} {'R² (in/out)':<15}")
    print(f"{'':8} {'(CPU->Dev)':<18} {'(Dev->CPU)':<18}")
    print("-" * 62)

    for device in ['CPU', 'GPU', 'NPU']:
        if device in all_results:
            r = all_results[device]

            in_bw = r.get('input_bandwidth')
            out_bw = r.get('output_bandwidth')

            in_str = f"{in_bw['bandwidth_GBps']:.2f} GB/s" if in_bw else "N/A"
            out_str = f"{out_bw['bandwidth_GBps']:.2f} GB/s" if out_bw else "N/A"

            r2_in = in_bw['r_squared'] if in_bw else 0
            r2_out = out_bw['r_squared'] if out_bw else 0
            r2_str = f"{r2_in:.2f} / {r2_out:.2f}"

            print(f"{device:<8} {in_str:<18} {out_str:<18} {r2_str:<15}")

    print("-" * 62)
    print()
    print(f"结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
