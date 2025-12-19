#!/usr/bin/env python3
"""
Convert lookup_table.json to expected format
从 stage_X.device.data 格式转换为 n,m,device,stage 格式
"""

import json
from pathlib import Path

def convert_lookup_table():
    """Convert lookup table format"""

    # Load old format
    old_path = Path('profiling/results/lookup_table.json')
    with open(old_path, 'r') as f:
        old_data = json.load(f)

    # Convert to new format
    new_data = {}

    for stage_key, stage_data in old_data.items():
        # Extract stage number from "stage_X"
        stage_num = int(stage_key.split('_')[1])

        for device, device_data in stage_data.items():
            if 'data' not in device_data:
                continue

            for size_key, time_ms in device_data['data'].items():
                # Parse "N_M" format
                parts = size_key.split('_')
                n = int(parts[0])
                m = int(parts[1])

                # Create new key: "n,m,device,stage"
                new_key = f"{n},{m},{device},{stage_num}"

                # Store with total_time_ms
                new_data[new_key] = {
                    "total_time_ms": time_ms
                }

    # Save converted data
    output_path = Path('profiling/results/lookup_table_converted.json')
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)

    print(f"✓ Converted {len(new_data)} entries")
    print(f"✓ Saved to: {output_path}")

    # Show sample entries
    print("\nSample entries:")
    for i, (key, value) in enumerate(list(new_data.items())[:5]):
        print(f"  {key}: {value['total_time_ms']:.2f} ms")

    # Replace original file
    import shutil
    shutil.copy(output_path, old_path)
    print(f"\n✓ Replaced original lookup_table.json")

if __name__ == '__main__':
    convert_lookup_table()
