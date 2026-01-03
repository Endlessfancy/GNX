"""
Model Exporter for Latency Testing

Exports 7-Stage GraphSAGE models for PEP configurations.
Reuses executer's model definitions with ONNX → OpenVINO IR conversion.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add executer to path
_parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_parent_dir / 'executer'))

import numpy as np

try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("WARNING: OpenVINO not available for IR conversion")

from model_export_utils import SimpleModelExporter
from pep_config import ALL_PEPS, PEP1, PEP2


class GNNModelExporter:
    """
    Export fused 7-Stage GraphSAGE models for latency testing.

    Features:
    - Exports ONNX models using executer's SimpleModelExporter
    - Converts to OpenVINO IR format
    - Supports dynamic (CPU/GPU) and static (NPU) shapes
    - Exports all models needed for a PEP configuration
    """

    def __init__(self,
                 output_dir: Path = None,
                 in_dim: int = 500,
                 out_dim: int = 256):
        """
        Initialize exporter.

        Args:
            output_dir: Directory to save models
            in_dim: Input feature dimension (Flickr = 500)
            out_dim: Output dimension (256)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / 'models'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Initialize SimpleModelExporter from executer
        self.exporter = SimpleModelExporter()
        self.exporter.initialize_stages(in_dim=in_dim, out_dim=out_dim)

        # Track exported models
        self.exported_models: Dict[str, str] = {}

    def _get_model_key(self, stages: List[int], device: str) -> str:
        """Generate unique key for a model configuration."""
        stages_str = '_'.join(map(str, stages))
        return f"stages_{stages_str}_{device}"

    def _get_onnx_path(self, stages: List[int], device: str) -> Path:
        """Get ONNX model path."""
        key = self._get_model_key(stages, device)
        return self.output_dir / f"{key}.onnx"

    def _get_ir_path(self, stages: List[int], device: str) -> Path:
        """Get OpenVINO IR model path (.xml)."""
        key = self._get_model_key(stages, device)
        suffix = "_static" if device == "NPU" else "_dynamic"
        return self.output_dir / f"{key}{suffix}.xml"

    def export_onnx(self,
                    stages: List[int],
                    device: str,
                    max_nodes: int,
                    max_edges: int,
                    force: bool = False) -> Path:
        """
        Export ONNX model for a stage configuration.

        Args:
            stages: List of stage indices (e.g., [1, 2, 3, 4, 5])
            device: Target device (CPU, GPU, NPU)
            max_nodes: Maximum nodes for static shape
            max_edges: Maximum edges for static shape
            force: Force re-export even if exists

        Returns:
            Path to ONNX model
        """
        onnx_path = self._get_onnx_path(stages, device)

        # Check if already exists
        if onnx_path.exists() and not force:
            size = os.path.getsize(onnx_path)
            if size > 200:  # Not a placeholder
                print(f"  ONNX exists: {onnx_path.name} ({size / 1024:.1f} KB)")
                return onnx_path

        # Export using SimpleModelExporter
        dynamic = (device != 'NPU')
        self.exporter.export_combined_model(
            device=device,
            stages=stages,
            output_path=str(onnx_path),
            num_nodes=max_nodes,
            num_edges=max_edges,
            num_features=self.in_dim,
            dynamic=dynamic
        )

        return onnx_path

    def convert_to_ir(self,
                      onnx_path: Path,
                      stages: List[int],
                      device: str,
                      force: bool = False) -> Path:
        """
        Convert ONNX model to OpenVINO IR.

        Args:
            onnx_path: Path to ONNX model
            stages: Stage indices (for naming)
            device: Target device
            force: Force re-conversion

        Returns:
            Path to IR model (.xml)
        """
        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO not available for IR conversion")

        ir_path = self._get_ir_path(stages, device)
        bin_path = ir_path.with_suffix('.bin')

        # Check if already exists
        if ir_path.exists() and bin_path.exists() and not force:
            # Check modification times
            onnx_mtime = os.path.getmtime(onnx_path)
            ir_mtime = os.path.getmtime(ir_path)
            if ir_mtime >= onnx_mtime:
                print(f"  IR exists: {ir_path.name}")
                return ir_path

        print(f"  Converting to IR: {ir_path.name}")

        # Convert ONNX to OpenVINO IR
        ov_model = ov.convert_model(str(onnx_path))

        # Save IR model (disable FP16 compression for GPU compatibility)
        ov.save_model(ov_model, str(ir_path), compress_to_fp16=False)

        print(f"  IR saved: {ir_path.name}")
        return ir_path

    def export_for_stages(self,
                          stages: List[int],
                          device: str,
                          max_nodes: int,
                          max_edges: int,
                          force: bool = False) -> Path:
        """
        Export model for a stage configuration (ONNX + IR).

        Args:
            stages: List of stage indices
            device: Target device
            max_nodes: Maximum nodes
            max_edges: Maximum edges
            force: Force re-export

        Returns:
            Path to IR model
        """
        print(f"\nExporting stages {stages} for {device}:")

        # 1. Export ONNX
        onnx_path = self.export_onnx(stages, device, max_nodes, max_edges, force)

        # 2. Convert to IR
        if OPENVINO_AVAILABLE:
            ir_path = self.convert_to_ir(onnx_path, stages, device, force)
            key = self._get_model_key(stages, device)
            self.exported_models[key] = str(ir_path)
            return ir_path
        else:
            return onnx_path

    def export_for_pep(self,
                       pep: List,
                       max_nodes: int,
                       max_edges: int,
                       force: bool = False) -> Dict[str, str]:
        """
        Export all models needed for a PEP configuration.

        Args:
            pep: PEP configuration list
            max_nodes: Maximum nodes per subgraph
            max_edges: Maximum edges per subgraph
            force: Force re-export

        Returns:
            {model_key: ir_path} mapping
        """
        print(f"\n{'='*60}")
        print("Exporting models for PEP configuration")
        print(f"{'='*60}")

        for block_id, block in enumerate(pep):
            devices = block[0]
            stages = block[1]

            print(f"\nBlock {block_id}: stages {stages}")

            for device in devices:
                self.export_for_stages(stages, device, max_nodes, max_edges, force)

        print(f"\n{'='*60}")
        print(f"Exported {len(self.exported_models)} models")
        print(f"{'='*60}")

        return self.exported_models

    def export_all_peps(self,
                        max_nodes: int,
                        max_edges: int,
                        force: bool = False) -> Dict[str, Dict[str, str]]:
        """
        Export models for all predefined PEP configurations.

        Args:
            max_nodes: Maximum nodes
            max_edges: Maximum edges
            force: Force re-export

        Returns:
            {pep_name: {model_key: ir_path}}
        """
        all_exports = {}

        for pep_name, pep in ALL_PEPS.items():
            print(f"\n\n{'#'*60}")
            print(f"# {pep_name.upper()}")
            print(f"{'#'*60}")

            self.exported_models.clear()
            self.export_for_pep(pep, max_nodes, max_edges, force)
            all_exports[pep_name] = self.exported_models.copy()

        return all_exports

    def get_model_path(self, stages: List[int], device: str) -> Optional[Path]:
        """
        Get the IR model path for a stage/device configuration.

        Returns None if model hasn't been exported.
        """
        ir_path = self._get_ir_path(stages, device)
        if ir_path.exists():
            return ir_path
        return None

    def list_exported_models(self):
        """Print list of all exported models."""
        print("\nExported Models:")
        print("-" * 60)
        for key, path in self.exported_models.items():
            size = os.path.getsize(path)
            print(f"  {key}: {path} ({size / 1024:.1f} KB)")


def main():
    """Main entry point for model export."""
    parser = argparse.ArgumentParser(description="Export GNN models for latency testing")
    parser.add_argument("--pep", type=str, default="pep1",
                        choices=list(ALL_PEPS.keys()) + ['all'],
                        help="PEP configuration to export")
    parser.add_argument("--max-nodes", type=int, default=15000,
                        help="Maximum nodes per subgraph (default: 15000)")
    parser.add_argument("--max-edges", type=int, default=150000,
                        help="Maximum edges per subgraph (default: 150000)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-export even if models exist")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for models")

    args = parser.parse_args()

    print("=" * 60)
    print("GNN Model Exporter for Latency Testing")
    print("=" * 60)
    print(f"PEP: {args.pep}")
    print(f"Max nodes: {args.max_nodes}")
    print(f"Max edges: {args.max_edges}")

    # Create exporter
    output_dir = Path(args.output_dir) if args.output_dir else None
    exporter = GNNModelExporter(output_dir=output_dir)

    # Export models
    if args.pep == 'all':
        exporter.export_all_peps(args.max_nodes, args.max_edges, args.force)
    else:
        pep = ALL_PEPS[args.pep]
        exporter.export_for_pep(pep, args.max_nodes, args.max_edges, args.force)

    # List exported models
    exporter.list_exported_models()

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
