"""
Project Path Configuration

Centralized path management using relative paths for GitHub portability.
All modules should import from this file to ensure consistent paths.
"""

from pathlib import Path


# ============================================================================
# Project Root
# ============================================================================

# Get project root (where this file is located)
PROJECT_ROOT = Path(__file__).parent.resolve()


# ============================================================================
# Main Directories
# ============================================================================

# Profiling module
PROFILING_DIR = PROJECT_ROOT / 'profiling'
PROFILING_RESULTS_DIR = PROFILING_DIR / 'results'
PROFILING_MODELS_DIR = PROFILING_DIR / 'exported_models'

# Compiler module
COMPILER_DIR = PROJECT_ROOT / 'compiler'
COMPILER_OUTPUT_DIR = COMPILER_DIR / 'output'
COMPILER_MODELS_DIR = COMPILER_OUTPUT_DIR / 'models'

# Executor module
EXECUTOR_DIR = PROJECT_ROOT / 'executer'

# Logs
LOGS_DIR = PROJECT_ROOT / 'logs'


# ============================================================================
# Key Files
# ============================================================================

# Compiler outputs
COMPILATION_RESULT_FILE = COMPILER_OUTPUT_DIR / 'compilation_result.json'

# Profiling outputs
PROFILING_LOOKUP_TABLE = PROFILING_RESULTS_DIR / 'lookup_table.json'
PROFILING_BANDWIDTH_TABLE = PROFILING_RESULTS_DIR / 'bandwidth_table.json'

# Pipeline summary
PIPELINE_SUMMARY_FILE = PROJECT_ROOT / 'pipeline_summary.txt'


# ============================================================================
# Helper Functions
# ============================================================================

def ensure_dirs():
    """Create all necessary directories if they don't exist"""
    dirs = [
        PROFILING_RESULTS_DIR,
        PROFILING_MODELS_DIR,
        COMPILER_OUTPUT_DIR,
        COMPILER_MODELS_DIR,
        LOGS_DIR
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print(f"✓ All directories ensured in: {PROJECT_ROOT}")


def get_relative_path(absolute_path: Path) -> Path:
    """
    Convert absolute path to relative path from project root

    Args:
        absolute_path: Absolute path

    Returns:
        Path relative to project root
    """
    try:
        return absolute_path.relative_to(PROJECT_ROOT)
    except ValueError:
        # Path is outside project, return as is
        return absolute_path


def resolve_path(relative_path: str) -> Path:
    """
    Resolve a relative path string to absolute path

    Args:
        relative_path: Path relative to project root (e.g., "compiler/output")

    Returns:
        Absolute Path object
    """
    return (PROJECT_ROOT / relative_path).resolve()


def print_paths():
    """Print all configured paths (for debugging)"""
    print("=" * 70)
    print("Project Path Configuration")
    print("=" * 70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"\nMain Directories:")
    print(f"  Profiling:        {PROFILING_DIR}")
    print(f"  Compiler:         {COMPILER_DIR}")
    print(f"  Executor:         {EXECUTOR_DIR}")
    print(f"  Logs:             {LOGS_DIR}")
    print(f"\nOutput Directories:")
    print(f"  Profiling Results: {PROFILING_RESULTS_DIR}")
    print(f"  Compiler Output:   {COMPILER_OUTPUT_DIR}")
    print(f"  Compiler Models:   {COMPILER_MODELS_DIR}")
    print(f"\nKey Files:")
    print(f"  Compilation Result: {COMPILATION_RESULT_FILE}")
    print(f"  Profiling Lookup:   {PROFILING_LOOKUP_TABLE}")
    print(f"  Pipeline Summary:   {PIPELINE_SUMMARY_FILE}")
    print("=" * 70)


# ============================================================================
# Module-Specific Path Helpers
# ============================================================================

def get_profiling_result_path(filename: str) -> Path:
    """
    Get path to a profiling result file

    Args:
        filename: Result filename (e.g., "lookup_table.json")

    Returns:
        Absolute path to file
    """
    return PROFILING_RESULTS_DIR / filename


def get_compiler_output_path(filename: str) -> Path:
    """
    Get path to a compiler output file

    Args:
        filename: Output filename (e.g., "compilation_result.json")

    Returns:
        Absolute path to file
    """
    return COMPILER_OUTPUT_DIR / filename


def get_model_path(model_name: str) -> Path:
    """
    Get path to a model file

    Args:
        model_name: Model filename (e.g., "CPU_stages_1_2_3.onnx")

    Returns:
        Absolute path to model file
    """
    return COMPILER_MODELS_DIR / model_name


def get_log_path(log_name: str) -> Path:
    """
    Get path to a log file

    Args:
        log_name: Log filename (e.g., "compiler_output.log")

    Returns:
        Absolute path to log file
    """
    return LOGS_DIR / log_name


# ============================================================================
# Initialization
# ============================================================================

# Automatically create directories when module is imported
ensure_dirs()


# ============================================================================
# Export for convenience
# ============================================================================

__all__ = [
    # Root
    'PROJECT_ROOT',

    # Directories
    'PROFILING_DIR',
    'PROFILING_RESULTS_DIR',
    'PROFILING_MODELS_DIR',
    'COMPILER_DIR',
    'COMPILER_OUTPUT_DIR',
    'COMPILER_MODELS_DIR',
    'EXECUTOR_DIR',
    'LOGS_DIR',

    # Files
    'COMPILATION_RESULT_FILE',
    'PROFILING_LOOKUP_TABLE',
    'PROFILING_BANDWIDTH_TABLE',
    'PIPELINE_SUMMARY_FILE',

    # Functions
    'ensure_dirs',
    'get_relative_path',
    'resolve_path',
    'print_paths',
    'get_profiling_result_path',
    'get_compiler_output_path',
    'get_model_path',
    'get_log_path',
]


if __name__ == '__main__':
    # Test the configuration
    print_paths()

    # Verify all directories exist
    print("\nVerifying directories...")
    for name, path in [
        ('Profiling Results', PROFILING_RESULTS_DIR),
        ('Compiler Output', COMPILER_OUTPUT_DIR),
        ('Compiler Models', COMPILER_MODELS_DIR),
        ('Logs', LOGS_DIR),
    ]:
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {name}: {path}")
