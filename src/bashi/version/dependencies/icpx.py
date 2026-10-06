"""Contains relationships with icpx and other parameter-values."""

from typing import List
from bashi.version.dependencies.base_version_support import ClangBase

# This list stores which ICPX version based on which Clang
# The list allows to reuse the knowledge of Clang and apply it on ICPX like the C++ standard
# support.
ICPX_CLANG_VERSION: List[ClangBase] = [
    ClangBase("2025.0", "19"),
    ClangBase("2025.1", "20"),
    ClangBase("2025.2", "21"),
    ClangBase("2025.3", "21"),
    ClangBase("2026.0", "22"),
    ClangBase("2026.1", "22"),
]
