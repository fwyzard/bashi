"""Filter rules which will generated during runtime depending on the input of the input
parameter-value-matrix"""

from typing import List, Dict
import packaging.version as pkv
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from bashi.version.dependencies.ubuntu import UbuntuSDKMinMax
from bashi.version.dependencies.base_version_support import CompilerCxxSupport
from bashi.version.dependencies.clang_cuda import ClangCudaSDKSupport
from bashi.types import ValueVersion


# pylint: disable=too-few-public-methods
class ValidUbuntuSDK:
    """Check if a given Ubuntu version is in the valid_ubuntu list."""

    def __init__(self, valid_ubuntus: List[Version]):
        self.valid_ubuntus = valid_ubuntus

    def __call__(self, ubuntu_ver: Version) -> bool:
        return ubuntu_ver in self.valid_ubuntus


def get_sdk_supporting_ubuntus(
    ubuntus: List[ValueVersion],
    sdk_versions: List[ValueVersion],
    ubuntu_sdk_version_range: List[UbuntuSDKMinMax],
):
    """Take a list of given Ubuntu and SDK versions and also a list of which SDK can be installed on
    which Ubuntu. Creates a validator object, which checks if a given Ubuntu version is in a list.
    The list contains a Ubuntu version if:

    - The Ubuntu version is in the argument ubuntus and
    - a version range is defined for the Ubuntu version in the argument ubuntu_hip_version_range and
    - there is a least one SDK version in the arguments sdk versions which matches the sdk version
        range of a given ubuntu sdk version range pair.

    Args:
        ubuntus (List[ValueVersion]): List of Ubuntu versions.
        sdk_versions (List[ValueVersion]): List of SDK versions
        ubuntu_sdk_version_range (List[UbuntuHipMinMax]): List of supported SDK versions for
            given Ubuntu versions.

    Raises:
        RuntimeError: If at least one of the input lists is empty

    Returns:
        ValidUbuntuSDK: An validator object with a call operator, which takes a Ubuntu version and
            returns True if the version is in the filtered list.
    """
    if not ubuntus or not sdk_versions or not ubuntu_sdk_version_range:
        raise RuntimeError(
            "It is not supported if arguments ubuntus, sdk_versions and ubuntu_sdk_version_range "
            "are empty."
        )

    # store which ubuntu version is valid
    valid_ubuntus: Dict[ValueVersion, bool] = {}
    for ub in ubuntus:
        valid_ubuntus[ub] = True

    # parse in a better suited data structure
    supported_ubuntus: Dict[ValueVersion, SpecifierSet] = {}
    for ubuntu, sdk_range in ubuntu_sdk_version_range:
        supported_ubuntus[ubuntu] = sdk_range

    # disable all ubuntu versions, which are not named in the ubuntu HIP SDK support list
    for ub_ver in valid_ubuntus:
        if ub_ver not in supported_ubuntus:
            valid_ubuntus[ub_ver] = False

    # disable all ubuntu versions, where no HIP SDK version is available for installation
    for ub_ver, valid in valid_ubuntus.items():
        if valid:
            found = False
            for sdk_ver in sdk_versions:
                if sdk_ver in supported_ubuntus[ub_ver]:
                    found = True
                    break

            if not found:
                valid_ubuntus[ub_ver] = False

    return ValidUbuntuSDK([ub for ub, valid in valid_ubuntus.items() if valid])


def cmake_supports_clang_cuda_with_cuda_13(cmake_version: ValueVersion) -> bool:
    """CMake 3.31.7 and older, and CMake 4.0.0 - 4.0.2, cannot identify Clang-CUDA as CUDA compiler
    with CUDA 13 and newer: the compiler identification compiles a test program for the sm_52, sm_30
    and sm_20 architectures, which are no longer supported by CUDA 13. Newer versions of CMake try
    sm_75 first, see https://gitlab.kitware.com/cmake/cmake/-/commit/b62f3257

    Args:
        cmake_version (ValueVersion): CMake version

    Returns:
        bool: True if the CMake version can identify Clang-CUDA with CUDA 13 and newer.
    """
    return not (
        cmake_version < pkv.parse("3.31.8")
        or pkv.parse("4.0") <= cmake_version < pkv.parse("4.0.3")
    )


# pylint: disable=too-few-public-methods
class AvailableCMakeForClangCudaCuda13:
    """Stores if the parameter-value-matrix contains a CMake version, which can identify Clang-CUDA
    as CUDA compiler with CUDA 13 and newer."""

    def __init__(self, available: bool):
        self.available = available

    def __call__(self) -> bool:
        return self.available


def get_cmake_for_clang_cuda_cuda_13(cmake_versions: List[ValueVersion]):
    """Check if at least one of the given CMake versions can identify Clang-CUDA as CUDA compiler
    with CUDA 13 and newer.

    Args:
        cmake_versions (List[ValueVersion]): CMake versions of the parameter-value-matrix

    Returns:
        AvailableCMakeForClangCudaCuda13: A functor which returns True if such a CMake version
            exists.
    """
    return AvailableCMakeForClangCudaCuda13(
        any(cmake_supports_clang_cuda_with_cuda_13(cmake) for cmake in cmake_versions)
    )


# pylint: disable=too-few-public-methods
class ValidClangCudaSDKCxx:
    """Check if at least one of the given Clang-CUDA versions supports a given CUDA SDK version and
    C++ standard."""

    def __init__(
        self,
        clang_cuda_versions: List[ValueVersion],
        clang_cuda_max_cuda_version: List[ClangCudaSDKSupport],
        clang_cuda_cxx_support: List[CompilerCxxSupport],
    ):
        self.clang_cuda_versions = clang_cuda_versions
        self.clang_cuda_max_cuda_version = sorted(clang_cuda_max_cuda_version, reverse=True)
        self.clang_cuda_cxx_support = sorted(clang_cuda_cxx_support, reverse=True)

    def _supports_cuda_sdk(self, clang_cuda_ver: Version, cuda_sdk_ver: Version) -> bool:
        # same as the filter rules c16 and b17: a Clang-CUDA version newer than the latest known
        # Clang-CUDA version supports every CUDA SDK version
        if (
            not self.clang_cuda_max_cuda_version
            or clang_cuda_ver > self.clang_cuda_max_cuda_version[0].clang_cuda
        ):
            return True
        for clang_cuda_sdk in self.clang_cuda_max_cuda_version:
            if clang_cuda_ver >= clang_cuda_sdk.clang_cuda:
                return cuda_sdk_ver <= clang_cuda_sdk.cuda
        return True

    def _supports_cxx(self, clang_cuda_ver: Version, cxx_ver: Version) -> bool:
        # same as _remove_unsupported_compiler_cxx_combination() in the compiler filter
        if not self.clang_cuda_cxx_support:
            return True
        for compiler_cxx in self.clang_cuda_cxx_support:
            if clang_cuda_ver >= compiler_cxx.compiler:
                return cxx_ver <= compiler_cxx.cxx
        return cxx_ver < self.clang_cuda_cxx_support[-1].cxx

    def __call__(self, cuda_sdk_ver: Version, cxx_ver: Version) -> bool:
        return any(
            self._supports_cuda_sdk(clang_cuda_ver, cuda_sdk_ver)
            and self._supports_cxx(clang_cuda_ver, cxx_ver)
            for clang_cuda_ver in self.clang_cuda_versions
        )


def get_clang_cuda_supporting_cuda_sdk_cxx(
    clang_cuda_versions: List[ValueVersion],
    clang_cuda_max_cuda_version: List[ClangCudaSDKSupport],
    clang_cuda_cxx_support: List[CompilerCxxSupport],
) -> ValidClangCudaSDKCxx:
    """Creates a validator object, which checks if at least one of the given Clang-CUDA versions
    supports a given CUDA SDK version and C++ standard. The Clang-CUDA versions are the versions in
    the input parameter-value-matrix. If there is no Clang-CUDA version in the input
    parameter-value-matrix, the validator returns always False.

    Args:
        clang_cuda_versions (List[ValueVersion]): List of Clang-CUDA versions.
        clang_cuda_max_cuda_version (List[ClangCudaSDKSupport]): Latest CUDA SDK version supported
            by a Clang-CUDA version.
        clang_cuda_cxx_support (List[CompilerCxxSupport]): Latest C++ standard supported by a
            Clang-CUDA version.

    Returns:
        ValidClangCudaSDKCxx: An validator object with a call operator, which takes a CUDA SDK
            version and a C++ standard and returns True if at least one of the Clang-CUDA versions
            supports both.
    """
    return ValidClangCudaSDKCxx(
        clang_cuda_versions, clang_cuda_max_cuda_version, clang_cuda_cxx_support
    )
