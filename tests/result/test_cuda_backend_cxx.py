# pylint: disable=missing-docstring
import unittest
from typing import Dict, Callable
from bashi.types import ParameterValuePair
from utils_test import (
    parse_expected_val_pairs,
    default_remove_test,
    parse_value_version,
)
from bashi.runtime_info import get_clang_cuda_supporting_cuda_sdk_cxx
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.version.relation import VersionRelation

from bashi.result_modules.cuda_support import (
    _remove_runtime_unsupported_cuda_backend_cxx_combinations,
)


class TestCUDABackendCxxRuntimeInfo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.version_relation = VersionRelation()
        cls.default_input: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (CXX_STANDARD, 23)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8), (CXX_STANDARD, 23)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0), (CXX_STANDARD, 20)),
                ((CXX_STANDARD, 23), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1), (CXX_STANDARD, 23)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1), (UBUNTU, 22.04)),
                ((HOST_COMPILER, CLANG_CUDA, 16), (CXX_STANDARD, 23)),
            ]
        )

    def get_runtime_infos(self, clang_cuda_versions) -> Dict[str, Callable[..., bool]]:
        return {
            RT_AVAILABLE_CLANG_CUDA_SDK_CXX: get_clang_cuda_supporting_cuda_sdk_cxx(
                parse_value_version(clang_cuda_versions),
                self.version_relation.get_clang_cuda_max_cuda_version(),
                self.version_relation.get_clang_cuda_cxx_support_version(),
            )
        }

    def test_empty_runtime_info(self):
        default_remove_test(
            _remove_runtime_unsupported_cuda_backend_cxx_combinations,
            self.default_input,
            list(self.default_input),
            self,
            version_relation=self.version_relation,
            runtime_infos={},
        )

    def test_clang_cuda_17(self):
        # Clang-CUDA 17 supports C++23 and CUDA up to 12.1
        default_remove_test(
            _remove_runtime_unsupported_cuda_backend_cxx_combinations,
            self.default_input,
            list(self.default_input),
            self,
            version_relation=self.version_relation,
            runtime_infos=self.get_runtime_infos([17]),
        )

    def test_clang_cuda_16(self):
        # Clang-CUDA 16 does not support C++23 and Nvcc does not support C++23, but Nvcc 12.0
        # supports C++20
        expected_output = parse_expected_val_pairs(
            [
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (CXX_STANDARD, 23)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1), (UBUNTU, 22.04)),
                ((HOST_COMPILER, CLANG_CUDA, 16), (CXX_STANDARD, 23)),
            ]
        )
        default_remove_test(
            _remove_runtime_unsupported_cuda_backend_cxx_combinations,
            self.default_input,
            expected_output,
            self,
            version_relation=self.version_relation,
            runtime_infos=self.get_runtime_infos([16]),
        )

    def test_no_clang_cuda(self):
        # without Clang-CUDA, C++20 with CUDA 11.8 is not possible
        expected_output = parse_expected_val_pairs(
            [
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (CXX_STANDARD, 23)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1), (UBUNTU, 22.04)),
                ((HOST_COMPILER, CLANG_CUDA, 16), (CXX_STANDARD, 23)),
            ]
        )
        default_remove_test(
            _remove_runtime_unsupported_cuda_backend_cxx_combinations,
            self.default_input,
            expected_output,
            self,
            version_relation=self.version_relation,
            runtime_infos=self.get_runtime_infos([]),
        )
