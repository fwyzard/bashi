# pylint: disable=missing-docstring
import unittest
import copy
import packaging.version as pkv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.version import VERSIONS
from bashi.version.utils import get_parameter_value_matrix
from bashi.version.relation import VersionRelation
from bashi.version.dependencies.base_version_support import CompilerCxxSupport
from bashi.version.dependencies.clang_cuda import ClangCudaSDKSupport
from bashi.runtime_info import get_clang_cuda_supporting_cuda_sdk_cxx
from bashi.generator import get_runtime_infos
from utils_test import parse_value_version as pvv


class TestGetClangCudaSupportingCudaSdkCxx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clang_cuda_max_cuda_version = [
            ClangCudaSDKSupport("14", "11.5"),
            ClangCudaSDKSupport("16", "11.8"),
            ClangCudaSDKSupport("17", "12.1"),
            ClangCudaSDKSupport("18", "12.3"),
            ClangCudaSDKSupport("19", "12.5"),
            ClangCudaSDKSupport("20", "12.8"),
        ]
        cls.clang_cuda_cxx_support = [
            CompilerCxxSupport("9", "17"),
            CompilerCxxSupport("14", "20"),
            CompilerCxxSupport("17", "23"),
        ]

    def get_validator(self, clang_cuda_versions):
        return get_clang_cuda_supporting_cuda_sdk_cxx(
            pvv(clang_cuda_versions),
            self.clang_cuda_max_cuda_version,
            self.clang_cuda_cxx_support,
        )

    def test_no_clang_cuda_version(self):
        validator = self.get_validator([])
        self.assertFalse(validator(pkv.parse("11.0"), pkv.parse("17")))
        self.assertFalse(validator(pkv.parse("12.1"), pkv.parse("23")))

    def test_clang_cuda_versions(self):
        validator = self.get_validator([16, 17, 18, 19])
        self.assertTrue(validator(pkv.parse("11.8"), pkv.parse("20")))
        self.assertTrue(validator(pkv.parse("12.1"), pkv.parse("23")))
        self.assertTrue(validator(pkv.parse("12.5"), pkv.parse("23")))
        # CUDA 12.6 requires Clang-CUDA 20
        self.assertFalse(validator(pkv.parse("12.6"), pkv.parse("20")))
        self.assertFalse(validator(pkv.parse("12.6"), pkv.parse("23")))
        # C++26 is not supported by any Clang-CUDA version
        self.assertFalse(validator(pkv.parse("12.1"), pkv.parse("26")))

    def test_old_clang_cuda_versions(self):
        validator = self.get_validator([14, 16])
        self.assertTrue(validator(pkv.parse("11.8"), pkv.parse("20")))
        # C++23 requires Clang-CUDA 17
        self.assertFalse(validator(pkv.parse("11.8"), pkv.parse("23")))
        # CUDA 12.0 requires Clang-CUDA 17
        self.assertFalse(validator(pkv.parse("12.0"), pkv.parse("20")))

    def test_clang_cuda_version_older_than_table(self):
        # a Clang-CUDA version older than the oldest table entry is not restricted by the CUDA SDK
        # table, but supports only C++ standards older than the oldest C++ support entry
        validator = self.get_validator([8])
        self.assertTrue(validator(pkv.parse("12.6"), pkv.parse("14")))
        self.assertFalse(validator(pkv.parse("12.6"), pkv.parse("17")))

    def test_clang_cuda_version_newer_than_table(self):
        # a Clang-CUDA version newer than the latest table entry supports every CUDA SDK version
        validator = self.get_validator([21])
        self.assertTrue(validator(pkv.parse("13.5"), pkv.parse("23")))

    def test_get_runtime_infos(self):
        versions = copy.deepcopy(VERSIONS)
        versions[CLANG] = [16, 17, 18]
        versions[CLANG_CUDA] = [16, 17, 18]
        version_relation = VersionRelation(
            clang_cuda_max_cuda_version=copy.deepcopy(self.clang_cuda_max_cuda_version)
        )
        runtime_infos = get_runtime_infos(get_parameter_value_matrix(versions), version_relation)
        self.assertIn(RT_AVAILABLE_CLANG_CUDA_SDK_CXX, runtime_infos)
        validator = runtime_infos[RT_AVAILABLE_CLANG_CUDA_SDK_CXX]
        self.assertTrue(validator(pkv.parse("12.3"), pkv.parse("23")))
        self.assertFalse(validator(pkv.parse("12.4"), pkv.parse("23")))

    def test_get_runtime_infos_without_clang_cuda(self):
        versions = copy.deepcopy(VERSIONS)
        del versions[CLANG_CUDA]
        runtime_infos = get_runtime_infos(get_parameter_value_matrix(versions), VersionRelation())
        self.assertFalse(
            runtime_infos[RT_AVAILABLE_CLANG_CUDA_SDK_CXX](pkv.parse("12.1"), pkv.parse("23"))
        )
