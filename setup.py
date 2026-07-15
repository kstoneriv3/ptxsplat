import glob
import os
import os.path as osp
import pathlib
import platform
import sys

from setuptools import find_packages, setup

__version__ = None
exec(open("ptxsplat/version.py", "r").read())

URL = "https://github.com/kstoneriv3/ptxsplat"

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"
MAX_JOBS = os.getenv("MAX_JOBS")
need_to_unset_max_jobs = False
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    os.environ["MAX_JOBS"] = "10"
    print(f"Setting MAX_JOBS to {os.environ['MAX_JOBS']}")


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_extensions():
    import torch
    from torch.__config__ import parallel_info
    from torch.utils.cpp_extension import CUDAExtension

    extensions_dir = osp.join("ptxsplat", "cuda")
    sources = glob.glob(osp.join(extensions_dir, "csrc", "*.cu")) + glob.glob(
        osp.join(extensions_dir, "csrc", "*.cpp")
    )
    sources += [osp.join(extensions_dir, "ext.cpp")]

    undef_macros = []
    define_macros = []

    extra_compile_args = {"cxx": ["-O3"]}
    if not os.name == "nt":  # Not on Windows:
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]
    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    info = parallel_info()
    if (
        "backend: OpenMP" in info
        and "OpenMP not found" not in info
        and sys.platform != "darwin"
    ):
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_compile_args["cxx"] += ["/openmp"]
        else:
            extra_compile_args["cxx"] += ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
    nvcc_flags += ["-O3", "--use_fast_math", "-std=c++17"]
    if LINE_INFO:
        nvcc_flags += ["-lineinfo"]
    if torch.version.hip:
        # USE_ROCM was added to later versions of PyTorch.
        # Define here to support older PyTorch versions as well:
        define_macros += [("USE_ROCM", None)]
        undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
    else:
        nvcc_flags += ["--expt-relaxed-constexpr"]

    # GLM/Torch has spammy and very annoyingly verbose warnings that this suppresses
    nvcc_flags += ["-diag-suppress", "20012,186"]
    extra_compile_args["nvcc"] = nvcc_flags
    if sys.platform == "win32":
        extra_compile_args["nvcc"] += ["-DWIN32_LEAN_AND_MEAN", "-allow-unsupported-compiler"]

    current_dir = pathlib.Path(__file__).parent.resolve()
    glm_path = osp.join(current_dir, "ptxsplat", "cuda", "csrc", "third_party", "glm")
    include_dirs = [glm_path, osp.join(current_dir, "ptxsplat", "cuda", "include")]

    extension = CUDAExtension(
        "ptxsplat.csrc",
        sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    return [extension]


setup(
    name="ptxsplat",
    version=__version__,
    description="RTX-optimized differentiable Gaussian rasterization",
    keywords="gaussian, splatting, cuda",
    url=URL,
    download_url=f"{URL}/archive/v{__version__}.tar.gz",
    python_requires=">=3.10",
    install_requires=[
        "ninja",
        "numpy",
        "jaxtyping",
        "rich>=12",
        "torch",
        "typing_extensions; python_version<'3.8'",
    ],
    extras_require={
        "gsplat_overload": [f"ptxsplat-gsplat-overload=={__version__}"],
        "ptx": ["pyptx[torch]==0.1.1", "cuda-python>=12.3"],
        # dev dependencies. Install them by `pip install ptxsplat[dev]`
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.2",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml==6.0",
            "build",
            "twine",
            "pyptx[torch]==0.1.1",
            "cuda-python>=12.3",
        ],
    },
    ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
    cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
    packages=find_packages(),
    # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
    include_package_data=True,
)

if need_to_unset_max_jobs:
    print("Unsetting MAX_JOBS")
    os.environ.pop("MAX_JOBS")
