"""Tests for NequIP model registration and calculator construction."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
from types import ModuleType, SimpleNamespace

from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.data import chemical_symbols
from mlipx.nodes.generic_ase import Device
import numpy as np
import pytest
import requests
import yaml

from ml_peg.models import models
from ml_peg.models.get_models import load_models
from ml_peg.models.models import NequIPCalc

NEQUIP_MODELS = (
    "nequip-oam-s-0p2",
    "nequip-oam-m-0p2",
    "nequip-oam-l-0p2",
    "nequip-oam-xl-0p2",
    "allegro-oam-s-0p2",
    "allegro-oam-m-0p2",
    "allegro-oam-l-0p2",
)
# The 89 atom types of every OAM package: H-Bi and Ac-Pu
OAM_ELEMENTS = set(chemical_symbols[1:84] + chemical_symbols[89:95])
ROOT = Path(__file__).parents[1]
# OAM-S energy of the rattled NaCl cell used in the real-model test below
REFERENCE_ENERGY = -216.15304486661688


def _registry() -> dict:
    """Load the default model registry."""
    path = ROOT / "ml_peg" / "models" / "models.yml"
    return yaml.safe_load(path.read_text(encoding="utf8"))


@contextlib.contextmanager
def _torch_default_dtype(dtype):
    """Set torch's default dtype inside the block, as ``nequip.utils.dtype`` does."""
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class _FakeNequIPCalculator:
    """Minimal stand-in for the NequIP ASE calculator."""

    loads = []

    def __init__(self, source):
        self.source = source

    def call_model(self, data):
        """Report which model evaluated ``data``."""
        return self.source

    def model(self, data):
        """Report which model evaluated ``data``."""
        return self.source

    @classmethod
    def _from_saved_model(cls, model_path, **kwargs):
        """Record an eager load, making float64 the default like nequip's first load."""
        import torch

        torch.set_default_dtype(torch.float64)
        cls.loads.append(("eager", model_path, kwargs))
        return cls("eager")

    @classmethod
    def from_compiled_model(cls, compile_path, **kwargs):
        """Record a compiled load."""
        cls.loads.append(("compiled", compile_path, kwargs))
        return cls("compiled")


def _fake_nequip_module() -> ModuleType:
    """Return a fake top-level ``nequip`` module."""
    module = ModuleType("nequip")
    module.__version__ = "0.0.0"
    return module


@pytest.fixture
def fake_nequip(monkeypatch):
    """Provide fake ``nequip`` modules, recording downloads and compilations."""
    ase_module = ModuleType("nequip.integrations.ase")
    ase_module.NequIPCalculator = _FakeNequIPCalculator
    data_module = ModuleType("nequip.data")
    data_module.AtomicDataDict = SimpleNamespace(
        num_nodes=lambda data: len(data["pos"]),
        num_edges=lambda data: data["edge_index"].shape[1],
    )
    dtype_module = ModuleType("nequip.utils.dtype")
    dtype_module.torch_default_dtype = _torch_default_dtype
    modify_module = ModuleType("nequip.model.modify_utils")
    modify_module.get_all_modifiers = lambda model: {}
    fake_modules = {
        "nequip": _fake_nequip_module(),
        "nequip.data": data_module,
        "nequip.integrations": ModuleType("nequip.integrations"),
        "nequip.integrations.ase": ase_module,
        "nequip.model": ModuleType("nequip.model"),
        "nequip.model.modify_utils": modify_module,
        "nequip.utils": ModuleType("nequip.utils"),
        "nequip.utils.dtype": dtype_module,
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    calls = {"downloads": [], "compiles": []}

    def download(url):
        calls["downloads"].append(url)
        return "cached.nequip.zip"

    def compile_package(model, device, mode="aotinductor", modifier=None):
        calls["compiles"].append((model, device, mode, modifier))
        return "compiled.nequip.pt2"

    monkeypatch.setattr(models, "_download_nequip_package", download)
    monkeypatch.setattr(models, "_compile_nequip_package", compile_package)
    monkeypatch.setattr(_FakeNequIPCalculator, "loads", [])
    models._load_nequip_calculator.cache_clear()
    yield calls
    models._load_nequip_calculator.cache_clear()


def test_nequip_registry_metadata():
    """NequIP entries point at the Zenodo packages and record their D3 behaviour."""
    registry = _registry()

    for name in NEQUIP_MODELS:
        config = registry[name]
        assert config["class_name"] == "NequIPCalculator"
        assert config["datasets"] == ["MPtrj", "sAlex", "OMAT"]
        assert config["device"] == "auto"
        assert config["trained_on_dispersion"] is False
        assert "overwrite_dtype" not in config
        assert config["kwargs"]["model"].startswith(
            "https://zenodo.org/api/records/22349749/"
        )


def test_nequip_registry_records_package_elements():
    """The registry datasets cover exactly the 89 atom types of the OAM packages."""
    registry = _registry()
    coverage = json.loads(
        (ROOT / "ml_peg" / "app" / "data" / "element_coverage.json").read_text(
            encoding="utf8"
        )
    )["datasets"]

    for name in NEQUIP_MODELS:
        config = registry[name]
        supported = set(config.get("additional_supported_elements", []))
        for dataset in config["datasets"]:
            supported.update(coverage[dataset]["supported"])
        assert supported == OAM_ELEMENTS


def test_nequip_models_use_dedicated_wrapper():
    """All registered NequIP models route through the dedicated wrapper."""
    loaded = load_models(NEQUIP_MODELS)

    assert tuple(loaded) == NEQUIP_MODELS
    assert all(isinstance(model, NequIPCalc) for model in loaded.values())


@pytest.mark.parametrize("precision", ["low", "high"])
def test_nequip_loads_compiled_package_once(fake_nequip, precision):
    """The wrapper compiles the package and loads it once, handing out copies."""
    import torch

    kwargs = _registry()["nequip-oam-s-0p2"]["kwargs"]
    model = load_models(("nequip-oam-s-0p2",))["nequip-oam-s-0p2"]
    default_dtype = torch.get_default_dtype()

    first = model.get_calculator(precision=precision)
    second = model.get_calculator(precision=precision)

    device = Device.resolve_auto()
    options = {"device": device, "chemical_species_to_atom_type_map": True}
    assert fake_nequip["downloads"] == [kwargs["model"]] * 2
    assert fake_nequip["compiles"] == [
        ("cached.nequip.zip", device, "aotinductor", None)
    ]
    assert _FakeNequIPCalculator.loads == [
        ("eager", "cached.nequip.zip", options),
        ("compiled", "compiled.nequip.pt2", options),
    ]
    assert first is not second
    assert first.source == second.source == "compiled"
    assert torch.get_default_dtype() == default_dtype


def test_nequip_runs_small_structures_eagerly(fake_nequip):
    """Structures below the compiled model's shape bounds use the eager model."""
    calc = NequIPCalc(kwargs={"model": "local.nequip.zip"}).get_calculator(
        precision="high"
    )

    def data(atoms, edges):
        return {"pos": np.zeros((atoms, 3)), "edge_index": np.zeros((2, edges))}

    assert calc.call_model(data(1, 0)) == "eager"
    assert calc.call_model(data(1, 8)) == "eager"
    assert calc.call_model(data(2, 0)) == "eager"
    assert calc.call_model(data(2, 2)) == "compiled"


def test_nequip_uses_eager_model_if_compiling_fails(fake_nequip, monkeypatch):
    """A failed compilation warns and returns the eager calculator."""

    def fail(model, device, mode, modifier):
        raise subprocess.CalledProcessError(1, "nequip-compile", stderr="no compiler")

    monkeypatch.setattr(models, "_compile_nequip_package", fail)
    model = NequIPCalc(kwargs={"model": "local.nequip.zip"})

    with pytest.warns(UserWarning) as warnings:
        calc = model.get_calculator(precision="high")
    assert calc.source == "eager"
    assert "no compiler" in str(warnings[0].message)
    assert "Using the eager NequIP model" in str(warnings[-1].message)


def test_nequip_compilation_can_be_disabled(fake_nequip):
    """Setting ``compile`` to false loads only the eager model."""
    model = NequIPCalc(compile=False, kwargs={"model": "local.nequip.zip"})

    assert model.get_calculator(precision="high").source == "eager"
    assert fake_nequip["compiles"] == []


def test_nequip_rejects_unknown_precision():
    """The NequIP wrapper rejects unknown ML-PEG precision choices."""
    model = NequIPCalc(kwargs={"model": "model.nequip.zip"})

    with pytest.raises(ValueError, match="Invalid precision"):
        model.get_calculator(precision="medium")


def test_nequip_wrapper_round_trips():
    """The wrapper survives pickling (joblib) and zntrack's params.yaml rebuild."""
    model = load_models(("nequip-oam-s-0p2",))["nequip-oam-s-0p2"]

    assert pickle.loads(pickle.dumps(model)) == model
    params = yaml.safe_load(yaml.safe_dump(dataclasses.asdict(model)))
    assert NequIPCalc(**params) == model


def test_download_caches(monkeypatch, tmp_path):
    """Packages are downloaded once per URL."""
    content = b"package"
    calls = []

    def get(url, timeout):
        calls.append(url)
        return SimpleNamespace(content=content, raise_for_status=lambda: None)

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(requests, "get", get)

    path = models._download_nequip_package("https://example.org/a")
    assert models._download_nequip_package("https://example.org/a") == path
    cached = Path(path)
    assert cached.name.endswith(".nequip.zip")
    assert cached.read_bytes() == content
    assert calls == ["https://example.org/a"]
    assert [path.name for path in cached.parent.iterdir()] == [cached.name]


def test_compile_caches_by_package_and_hardware(monkeypatch, tmp_path):
    """Packages compile once per package, device, hardware, threads and versions."""
    import torch

    package = tmp_path / "model.nequip.zip"
    package.write_bytes(b"package")
    runs = []

    def run(command, **kwargs):
        runs.append((command, kwargs["env"]))
        Path(command[4]).write_bytes(b"compiled")

    monkeypatch.setitem(sys.modules, "nequip", _fake_nequip_module())
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ATEN_CPU_CAPABILITY", raising=False)
    monkeypatch.setattr(subprocess, "run", run)

    path = Path(models._compile_nequip_package(str(package), "cpu"))
    assert models._compile_nequip_package(str(package), "cpu") == str(path)

    assert len(runs) == 1
    command, env = runs[0]
    assert command[1:4] == ["-m", "nequip.scripts.compile", str(package)]
    assert command[5:] == [
        "--mode",
        "aotinductor",
        "--device",
        "cpu",
        "--target",
        "ase",
    ]
    # Scalar C++ codegen on every CPU
    assert env["ATEN_CPU_CAPABILITY"] == "default"
    assert path.name.startswith(f"{hashlib.sha256(b'package').hexdigest()[:16]}-cpu-")
    assert f"-cpu-{platform.machine()}-" in path.name
    assert "-simd_default-" in path.name
    threads = torch.get_num_threads()
    assert env["OMP_NUM_THREADS"] == str(threads)
    assert f"-threads{threads}-" in path.name
    assert path.name.endswith(
        f"-aotinductor-torch{torch.__version__}-nequip0.0.0.nequip.pt2"
    )
    assert path.read_bytes() == b"compiled"
    files = [file.name for file in path.parent.iterdir()]
    assert [name for name in files if not name.endswith(".lock")] == [path.name]

    # Compiled kernels keep their thread count, so other counts compile separately
    monkeypatch.setattr(torch, "get_num_threads", lambda: threads + 1)
    other = models._compile_nequip_package(str(package), "cpu")
    assert other == str(path).replace(f"-threads{threads}-", f"-threads{threads + 1}-")
    assert len(runs) == 2
    assert runs[1][1]["OMP_NUM_THREADS"] == str(threads + 1)


def test_compile_skips_kernel_variants_that_failed(monkeypatch, tmp_path):
    """A failed GPU-kernel compile is recorded and not retried; plain ones are."""
    package = tmp_path / "model.nequip.zip"
    package.write_bytes(b"package")
    runs = []

    def run(command, **kwargs):
        runs.append(command)
        raise subprocess.CalledProcessError(1, command, stderr="no nvcc")

    monkeypatch.setitem(sys.modules, "nequip", _fake_nequip_module())
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(models, "version", lambda name: "1.0")
    kernel = ("torchscript", "enable_OpenEquivariance")

    with pytest.raises(subprocess.CalledProcessError):
        models._compile_nequip_package(str(package), "cpu", *kernel)
    assert runs[-1][-2:] == ["--modifiers", "enable_OpenEquivariance"]
    with pytest.raises(RuntimeError, match="compiling failed before"):
        models._compile_nequip_package(str(package), "cpu", *kernel)
    (failed,) = (tmp_path / ".cache" / "ml_peg" / "nequip" / "compiled").glob(
        "*.failed"
    )
    assert "openequivariance1.0" in failed.name
    assert failed.read_text() == "no nvcc"

    for _ in range(2):
        with pytest.raises(subprocess.CalledProcessError):
            models._compile_nequip_package(str(package), "cpu")
    assert len(runs) == 3


def test_nequip_compile_variants(monkeypatch):
    """GPU kernels are tried first on CUDA, when installed, then plain AOTInductor."""
    import torch

    nequip = {"enable_OpenEquivariance", "enable_CuEquivariance"}
    allegro = {"enable_CuEquivarianceContracter", "enable_TritonContracter"}
    plain = ("aotinductor", None)
    installed = {"openequivariance", "cuequivariance_torch"}
    monkeypatch.setattr(models, "find_spec", lambda name: name in installed)
    monkeypatch.setattr(torch, "__version__", "2.8.0+cu128")

    assert models._nequip_compile_variants(nequip, "cpu") == [plain]
    assert models._nequip_compile_variants(nequip, "cuda") == [
        ("torchscript", "enable_OpenEquivariance"),
        ("aotinductor", "enable_CuEquivariance"),
        plain,
    ]
    assert models._nequip_compile_variants(allegro, "cuda") == [
        ("aotinductor", "enable_CuEquivarianceContracter"),
        plain,
    ]
    monkeypatch.setattr(torch, "__version__", "2.10.0")
    assert models._nequip_compile_variants(nequip, "cuda")[0] == (
        "aotinductor",
        "enable_OpenEquivariance",
    )
    installed.clear()
    assert models._nequip_compile_variants(nequip | allegro, "cuda") == [plain]


def test_nequip_falls_back_from_failed_gpu_kernels(fake_nequip, monkeypatch):
    """Kernel variants that fail, or change the results, fall back to plain ones."""
    checks = []

    def check(compiled, eager):
        checks.append((compiled.source, eager.source))
        raise RuntimeError("the GPU kernels change the energy")

    variants = [
        ("torchscript", "enable_OpenEquivariance"),
        ("aotinductor", "enable_CuEquivariance"),
        ("aotinductor", None),
    ]
    monkeypatch.setitem(sys.modules, "openequivariance", ModuleType("openequivariance"))
    # Fail to import cuEquivariance, whether or not it is installed
    monkeypatch.setitem(sys.modules, "cuequivariance_torch", None)
    monkeypatch.setattr(models, "_nequip_compile_variants", lambda *args: variants)
    monkeypatch.setattr(models, "_check_nequip_kernels", check)
    model = NequIPCalc(kwargs={"model": "local.nequip.zip"})

    with pytest.warns(UserWarning) as warnings:
        calc = model.get_calculator(precision="high")

    assert calc.source == "compiled"
    assert [compile[2:] for compile in fake_nequip["compiles"]] == [
        variants[0],
        variants[2],
    ]
    assert checks == [("compiled", "eager")]
    messages = [str(warning.message) for warning in warnings]
    assert "change the energy" in messages[0]
    assert "cuequivariance_torch" in messages[1]


def test_check_nequip_kernels():
    """Kernels must reproduce the eager energy and forces to 1 meV and 1 meV/A."""
    models._check_nequip_kernels(LennardJones(), LennardJones())
    with pytest.raises(RuntimeError, match="change the energy"):
        models._check_nequip_kernels(LennardJones(epsilon=1.1), LennardJones())


# Failed compilation falls back to the eager model with this warning
@pytest.mark.filterwarnings("error:Using the eager NequIP model")
def test_nequip_oam_s_compiled_matches_eager():
    """The compiled OAM-S package matches the eager model, including tiny structures."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("nequip")
    default_dtype = torch.get_default_dtype()
    models._load_nequip_calculator.cache_clear()
    model = load_models(("nequip-oam-s-0p2",))["nequip-oam-s-0p2"]
    compiled = model.get_calculator(precision="high")
    eager = dataclasses.replace(model, compile=False).get_calculator(precision="high")

    assert torch.get_default_dtype() == default_dtype
    assert eager.model.metadata["model_dtype"] == "float32"
    assert set(eager.model.metadata["type_names"].split()) == OAM_ELEMENTS

    cell = bulk("NaCl", "rocksalt", a=5.64, cubic=True) * (2, 2, 2)
    cell.rattle(0.05, seed=1)
    structures = (
        cell,
        bulk("Fe", "bcc", a=2.87),
        Atoms("Na"),
        # No edges beyond the 4.5 A cutoff, then two edges within it
        Atoms("NaCl", positions=[[0, 0, 0], [0, 0, 10]]),
        Atoms("NaCl", positions=[[0, 0, 0], [0, 0, 2.4]]),
    )

    def evaluate(calc, atoms, dtype=torch.float32):
        """Evaluate a fresh copy of ``atoms``, bypassing ASE's results cache."""
        calc.reset()
        atoms = atoms.copy()
        atoms.calc = calc
        with _torch_default_dtype(dtype):
            return atoms.get_potential_energy(), atoms.get_forces()

    assert evaluate(eager, cell)[0] == pytest.approx(REFERENCE_ENERGY, abs=1e-3)
    for atoms in structures:
        energy, forces = evaluate(eager, atoms, torch.float64)
        # Without the float64 guard, a float32 default gives NaN compiled results
        for calc in (compiled, copy.copy(compiled), copy.deepcopy(compiled)):
            compiled_energy, compiled_forces = evaluate(calc, atoms)
            assert compiled_forces.dtype == np.float64
            assert compiled_energy == pytest.approx(energy, abs=1e-4)
            np.testing.assert_allclose(compiled_forces, forces, atol=1e-4)

    cell.calc = compiled
    stress = cell.get_stress()
    cell.calc = eager
    np.testing.assert_allclose(stress, cell.get_stress(), atol=1e-5)
