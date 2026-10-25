"""Define classes for all models."""

# ruff: noqa: D101, D102, F401

from __future__ import annotations

import copy
import dataclasses
from functools import cache, wraps
import hashlib
from importlib import import_module
from importlib.metadata import packages_distributions, version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Literal, get_args
from warnings import warn

from mlipx import GenericASECalculator as MlipxGenericASECalc
from mlipx.nodes.generic_ase import Device

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from ase.calculators.mixing import SumCalculator

Precision = Literal["low", "high"]


def check_precision(precision: str) -> None:
    """
    Validate the precision argument.

    Parameters
    ----------
    precision
        Precision string to validate.

    Raises
    ------
    ValueError
        If the precision is not one of "low" or "high".
    """
    if precision not in get_args(Precision):
        raise ValueError(
            f"Invalid precision '{precision}'. Must be one of {get_args(Precision)}."
        )


def _patch_metatomic_nvalchemi_max_neighbors() -> None:
    """
    Make metatomic's CUDA neighbor-list call compatible with nvalchemi.

    ``metatomic-ase`` computes ``max_neighbors`` using ``cutoff**3``. For
    cutoffs above roughly 5 Å this produces a float, while nvalchemi requires
    an integer tensor dimension. PET-OAM uses a 10 Å cutoff.
    """
    import metatomic_ase._neighbors as metatomic_neighbors

    neighbor_list = metatomic_neighbors.nvalchemi_neighbor_list
    if getattr(neighbor_list, "_ml_peg_integer_max_neighbors", False):
        return

    @wraps(neighbor_list)
    def neighbor_list_with_integer_max_neighbors(*args: Any, **kwargs: Any) -> Any:
        """
        Convert nvalchemi's maximum-neighbor estimate to an integer.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the original neighbor-list function.
        **kwargs
            Keyword arguments forwarded to the original neighbor-list function.

        Returns
        -------
        Any
            The result from the original neighbor-list function.
        """
        max_neighbors = kwargs.get("max_neighbors")
        if max_neighbors is not None:
            kwargs["max_neighbors"] = int(max_neighbors)
        return neighbor_list(*args, **kwargs)

    neighbor_list_with_integer_max_neighbors._ml_peg_integer_max_neighbors = True
    metatomic_neighbors.nvalchemi_neighbor_list = (
        neighbor_list_with_integer_max_neighbors
    )


# Modules providing the GPU kernels of each nequip modifier, in order of preference.
# cuequivariance_torch quietly runs plain PyTorch if its CUDA ops can't load (e.g. the
# CUDA 12 build with a CUDA 13 torch), so we import the ops themselves
_NEQUIP_KERNEL_PACKAGES = {
    "enable_OpenEquivariance": "openequivariance",
    "enable_CuEquivariance": "cuequivariance_ops_torch",
    "enable_CuEquivarianceContracter": "cuequivariance_ops_torch",
}


# TODO: load the models from nequip.net once it serves the 0.2 packages, replacing this
# Zenodo download
def _download_nequip_package(url: str) -> str:
    """
    Download a NequIP package once and return the local path.

    Parameters
    ----------
    url
        URL of the ``.nequip.zip`` package, e.g. a Zenodo file ``content`` link.

    Returns
    -------
    str
        Path to the cached package.
    """
    import requests

    # nequip only treats files ending in .nequip.zip as packages
    name = hashlib.sha256(url.encode()).hexdigest()[:16]
    path = Path.home() / ".cache" / "ml_peg" / "nequip" / f"{name}.nequip.zip"
    if not path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            file.write(response.content)
        # Concurrent first uses may download twice; the rename keeps it atomic
        Path(file.name).replace(path)
    return str(path)


def _nequip_compile_variants(
    modifiers: set[str], device: str
) -> list[tuple[str, str | None]]:
    """
    List the ways to compile a NequIP package for a device, fastest first.

    On CUDA, NequIP models use OpenEquivariance kernels if installed, then
    cuEquivariance, and Allegro models use cuEquivariance. AOTInductor without GPU
    kernels comes last.

    Parameters
    ----------
    modifiers
        Names of the modifiers the package supports.
    device
        Device the model will run on.

    Returns
    -------
    list[tuple[str, str | None]]
        ``nequip-compile`` mode and GPU kernel modifier (or ``None``) of each variant.
    """
    import torch

    variants = []
    if torch.device(device).type == "cuda":
        torch_version = tuple(int(part) for part in torch.__version__.split(".")[:2])
        for modifier, package in _NEQUIP_KERNEL_PACKAGES.items():
            if modifier in modifiers and find_spec(package):
                # OpenEquivariance needs torch 2.9 for AOTInductor, so use TorchScript
                # (which torch 2.10 drops) before then
                old = modifier == "enable_OpenEquivariance" and torch_version < (2, 9)
                variants.append(("torchscript" if old else "aotinductor", modifier))
    return [*variants, ("aotinductor", None)]


def _compile_nequip_package(
    model: str, device: str, mode: str = "aotinductor", modifier: str | None = None
) -> str:
    """
    Compile a NequIP package once per machine type and thread count, returning the file.

    Parameters
    ----------
    model
        Path to the ``.nequip.zip`` package.
    device
        Device to compile for, which the compiled model must then be loaded on.
    mode
        ``nequip-compile`` mode, ``aotinductor`` or ``torchscript``.
    modifier
        GPU kernel modifier to compile with. If compiling with it fails, later calls
        raise without retrying until the ``.failed`` file beside the cached model is
        deleted.

    Returns
    -------
    str
        Path to the cached compiled model.

    Raises
    ------
    RuntimeError
        If compiling with ``modifier`` failed before.
    """
    from filelock import FileLock
    import nequip
    import torch

    env = dict(os.environ)
    # Compiled code is specific to the CPU's instruction set or the GPU model
    if torch.device(device).type == "cuda":
        hardware = torch.cuda.get_device_name(device)
    else:
        hardware = f"{platform.machine()}-{torch.backends.cpu.get_cpu_capability()}"
        # Scalar C++ is faster on x86, avoids a torch 2.10-2.11 compile failure, and
        # from torch 2.9 avoids vectorised NequIP kernels opening an OpenMP region per
        # edge (4-14x slower multithreaded on macOS)
        # TODO: nequip > 0.19.1 compiles scalar C++ unless ATEN_CPU_CAPABILITY is set,
        # so stop setting it here (but keep it in the key) once that is required
        env.setdefault("ATEN_CPU_CAPABILITY", "default")
        hardware += f"-simd_{env['ATEN_CPU_CAPABILITY']}"
        # Compiled kernels keep the thread count they were compiled with, so use ours
        env["OMP_NUM_THREADS"] = str(torch.get_num_threads())
        hardware += f"-threads{env['OMP_NUM_THREADS']}"
    if sys.platform == "linux":
        # torch compiles with -march=native here, so also key on the exact host CPU
        cpu = re.findall(
            r"^(?:model name|flags|Features|CPU part)\s*:.*$",
            Path("/proc/cpuinfo").read_text(),
            re.M,
        )
        cpu_id = hashlib.sha256("\n".join(sorted(set(cpu))).encode()).hexdigest()
        hardware += f"-{cpu_id[:12]}"
    digest = hashlib.sha256(Path(model).read_bytes()).hexdigest()
    key = [digest[:16], device, hardware, mode]
    if modifier:
        package = packages_distributions()[_NEQUIP_KERNEL_PACKAGES[modifier]][0]
        key += [modifier, f"{package}{version(package)}"]
    key += [f"torch{torch.__version__}", f"nequip{nequip.__version__}"]
    name = re.sub(r"[^\w.+-]", "_", "-".join(key))
    suffix = ".nequip.pth" if mode == "torchscript" else ".nequip.pt2"
    path = Path.home() / ".cache" / "ml_peg" / "nequip" / "compiled" / f"{name}{suffix}"
    if path.exists():
        return str(path)
    failed = path.with_name(f"{path.name}.failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    # The lock stops parallel workers compiling the same model (timing out in case a
    # killed worker leaves it held where flock is unsupported); the rename keeps
    # writes atomic where file locks are not shared, e.g. across nodes
    with FileLock(f"{path}.lock", timeout=3600):
        if failed.exists():
            raise RuntimeError(f"compiling failed before, so delete {failed} to retry")
        if not path.exists():
            # nequip-compile requires the .nequip.pt2 or .nequip.pth suffix
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=suffix, delete=False
            ) as file:
                partial = Path(file.name)
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "nequip.scripts.compile",
                        str(Path(model).absolute()),
                        str(partial),
                        "--mode",
                        mode,
                        "--device",
                        device,
                        "--target",
                        "ase",
                        *(["--modifiers", modifier] if modifier else []),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=path.parent,
                )
                partial.replace(path)
            except subprocess.CalledProcessError as error:
                # Kernel variants fall back to plain compilation, so don't retry them
                if modifier:
                    failed.write_text(error.stderr[-2000:])
                raise
            finally:
                partial.unlink(missing_ok=True)
    return str(path)


# TODO: nequip > 0.19.1 reuses one importer per package file, so repeated loads no
# longer leak; once that is required, drop this cache, with the JSON-encoded options and
# copies it needs
@cache
def _load_nequip_calculator(
    model: str, device: str, compile: bool, options: str
) -> Calculator:
    """
    Load a NequIP package once per process, compiled if requested.

    In nequip 0.19.1, every load keeps a ``torch.package`` importer and its open file
    alive until the process exits, so loading for each structure exhausts the
    file-descriptor limit. The first load also makes float64 torch's default dtype,
    which callers should restore.

    Parameters
    ----------
    model
        Path to the ``.nequip.zip`` package.
    device
        Device to load the model on.
    compile
        Whether to compile the model, with GPU kernels where available. Each variant
        that fails is skipped with a warning, ending with the eager model.
    options
        JSON-encoded keyword arguments for the ``NequIPCalculator`` constructors.

    Returns
    -------
    Calculator
        Loaded NequIP ASE calculator.
    """
    from nequip.integrations.ase import NequIPCalculator
    from nequip.model.modify_utils import get_all_modifiers
    from nequip.utils.dtype import torch_default_dtype
    import torch

    class MLPEGNequIPCalculator(NequIPCalculator):
        """
        ``NequIPCalculator`` with float64 inputs.

        NequIP keeps its data (positions, cell) in float64 and runs the model in the
        package's ``model_dtype``. ``NequIPCalculator`` builds the data with
        ``torch.get_default_dtype()``, relying on nequip's first load making float64
        the process-wide default, which ML-PEG restores after loading. Compiled models
        silently return infinite energies for float32 data.
        """

        def calculate(self, *args: Any, **kwargs: Any) -> None:
            """
            Run ``NequIPCalculator.calculate`` with float64 as torch's default dtype.

            Parameters
            ----------
            *args
                Positional arguments for ``NequIPCalculator.calculate``.
            **kwargs
                Keyword arguments for ``NequIPCalculator.calculate``.
            """
            with torch_default_dtype(torch.float64):
                super().calculate(*args, **kwargs)

    kwargs = json.loads(options)
    # Private eager loader: nequip 0.19.1 has no public no-compile constructor
    eager = MLPEGNequIPCalculator._from_saved_model(model, device=device, **kwargs)
    if not compile:
        return eager
    if torch.device(device).index is not None:
        warn(
            f"Using the eager NequIP model, as compiled models need device="
            f"{torch.device(device).type!r}; choose the GPU with "
            f"CUDA_VISIBLE_DEVICES rather than {device!r}",
            stacklevel=2,
        )
        return eager
    modifiers = set(get_all_modifiers(eager.model))
    for mode, modifier in _nequip_compile_variants(modifiers, device):
        try:
            if modifier:
                # GPU kernel libraries must be imported before loading the model
                import_module(_NEQUIP_KERNEL_PACKAGES[modifier])
            path = _compile_nequip_package(model, device, mode, modifier)
            return MLPEGNequIPCalculator.from_compiled_model(
                path, device=device, **kwargs
            )
        # Compiling needs a C++ compiler, and only changes speed
        except Exception as error:
            details = getattr(error, "stderr", None) or repr(error)
            warn(
                f"Skipping the {mode} NequIP model with "
                f"{modifier or 'no GPU kernels'}: {details[-2000:]}",
                stacklevel=2,
            )
    warn(
        "Using the eager NequIP model (set `compile: false` to skip compiling)",
        stacklevel=2,
    )
    return eager


@dataclasses.dataclass(kw_only=True)
class SumCalc:
    """
    Base class that tracks whether a model already includes dispersion corrections.

    ``add_d3_calculator`` only wraps calculators with an explicit TorchDFTD3
    correction when ``trained_on_dispersion`` is ``False``; otherwise the original
    calculator is returned untouched.
    """

    trained_on_dispersion: bool = False
    dispersion_kwargs: dict = dataclasses.field(default_factory=dict)

    def add_d3_calculator(self, calcs) -> Calculator | SumCalculator:
        """
        Add dispersion corrections to calculator(s).

        Parameters
        ----------
        calcs
            Calculator, or list of calculators, to add dispersion corrections to via a
            SumCalculator.

        Returns
        -------
        SumCalculator | Calculator
            Calculator(s) with dispersion corrections added, or the original calculator
            when the model is already trained with dispersion corrections.
        """
        if self.trained_on_dispersion:
            return calcs
        from ase import units
        from ase.calculators.mixing import SumCalculator
        import torch
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

        if not isinstance(calcs, list):
            calcs = [calcs]

        d3_calc = TorchDFTD3Calculator(
            device=self.dispersion_kwargs.get("device", "cpu"),
            damping=self.dispersion_kwargs.get("damping", "bj"),
            xc=self.dispersion_kwargs.get("xc", "pbe"),
            dtype=getattr(torch, self.dispersion_kwargs.get("dtype", "float32")),
            cutoff=self.dispersion_kwargs.get("cutoff", 40.0 * units.Bohr),
        )
        calcs.append(d3_calc)

        return SumCalculator(calcs)


@dataclasses.dataclass(kw_only=True)
class GenericASECalc(SumCalc, MlipxGenericASECalc):
    """Data class for generic ASE calculators."""

    default_dtype: str | None = None

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Level of precision to evaluate the model.
        **kwargs
            Any keyword arguments to pass to `get_calculator`.

        Returns
        -------
        Calculator
            Loaded ASE Calculator.
        """
        check_precision(precision)
        precision_map = {"low": "float32", "high": "float64"}
        kwargs["default_dtype"] = precision_map[precision]

        if self.default_dtype is not None:
            kwargs["default_dtype"] = self.default_dtype

        return MlipxGenericASECalc.get_calculator(self, **kwargs)


@dataclasses.dataclass(kw_only=True)
class DpaCalc(SumCalc):
    """Dataclass for DPA calculators provided by DeePMD-kit."""

    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the DeePMD ASE calculator.

        Parameters
        ----------
        precision
            Requested ML-PEG precision. DPA checkpoints are fixed at float32, so
            both ``low`` and ``high`` use the checkpoint's native precision.
        **kwargs
            Keyword arguments that override the registry calculator options.

        Returns
        -------
        Calculator
            Loaded DeePMD ASE calculator.
        """
        check_precision(precision)

        from deepmd.calculator import DP

        calculator_kwargs = {**self.kwargs, **kwargs}
        return DP(**calculator_kwargs)

    @property
    def available(self) -> bool:
        """
        Check whether the DeePMD ASE calculator is available.

        Returns
        -------
        bool
            Whether the calculator can be imported.
        """
        try:
            from deepmd.calculator import DP  # noqa: F401

            return True
        except ImportError:
            return False


@dataclasses.dataclass(kw_only=True)
class NequIPCalc(SumCalc):
    """Dataclass for NequIP framework (NequIP and Allegro) calculators."""

    device: Device | str = Device.AUTO
    compile: bool = True
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the NequIP ASE calculator.

        Parameters
        ----------
        precision
            Requested ML-PEG precision. Packaged NequIP models fix their model dtype
            (float32 for the OAM models), so both ``low`` and ``high`` use the package's
            native precision.
        **kwargs
            Keyword arguments that override the registry calculator options. ``model``
            is a ``.nequip.zip`` path, or an ``https://`` URL downloaded once. Other
            options are passed to both ``NequIPCalculator`` constructors, but those only
            the eager loader takes (e.g. ``model_name``) do not reach compiled models.

        Returns
        -------
        Calculator
            Loaded NequIP ASE calculator.
        """
        check_precision(precision)
        # Fail before downloading the package if nequip is not installed
        from nequip.utils.dtype import torch_default_dtype
        import torch

        calc_kwargs = {
            "chemical_species_to_atom_type_map": True,
            **self.kwargs,
            **kwargs,
        }
        model = calc_kwargs.pop("model")
        if model.startswith("https://"):
            model = _download_nequip_package(model)
        device = Device.resolve_auto() if self.device == Device.AUTO else self.device
        options = json.dumps(calc_kwargs, sort_keys=True)
        # nequip's first load makes float64 the default dtype (and seeds the RNGs and
        # sets other torch flags process-wide); keep the caller's dtype
        with torch_default_dtype(torch.get_default_dtype()):
            calc = _load_nequip_calculator(model, device, self.compile, options)
        # Copies share the cached model, but not its results or atoms
        return copy.copy(calc)


@dataclasses.dataclass(kw_only=True)
class MatterSimCalc(GenericASECalc):
    """Dataclass for MatterSim calculator."""

    @staticmethod
    def _patch_mattersim_setstate() -> None:
        """Rebuild the model from mattersim's own saved model_args on copy()."""
        from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
        from mattersim.forcefield.potential import MatterSimCalculator, Potential
        import torch

        def __setstate__(self, state):  # noqa: N807
            """
            Restore from copy/pickle by rebuilding at the saved architecture.

            Parameters
            ----------
            self
                The ``MatterSimCalculator`` instance being restored.
            state
                State dict produced by ``__getstate__``, containing the saved
                model weights, architecture (``model_args``) and name.
            """
            model_state_dict = state.pop("_model_state_dict")
            model_args = state.pop("_model_args")
            model_name = state.pop("_model_name")
            self.__dict__.update(state)
            model = M3Gnet(device=self.device, **model_args).to(self.device)
            model.load_state_dict(model_state_dict)
            model.eval()
            self.potential = Potential(
                model,
                device=self.device,
                model_name=model_name,
                load_training_state=False,
            )
            if self.dtype == torch.float64:
                self.potential.model.double()

        MatterSimCalculator.__setstate__ = __setstate__

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Level of precision to evaluate the model.
        **kwargs
            Any keyword arguments to pass to `get_calculator`.

        Returns
        -------
        Calculator
            Loaded ASE Calculator.
        """
        check_precision(precision)
        precision_map = {"low": "float32", "high": "float64"}
        kwargs["dtype"] = precision_map[precision]

        if self.default_dtype is not None:
            kwargs["dtype"] = self.default_dtype

        self._patch_mattersim_setstate()

        return MlipxGenericASECalc.get_calculator(self, **kwargs)


@dataclasses.dataclass(kw_only=True)
class VivaceCalc(SumCalc):
    """Dataclass for Vivace calculator."""

    device: Device | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Unused precision argument, kept for the common model API.
        **kwargs
            Keyword arguments passed to the Vivace calculator.

        Returns
        -------
        Calculator
            Loaded ASE calculator.
        """
        from simpoly.vivace.calculator import MLFFCalculator

        check_precision(precision)

        kwargs.update(self.kwargs)
        calc = MLFFCalculator(**kwargs)

        # Vivace sets dtype from checkpoint metadata inside MLFFCalculator.
        # Leave precision/overwrite_dtype untouched unless SimPoly exposes it.
        device = Device.resolve_auto() if self.device == Device.AUTO else self.device
        if device is not None:
            calc.device = device
            calc.model = calc.model.to(device=device)

        return calc


# https://github.com/orbital-materials/orb-models
@dataclasses.dataclass(kw_only=True)
class OrbCalc(SumCalc):
    """Dataclass for Orb calculator."""

    name: str
    device: Device | None = None
    default_dtype: str | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Level of precision to evaluate the model.
        **kwargs
            Any keyword arguments to pass to `get_calculator`.

        Returns
        -------
        Calculator
            Loaded ASE Orb Calculator.
        """
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        torch._dynamo.disable()
        import os

        check_precision(precision)
        os.environ["TORCH_DISABLE_MODULE_HIERARCHY_TRACKING"] = "1"

        method = getattr(pretrained, self.name)

        precision_map = {"low": "float32-high", "high": "float64"}
        dtype = precision_map[precision]

        if self.default_dtype is not None:
            dtype = self.default_dtype

        if self.device is None:
            orbff, atoms_adapter = method(precision=dtype, **self.kwargs)
            calc = ORBCalculator(orbff, atoms_adapter=atoms_adapter, **self.kwargs)
        elif self.device == Device.AUTO:
            orbff = method(
                device=Device.resolve_auto(),
                precision=dtype,
                **self.kwargs,
            )
            calc = ORBCalculator(orbff, device=Device.resolve_auto(), **self.kwargs)
        else:
            orbff, atoms_adapter = method(
                device=self.device, precision=dtype, **self.kwargs
            )
            calc = ORBCalculator(
                orbff, atoms_adapter=atoms_adapter, device=self.device, **self.kwargs
            )

        return calc

    @property
    def available(self) -> bool:
        """
        Check whether the calculator module is available.

        Returns
        -------
        bool
            Whether the calculator can be loaded.
        """
        try:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator

            return True
        except ImportError:
            return False


@dataclasses.dataclass(kw_only=True)
class FairChemCalc(SumCalc):
    """Dataclass for fairchem (UMA) calculator."""

    model_name: str
    task_name: str
    device: Device | str = "cpu"
    default_dtype: str | None = None
    overrides: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Level of precision to evaluate the model.
        **kwargs
            Any additional keyword arguments.

        Returns
        -------
        Calculator
            Loaded ASE fairchem Calculator.
        """
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        from fairchem.core.units.mlip_unit.api.inference import (
            inference_settings_default,
        )

        # fairchem defaults to float32; map the requested precision to the base
        # dtype so precision="high" runs in float64. A configured default_dtype
        # overrides this.
        check_precision(precision)
        precision_map = {"low": "float32", "high": "float64"}
        dtype = self.default_dtype or precision_map[precision]
        inference_settings = dataclasses.replace(
            inference_settings_default(), base_precision_dtype=dtype
        )

        predictor = pretrained_mlip.get_predict_unit(
            self.model_name,
            device=self.device,
            overrides=self.overrides,
            inference_settings=inference_settings,
        )
        return FAIRChemCalculator(predictor, task_name=self.task_name)

    @property
    def available(self) -> bool:
        """
        Check whether the calculator module is available.

        Returns
        -------
        bool
            Whether the calculator can be loaded.
        """
        try:
            from fairchem.core import pretrained_mlip

            return self.model_name in pretrained_mlip._MODEL_CKPTS.checkpoints
        except Exception:
            return False


@dataclasses.dataclass(kw_only=True)
class MockCalc(SumCalc):
    """Dataclass for mock calculator."""

    model_name: str = "mock"
    trained_on_dispersion: bool = True

    def get_calculator(self, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        **kwargs
            Any additional keyword arguments passed to `get_calculator`.

        Returns
        -------
        Calculator
            Loaded mock ASE Calculator.
        """
        from ml_peg.models.mock import MockCalculator

        return MockCalculator()


@dataclasses.dataclass(kw_only=True)
class UPETCalc(GenericASECalc):
    """Dataclass for upet (PET-MAD / PET-OAM) calculator."""

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Level of precision to evaluate the model.
        **kwargs
            Any keyword arguments to pass to `get_calculator`.

        Returns
        -------
        Calculator
            Loaded upet ASE calculator.
        """
        check_precision(precision)
        precision_map = {"low": "float32", "high": "float64"}
        kwargs["dtype"] = precision_map[precision]

        if self.default_dtype is not None:
            kwargs["dtype"] = self.default_dtype

        _patch_metatomic_nvalchemi_max_neighbors()
        return MlipxGenericASECalc.get_calculator(self, **kwargs)


@dataclasses.dataclass(kw_only=True)
class SevenNetCalc(SumCalc):
    """Dataclass for SevenNet calculator."""

    device: Device | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Currently unused. Once supported, this will correspond to the level of
            precision to evaluate the model.
        **kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        Calculator
            Loaded SevenNet ASE calculator.
        """
        from sevenn.sevennet_calculator import SevenNetCalculator

        check_precision(precision)
        device = Device.resolve_auto() if self.device == Device.AUTO else self.device
        device_str = device.value if isinstance(device, Device) else (device or "cpu")
        return SevenNetCalculator(device=device_str, **self.kwargs)


@dataclasses.dataclass(kw_only=True)
class GraceCalc(GenericASECalc):
    """Dataclass for GRACE calculator."""

    device: Device | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Level of precision to evaluate the model.
        **kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        Calculator
            Loaded GRACE ASE calculator.
        """
        from tensorpotential.calculator.foundation_models import MODELS_NAME_LIST

        check_precision(precision)
        precision_map = {"low": "", "high": "-fp64"}
        suffix = precision_map[precision]

        if self.default_dtype is not None:
            suffix = self.default_dtype

        model_name = f"{self.kwargs['model']}{suffix}"
        if model_name in MODELS_NAME_LIST:
            self.kwargs["model"] = model_name
        else:
            warn(
                "Unable to find model with requested precision, using default",
                stacklevel=2,
            )

        return MlipxGenericASECalc.get_calculator(self, **kwargs)
