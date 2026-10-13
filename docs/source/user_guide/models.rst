======
Models
======

The machine-learned interatomic potentials (MLIPs) known to ML-PEG are listed
below, with the configuration defined in ``ml_peg/models/models.yml``. Models
marked *(not currently enabled)* are commented out in ``models.yml`` and are not
run in the current benchmarks; they are listed here for completeness.

.. note::

   Citations and longer descriptions are still to be added for each model.

MACE
====

mace-mp-0a
----------

.. code-block:: yaml

   mace-mp-0a:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "medium"

mace-mp-0b3
-----------

.. code-block:: yaml

   mace-mp-0b3:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "medium-0b3"

mace-mpa-0
----------

.. code-block:: yaml

   mace-mpa-0:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "medium-mpa-0"

mace-omat-0
-----------

.. code-block:: yaml

   mace-omat-0:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "medium-omat-0"

mace-matpes-r2scan
------------------

.. code-block:: yaml

   mace-matpes-r2scan:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: r2SCAN
     kwargs:
       model: "mace-matpes-r2scan-0"
       head: "matpes_pbe"
     dispersion_kwargs:
       xc: r2scan
       label: D4

mace-mh-1-omat
--------------

.. code-block:: yaml

   mace-mh-1-omat:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "mh-1"
       head: omat_pbe


MACE-OFF23(L)
-------------

.. code-block:: yaml

   MACE-OFF23(L):
     module: mace.calculators
     class_name: mace_off
     device: "auto"
     trained_on_dispersion: true
     kwargs:
       model: large


MACE-OFF24(M)
-------------

.. code-block:: yaml

   MACE-OFF24(M):
     module: mace.calculators
     class_name: mace_off
     device: "auto"
     trained_on_dispersion: true
     kwargs:
       model: "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off24/MACE-OFF24_medium.model"


mace-omol
---------

.. code-block:: yaml

   mace-omol:
     module: mace.calculators
     class_name: mace_omol
     device: "auto"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model: "extra_large"

mace-mh-1-omol
--------------

.. code-block:: yaml

   mace-mh-1-omol:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model: "mh-1"
       head: omol

mace-polar-1-s
--------------

.. code-block:: yaml

   mace-polar-1-s:
     module: mace.calculators
     class_name: mace_polar
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V
     kwargs:
       model: "polar-1-s"

mace-polar-1-m
--------------

.. code-block:: yaml

   mace-polar-1-m:
     module: mace.calculators
     class_name: mace_polar
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V
     kwargs:
       model: "polar-1-m"

mace-polar-1-l
--------------

.. code-block:: yaml

   mace-polar-1-l:
     module: mace.calculators
     class_name: mace_polar
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V
     kwargs:
       model: "polar-1-l"


DPA
===

Install these fixed-float32 DeePMD checkpoints with ``uv sync --extra dpa``.
DPA-3.3-1M uses OpenLAM-v1 with the OMat24 head and the DPA4 variants use OMat24.
DPA-3.3-1M is CC-BY-4.0, while DPA4 is CC-BY-NC-4.0 and restricted to
non-commercial use.

dpa-3p3-1M-omat
----------------

.. code-block:: yaml

   dpa-3p3-1M-omat:
     module: deepmd.calculator
     class_name: DP
     datasets: [OpenLAM-v1, OMAT, MPtrj, OC20, OC22, ODAC23, SPICE2]
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: DPA-3.3-1M
       head: Omat24

dpa-4-nano-omat
---------------

.. code-block:: yaml

   dpa-4-nano-omat:
     module: deepmd.calculator
     class_name: DP
     datasets: [OMAT]
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: DPA4-Nano-OMat24-v20260805

dpa-4-neo-omat
--------------

.. code-block:: yaml

   dpa-4-neo-omat:
     module: deepmd.calculator
     class_name: DP
     datasets: [OMAT]
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: DPA4-Neo-OMat24-v20260805

dpa-4-plus-omat
---------------

.. code-block:: yaml

   dpa-4-plus-omat:
     module: deepmd.calculator
     class_name: DP
     datasets: [OMAT]
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: DPA4-Plus-OMat24-v20260805

Orb
===

orb-v3-consv-inf-omat
---------------------

.. code-block:: yaml

   orb-v3-consv-inf-omat:
     module: orb_models.inference.calculator
     class_name: OrbCalc
     device: "cpu"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       name: "orb_v3_conservative_inf_omat"

orb-v3-direct-inf-omat
----------------------

.. code-block:: yaml

   orb-v3-direct-inf-omat:
     module: orb_models.inference.calculator
     class_name: OrbCalc
     device: "cpu"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       name: "orb_v3_direct_inf_omat"

orb-v3-consv-omol
-----------------

.. code-block:: yaml

   orb-v3-consv-omol:
     module: orb_models.forcefield.inference.calculator
     class_name: OrbCalc
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: PBE
     kwargs:
       name: "orb_v3_conservative_omol"


PET
===

pet-mad
-------

.. code-block:: yaml

  pet-mad:
    module: upet.calculator
    class_name: UPETCalculator
    device: "cpu"
    trained_on_dispersion: false
    level_of_theory: PBEsol
    kwargs:
      model: "pet-mad-s"
      version: "1.0.2"
    dispersion_kwargs:
      xc: pbesol

pet-mad-s-1p5
-------------

.. code-block:: yaml

  pet-mad-s-1p5:
    module: upet.calculator
    class_name: UPETCalculator
    device: "cpu"
    trained_on_dispersion: false
    level_of_theory: r2SCAN
    kwargs:
      model: "pet-mad-s"
      version: "1.5.0"
    dispersion_kwargs:
      xc: r2scan

pet-oam-xl
----------

.. code-block:: yaml

  pet-oam-xl:
    module: upet.calculator
    class_name: UPETCalculator
    device: "cpu"
    trained_on_dispersion: False
    level_of_theory: PBE
    kwargs:
      model: "pet-oam-xl"
      version: "1.0.0"


UMA (FairChem)
==============

uma-s-1p1-omat
--------------

.. code-block:: yaml

   uma-s-1p1-omat:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     level_of_theory: PBE
     trained_on_dispersion: false
     kwargs:
       model_name: "uma-s-1p1"
       task_name: "omat"

uma-m-1p1-omat
--------------

.. code-block:: yaml

   uma-m-1p1-omat:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     level_of_theory: PBE
     trained_on_dispersion: false
     kwargs:
       model_name: "uma-m-1p1"
       task_name: "omat"

uma-s-1p1-omol
--------------

.. code-block:: yaml

   uma-s-1p1-omol:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model_name: "uma-s-1p1"
       task_name: "omol"

uma-s-1p2-omol
--------------

.. code-block:: yaml

   uma-s-1p2-omol:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model_name: "uma-s-1p2"
       task_name: "omol"

uma-m-1p1-omol
--------------

.. code-block:: yaml

   uma-m-1p1-omol:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model_name: "uma-m-1p1"
       task_name: "omol"


MatterSim
=========

mattersim-5M
------------

.. code-block:: yaml

   mattersim-5M:
     module: mattersim.forcefield
     class_name: MatterSimCalculator
     device: "cpu"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       load_path: "mattersim-v1.0.0-5m"

mattersim-1M
------------

.. code-block:: yaml

   mattersim-1M:
     module: mattersim.forcefield
     class_name: MatterSimCalculator
     device: "cpu"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       load_path: "mattersim-v1.0.0-1m"


GRACE
=====

GRACE-2L-OMAT
-------------

.. code-block:: yaml

  grace-2l-omat:
    module: tensorpotential.calculator
    class_name: grace_fm
    device: "cpu"
    trained_on_dispersion: false
    kwargs:
      model: GRACE-2L-OMAT


GRACE-2L-SMAX-OMAT-medium
-------------------------
.. code-block:: yaml

  grace-2l-smax-omat-medium:
    module: tensorpotential.calculator
    class_name: grace_fm
    device: "cpu"
    trained_on_dispersion: false
    kwargs:
      model: GRACE-2L-SMAX-OMAT-medium


SevenNet
========

sevennet-omni-mpa
-----------------

.. code-block:: yaml

  sevennet-omni-mpa:
    module: sevenn.sevennet_calculator
    class_name: SevenNetCalculator
    device: "cpu"
    trained_on_dispersion: false
    level_of_theory: PBE
    kwargs:
      model: "7net-omni"
      modal: "mpa"


NequIP
======

Install these NequIP and Allegro foundation models with
``uv sync --extra nequip`` (add ``--extra d3`` for benchmarks with a D3
correction); the ``nequip`` extra cannot be combined with ``mace``. Each entry
pins a Zenodo file by record and sha256, downloaded once to
``~/.cache/ml_peg/nequip``. On first use on each type of CPU or GPU, the package
is compiled with ``nequip-compile`` (which needs a C++ compiler) and cached under
``~/.cache/ml_peg/nequip/compiled``. On CUDA GPUs, NequIP models are compiled with
OpenEquivariance kernels and Allegro models with cuEquivariance kernels, which the
``nequip`` extra installs on Linux (its CUDA 12 build of the cuEquivariance
kernels shares files with the CUDA 13 build in the ``vivace`` extra, so install
those extras in separate environments). NequIP models fall back to cuEquivariance,
then each model to AOTInductor without GPU kernels. Each GPU kernel build is
checked against the eager model before use. If compilation fails, the eager model
is used with a warning, and a top-level ``compile: false`` (next to ``device``)
always uses it. The packages are float32 models, so both precisions use their
native precision. The models are pre-trained on OMat24, fine-tuned on sAlex and
MPtrj, and are CC-BY-4.0.

nequip-oam-s-0p2
----------------

.. code-block:: yaml

   nequip-oam-s-0p2:
     module: nequip.integrations.ase
     class_name: NequIPCalculator
     datasets: [MPtrj, sAlex, OMAT]
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "https://zenodo.org/api/records/22349749/files/NequIP-OAM-S-0.2.nequip.zip/content"
       sha256: "5a3357b70c67cf497c718b785b2744f3405db14619e56f42a95cc6d372b00fea"

nequip-oam-m-0p2
----------------

.. code-block:: yaml

   nequip-oam-m-0p2:
     module: nequip.integrations.ase
     class_name: NequIPCalculator
     datasets: [MPtrj, sAlex, OMAT]
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "https://zenodo.org/api/records/22349749/files/NequIP-OAM-M-0.2.nequip.zip/content"
       sha256: "5acc555985feda2857c91fa7095bb3bb3e1cdd67f3bed0d8668df812b87da409"

nequip-oam-l-0p2
----------------

.. code-block:: yaml

   nequip-oam-l-0p2:
     module: nequip.integrations.ase
     class_name: NequIPCalculator
     datasets: [MPtrj, sAlex, OMAT]
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "https://zenodo.org/api/records/22349749/files/NequIP-OAM-L-0.2.nequip.zip/content"
       sha256: "ab0f1a8248e86fa6b0fb36899f072fc2e24966a9649c72d1b1fc2d5b0ebd3203"

nequip-oam-xl-0p2
-----------------

.. code-block:: yaml

   nequip-oam-xl-0p2:
     module: nequip.integrations.ase
     class_name: NequIPCalculator
     datasets: [MPtrj, sAlex, OMAT]
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "https://zenodo.org/api/records/22349749/files/NequIP-OAM-XL-0.2.nequip.zip/content"
       sha256: "b064eacf17d9c2625ce2f4e432517b106044ea2dee14142273622b75c93a01d6"

allegro-oam-s-0p2
-----------------

.. code-block:: yaml

   allegro-oam-s-0p2:
     module: nequip.integrations.ase
     class_name: NequIPCalculator
     datasets: [MPtrj, sAlex, OMAT]
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "https://zenodo.org/api/records/22349749/files/Allegro-OAM-S-0.2.nequip.zip/content"
       sha256: "6ce4bb1db4076b4176cb63bae2eaae091ec499660227530a0071153bfdc62a4d"

allegro-oam-m-0p2
-----------------

.. code-block:: yaml

   allegro-oam-m-0p2:
     module: nequip.integrations.ase
     class_name: NequIPCalculator
     datasets: [MPtrj, sAlex, OMAT]
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "https://zenodo.org/api/records/22349749/files/Allegro-OAM-M-0.2.nequip.zip/content"
       sha256: "1e4a94a75ead37625eadca53efbbc5e82396deb145c44203bdb9e91aaf0646e4"

allegro-oam-l-0p2
-----------------

.. code-block:: yaml

   allegro-oam-l-0p2:
     module: nequip.integrations.ase
     class_name: NequIPCalculator
     datasets: [MPtrj, sAlex, OMAT]
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "https://zenodo.org/api/records/22349749/files/Allegro-OAM-L-0.2.nequip.zip/content"
       sha256: "0396734c4aa6d9782a9122acb4e7220f693ab7a327e2c457490ab01015ed9af1"
