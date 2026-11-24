import random
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from ..developer import NotAnElementError, obj
from ..generic import NexusUserWarning
from ..periodic_table import Elements
from ..physical_system import generate_physical_system
from ..pseudoset import PseudoSet
from ..pwscf_input import (
    AtomCoordConstraint,
    AtomicForces,
    AtomicPositions,
    AtomicSpecies,
    AtomicVelocities,
    BennettProjConstraint,
    CardBase,
    Cell,
    CellParameters,
    Constraints,
    Control,
    DistanceConstraint,
    Electrons,
    Fcp,
    Hubbard,
    HubbardAlpha,
    HubbardJ,
    HubbardJ0,
    HubbardU,
    HubbardV,
    Ions,
    KPoints,
    NamelistBase,
    Occupations,
    PlanarAngleConstraint,
    PotentialWallConstraint,
    PwscfInput,
    Rism,
    Solvents,
    System,
    TorsionalAngleConstraint,
    TypeCoordConstraint,
    generate_pwscf_input,
)
from ..pwscf_input_defs import SystemDefinitions
from ..structure import Structure, kmesh, read_structure
from ..unit_converter import convert
from . import TEST_DIR, NexusTestOrder, isolate_nexus_core, register_pseudo_files

pytestmark = pytest.mark.order(NexusTestOrder.PWSCF_INPUT)

TEST_FILES = {
    "Cr_noncolin.in": TEST_DIR / "test_pwscf_input_files/Cr_noncolin.in",
    "Fe_start_ns_eig.in": TEST_DIR / "test_pwscf_input_files/Fe_start_ns_eig.in",
    "LiI_vc_relax.in": TEST_DIR / "test_pwscf_input_files/LiI_vc_relax.in",
    "Ni_surface.in": TEST_DIR / "test_pwscf_input_files/Ni_surface.in",
    "nexus_argon_scf.in": TEST_DIR / "test_pwscf_input_files/nexus_argon_scf.in",
    "nexus_h2_relax.in": TEST_DIR / "test_pwscf_input_files/nexus_h2_relax.in",
    "nexus_argon_bands.in": TEST_DIR / "test_pwscf_input_files/nexus_argon_bands.in",
    "nexus_hubbard_e2_e3.in": TEST_DIR / "test_pwscf_input_files/nexus_hubbard_e2_e3.in",
    "TiO2_band_structure.in": TEST_DIR / "test_pwscf_input_files/TiO2_band_structure.in",
    "TiO2_relax_freeze.in": TEST_DIR / "test_pwscf_input_files/TiO2_relax_freeze.in",
    "VO2_M1_afm.in": TEST_DIR / "test_pwscf_input_files/VO2_M1_afm.in",
    "VO2_M1_afm.xsf": TEST_DIR / "test_pwscf_input_files/VO2_M1_afm.xsf",
    "WSe2_band_structure.in": TEST_DIR / "test_pwscf_input_files/WSe2_band_structure.in",
    "co2_constrained.in": TEST_DIR / "test_pwscf_input_files/co2_constrained.in",
    "furfural_pd_surface.in": TEST_DIR / "test_pwscf_input_files/furfural_pd_surface.in",
    "h2o_nacl_rism.in": TEST_DIR / "test_pwscf_input_files/h2o_nacl_rism.in",
    "nexus_water_forces.in": TEST_DIR / "test_pwscf_input_files/nexus_water_forces.in",
    "nexus_oxygen_occupations.in": TEST_DIR / "test_pwscf_input_files/nexus_oxygen_occupations.in",
    "nexus_cd_fractions.in": TEST_DIR / "test_pwscf_input_files/nexus_cd_fractions.in",
    "README": TEST_DIR / "test_pwscf_input_files/README",
    "pseudos": (TEST_DIR / "../examples/qmcpack/pseudopotentials/").resolve(),
}

# These are input-only counterparts to supplemental PwscfAnalyzer runs.  Keep
# them in the analyzer fixture tree: the analyzer tests need the complete run
# products, while the input tests exercise the same representative inputs.
ANALYZER_SUPPLEMENTAL_DIR = (
    TEST_DIR / "test_pwscf_analyzer_files/qe_7_0/supplemental"
    )
for input_name in (
    'cbn_scf',
    'cbn_relax',
    'cbn_vc_relax',
    'cbn_smearing',
    'cbn_spin',
    'cbn_crystal_kpoints',
    'md_iprint',
    'vc_md_iprint',
    'scf_crystal_kpoints',
    'scf_no_symmetry',
    ):
    TEST_FILES[f'supplemental_{input_name}.in'] = (
        ANALYZER_SUPPLEMENTAL_DIR / input_name / 'pwscf.in'
        )

for file in TEST_FILES.values():
    assert(file.exists()), f"Test file not found! {file}"

@pytest.fixture(autouse=True)
def cleanup():
    PseudoSet.pseudo_files.update(
        {
            "C.BFD.upf": TEST_FILES["pseudos"] / "C.BFD.upf",
            "H.BFD.upf": TEST_FILES["pseudos"] / "H.BFD.upf",
            "O.BFD.upf": TEST_FILES["pseudos"] / "O.BFD.upf",
        }
    )

    yield

    del PseudoSet.pseudo_files["C.BFD.upf"]
    del PseudoSet.pseudo_files["H.BFD.upf"]
    del PseudoSet.pseudo_files["O.BFD.upf"]

@pytest.fixture
def phys_sys():
    return generate_physical_system(
        units = "A",
        axes = np.eye(3) * 2,
        elem = ["C", "H", "O"],
        pos = np.arange(9, dtype=float).reshape((3, 3)),
        kgrid = (2, 2, 2),
        kshift = (0, 0, 0),
        C = 4,
        H = 1,
        O = 6,
    )

@pytest.fixture
def frozen_phys_sys(phys_sys):
    ps = phys_sys.copy()
    # Note that this array is the inverse of what QE expects for force multipliers
    ps.structure.frozen = np.array(
        [
            [False, True,  True],
            [False, False, True],
            [True,  False, False],
        ],
        dtype=bool,
    )
    return ps

def assert_namelist_equals[T: NamelistBase](calc_nml: T, ref_nml: T) -> None:
    assert(type(calc_nml) is type(ref_nml))
    assert(isinstance(calc_nml, NamelistBase))
    assert(calc_nml.params.keys() == ref_nml.params.keys())
    for key, value_1 in calc_nml.params.items():
        value_2 = ref_nml.params[key]
        if isinstance(value_1, np.ndarray):
            np.testing.assert_allclose(value_1, value_2, err_msg=key)
        else:
            assert(value_1 == value_2), key

def assert_card_equals[T: CardBase](calc_card: T, ref_card: T) -> None:
    assert(type(calc_card) is type(ref_card))
    assert(isinstance(calc_card, CardBase))
    if isinstance(calc_card, AtomicSpecies) and isinstance(ref_card, AtomicSpecies):
        assert(calc_card.atoms == ref_card.atoms)
        assert(calc_card.masses == ref_card.masses)
        assert(calc_card.pseudos == ref_card.pseudos)

    elif isinstance(calc_card, AtomicPositions) and isinstance(ref_card, AtomicPositions):
        assert(calc_card.specifier == ref_card.specifier)
        assert(calc_card.atoms == ref_card.atoms)
        np.testing.assert_allclose(calc_card.positions, ref_card.positions, rtol=1e-6)
        assert(type(calc_card.force_multipliers) is type(ref_card.force_multipliers))
        if calc_card.force_multipliers is not None:
            np.testing.assert_allclose(calc_card.force_multipliers, ref_card.force_multipliers)

    elif isinstance(calc_card, KPoints) and isinstance(ref_card, KPoints):
        assert(calc_card.specifier == ref_card.specifier)
        if calc_card.specifier == "automatic":
            np.testing.assert_array_equal(calc_card.grid, ref_card.grid)
            np.testing.assert_array_equal(calc_card.shift, ref_card.shift)
        elif calc_card.specifier != "gamma":
            assert(calc_card.nkpoints == ref_card.nkpoints)
            np.testing.assert_allclose(calc_card.kpoints, ref_card.kpoints, rtol=1e-6)
            assert(type(calc_card.weights) is type(ref_card.weights))
            if calc_card.weights is not None:
                np.testing.assert_allclose(calc_card.weights, ref_card.weights)

    elif isinstance(calc_card, CellParameters) and isinstance(ref_card, CellParameters):
        assert(calc_card.specifier == ref_card.specifier)
        np.testing.assert_allclose(calc_card.axes, ref_card.axes)

    elif isinstance(calc_card, Hubbard) and isinstance(ref_card, Hubbard):
        assert(calc_card.specifier == ref_card.specifier)
        assert(len(calc_card.params) == len(ref_card.params))
        assert(set(calc_card.params) == set(ref_card.params))

    elif isinstance(calc_card, Occupations) and isinstance(ref_card, Occupations):
        np.testing.assert_allclose(calc_card.occupations, ref_card.occupations)
        assert(type(calc_card.minority_spin_occs) is type(ref_card.minority_spin_occs))
        if calc_card.minority_spin_occs is not None:
            np.testing.assert_allclose(calc_card.minority_spin_occs, ref_card.minority_spin_occs)

    elif isinstance(calc_card, AtomicForces) and isinstance(ref_card, AtomicForces):
        assert(calc_card.atoms == ref_card.atoms)
        np.testing.assert_allclose(calc_card.forces, ref_card.forces)

    elif isinstance(calc_card, AtomicVelocities) and isinstance(ref_card, AtomicVelocities):
        assert(calc_card.atoms == ref_card.atoms)
        np.testing.assert_allclose(calc_card.velocities, ref_card.velocities)

    elif isinstance(calc_card, Constraints) and isinstance(ref_card, Constraints):
        assert(calc_card.tolerance == ref_card.tolerance)
        # The classes make equality checks a bit difficult, so we check instead
        # that their output is the same.
        con1 = set(calc_card.write().splitlines())
        con2 = set(ref_card.write().splitlines())
        assert(con1 == con2)

    elif isinstance(calc_card, Solvents) and isinstance(ref_card, Solvents):
        assert(calc_card.specifier == ref_card.specifier)
        assert(calc_card.labels == ref_card.labels)
        assert(type(calc_card.densities) is type(ref_card.densities))
        assert(type(calc_card.densities_left) is type(ref_card.densities_left))
        assert(type(calc_card.densities_right) is type(ref_card.densities_right))
        if calc_card.densities is not None:
            np.testing.assert_allclose(calc_card.densities, ref_card.densities)
        else:
            np.testing.assert_allclose(calc_card.densities_left, ref_card.densities_left)
            np.testing.assert_allclose(calc_card.densities_right, ref_card.densities_right)

    else:
        msg = f"Unsupported card type {type(calc_card).__name__}"
        raise TypeError(msg)

def assert_pwscf_input_equals(calc_pwi: PwscfInput, ref_pwi: PwscfInput) -> None:
    for nml in PwscfInput.namelist_types:
        assert(hasattr(calc_pwi, nml))
        assert(hasattr(ref_pwi, nml))
        calc_nml = getattr(calc_pwi, nml)
        ref_nml = getattr(ref_pwi, nml)
        assert(type(calc_nml) is type(ref_nml))
        if not (calc_nml is None and ref_nml is None):
            assert_namelist_equals(calc_nml, ref_nml)

    for card in PwscfInput.card_types:
        assert(hasattr(calc_pwi, card))
        assert(hasattr(ref_pwi, card))
        calc_card = getattr(calc_pwi, card)
        ref_card = getattr(ref_pwi, card)
        assert(type(calc_card) is type(ref_card))
        if not (calc_card is None and ref_card is None):
            assert_card_equals(calc_card, ref_card)


def test_join_array_params():
    ref = {
        'starting_ns_eigenvalue': {
            (1, 2, 1): "a",
            (2, 2, 1): "b",
            (3, 2, 1): "c",
            (4, 2, 1): "d",
            (5, 2, 1): "e",
        },
    }
    params = {
        'starting_ns_eigenvalue(1,2,1)' : "a",
        'starting_ns_eigenvalue(2,2,1)' : "b",
        'starting_ns_eigenvalue(3,2,1)' : "c",
        'starting_ns_eigenvalue(4,2,1)' : "d",
        'starting_ns_eigenvalue(5,2,1)' : "e",
    }
    normalized = System.join_array_params(params)
    assert(normalized == ref)

    ref = {
        'celldm': {
            1: "a",
            2: "b",
            4: "d",
            5: "e",
        },
    }
    params = {
        "celldm(1)" : "a",
        "celldm(2)" : "b",
        "celldm(4)" : "d",
        "celldm(5)" : "e",
    }
    normalized = System.join_array_params(params)
    assert(normalized == ref)

    with pytest.raises(
        ValueError,
        match="Multiple formats of parameter 'celldm' found!",
    ):
        normalized = System.join_array_params(
            {"celldm(1)": "a", "celldm": ["a", "b"]}
        )


def test_check_param_shape():
    celldm = {
        1: "a",
        2: "b",
        4: "d",
        5: "e",
    }
    valid, msg = NamelistBase._check_param_shape(
        name = "celldm",
        value = celldm,
        shape = SystemDefinitions.celldm.shape,
        phys_sys = None,
    )
    assert(valid), msg

    celldm = {
        7: "f",
    }
    valid, msg = NamelistBase._check_param_shape(
        name = "celldm",
        value = celldm,
        shape = SystemDefinitions.celldm.shape,
        phys_sys = None,
    )
    assert(not valid)
    assert("celldm" in msg)
    assert("7" in msg)
    assert("(1, 6)" in msg)


    struct_file = TEST_FILES['VO2_M1_afm.xsf']
    phys_sys = generate_physical_system(structure=struct_file)
    hub_u = {
        1: "a",
        2: "b",
    }
    valid, msg = NamelistBase._check_param_shape(
        name = "hubbard_u",
        value = hub_u,
        shape = SystemDefinitions.hubbard_u.shape,
        phys_sys = phys_sys,
    )
    assert(valid), msg

    hub_u = {
        1: "a",
        3: "c",
    }
    valid, msg = NamelistBase._check_param_shape(
        name = "hubbard_u",
        value = hub_u,
        shape = SystemDefinitions.hubbard_u.shape,
        phys_sys = phys_sys,
    )
    assert(not valid)
    assert("hubbard_u" in msg)
    assert("3" in msg)
    assert("(1, 2)" in msg)


    hub_v = {
        ( 1, 2, 1): "a",
        ( 2, 2, 2): "b",
        ( 6, 2, 3): "c",
        (12, 2, 4): "d",
    }
    valid, msg = NamelistBase._check_param_shape(
        name = "hubbard_v",
        value = hub_v,
        shape = SystemDefinitions.hubbard_v.shape,
        phys_sys = phys_sys,
    )
    assert(valid), msg

    hub_v = {
        (13, 2, 1): "a",
    }
    valid, msg = NamelistBase._check_param_shape(
        name = "hubbard_v",
        value = hub_v,
        shape = SystemDefinitions.hubbard_v.shape,
        phys_sys = phys_sys,
    )
    assert(not valid)
    assert("hubbard_v" in msg)
    assert("(13, 2, 1)" in msg)
    assert("(12, '27*natx', 4)" in msg)


def test_check_param_version():
    valid, msg = NamelistBase._check_param_version(
        name = "hubbard_u",
        current = "7.0.0",
        added = None,
        removed = "7.1.0",
    )
    assert(valid), msg

    valid, msg = NamelistBase._check_param_version(
        name = "exx_maxstep",
        current = "7.5.0",
        added = "7.2.0",
        removed = None,
    )
    assert(valid), msg


    added = "7.1.0"
    removed = "7.5.0"
    current = "7.2.0"
    valid, msg = NamelistBase._check_param_version(
        name = "diago_ppcg_maxiter",
        current = current,
        added = added,
        removed = removed,
    )
    assert(valid), msg

    current = "7.0.0"
    valid, msg = NamelistBase._check_param_version(
        name = "diago_ppcg_maxiter",
        current = current,
        added = added,
        removed = removed,
    )
    assert(not valid)
    assert(current in msg)
    assert("added" in msg)
    assert(added in msg)

    current = "7.6.0"
    valid, msg = NamelistBase._check_param_version(
        name = "diago_ppcg_maxiter",
        current = current,
        added = added,
        removed = removed,
    )
    assert(not valid)
    assert(current in msg)
    assert("removed" in msg)
    assert(removed in msg)


@pytest.mark.parametrize(
    argnames=(
        "namelist", "name", "value", "version", "raises", "msg",
    ),
    argvalues=(
        (Control, "calculation", "scf",     "7.6.0", None, None),
        (Control, "calculation", "bean-md", "7.6.0", ValueError, "value"),
        (Control, "calculation", 1234,      "7.6.0", TypeError,  "type"),
        (Control, "nstep", 1,   "7.6.0", None, None),
        (Control, "nstep", "1", "7.6.0", TypeError, "type"),
        (Control, "trism", False, "7.1.0", None, None),
        (Control, "trism", True,  "7.0.0", ValueError, "version"),

        (System, "occupations", "smearing", "7.6.0", None, None),
        (System, "occupations", "bean",     "7.6.0", ValueError, "value"),
        (System, "occupations", 1234,       "7.6.0", TypeError,  "type"),
        (System, "ibrav", 1,   "7.6.0", None, None),
        (System, "ibrav", "1", "7.6.0", TypeError, "type"),
        (System, "celldm", {1: 1.0, 2: 1.0}, "7.6.0", None, None),
        (System, "celldm", {1: "a", 2: "b"}, "7.6.0", TypeError,  "type"),
        (System, "celldm", {1: 1.0, 2: "b"}, "7.6.0", TypeError,  "inconsistent"),
        (System, "celldm", {1: 1.0, 7: 1.0}, "7.6.0", ValueError, "shape"),
        (System, "starting_charge", {1: 1.0, 2: 1.0}, "7.6.0", None, None),
        (System, "starting_charge", {1: "a", 2: "b"}, "7.6.0", TypeError,  "type"),
        (System, "starting_charge", {1: 1.0, 2: "b"}, "7.6.0", TypeError,  "inconsistent"),
        (System, "starting_charge", {1: 1.0, 4: 1.0}, "7.6.0", ValueError, "shape"),

        (Electrons, "diagonalization", "david", "7.6.0", None, None),
        (Electrons, "diagonalization", "bean",  "7.6.0", ValueError, "value"),
        (Electrons, "diagonalization", 1234,    "7.6.0", TypeError,  "type"),
        (Electrons, "efield_cart", {1: 1.0, 2: 1.0}, "7.6.0", None, None),
        (Electrons, "efield_cart", {1: "a", 2: "b"}, "7.6.0", TypeError,  "type"),
        (Electrons, "efield_cart", {1: 1.0, 2: "b"}, "7.6.0", TypeError,  "inconsistent"),
        (Electrons, "efield_cart", {1: 1.0, 7: 1.0}, "7.6.0", ValueError, "shape"),

        (Ions, "ion_dynamics", "bfgs", "7.6.0", None, None),
        (Ions, "ion_dynamics", "bean",  "7.6.0", ValueError, "value"),
        (Ions, "ion_dynamics", 1234,    "7.6.0", TypeError,  "type"),
        (Ions, "nhgrp", {1: 1,   2: 1},   "7.6.0", None, None),
        (Ions, "nhgrp", {1: "a", 2: "b"}, "7.6.0", TypeError,  "type"),
        (Ions, "nhgrp", {1: 1,   2: "b"}, "7.6.0", TypeError,  "inconsistent"),
        (Ions, "nhgrp", {1: 1,   4: 1},   "7.6.0", ValueError, "shape"),
        (Ions, "nhgrp", {1: 1,   2: 1},   "7.2.0", ValueError, "version"),

        (Cell, "cell_dynamics", "bfgs",  "7.6.0", None, None),
        (Cell, "cell_dynamics", "bean",  "7.6.0", ValueError, "value"),
        (Cell, "cell_dynamics", 1234,    "7.6.0", TypeError,  "type"),
        (Cell, "press", 1.0, "7.6.0", None, None),
        (Cell, "press", "a", "7.6.0", TypeError, "type"),

        (Fcp, "fcp_dynamics", "bfgs",  "7.6.0", None, None),
        (Fcp, "fcp_dynamics", "bean",  "7.6.0", ValueError, "value"),
        (Fcp, "fcp_dynamics", 1234,    "7.6.0", TypeError,  "type"),

        (Rism, "laue_wall", "auto", "7.6.0", None, None),
        (Rism, "laue_wall", "bean", "7.6.0", ValueError, "value"),
        (Rism, "laue_wall", 1234,   "7.6.0", TypeError,  "type"),
        (Rism, "laue_wall", "auto", "7.0.0", ValueError, "version"),
        (Rism, "solute_lj", {1: "uff", 2: "none"},    "7.6.0", None, None),
        (Rism, "solute_lj", {1: 1,     2: 1},         "7.6.0", TypeError,  "type"),
        (Rism, "solute_lj", {1: 1,     2: "none"},    "7.6.0", TypeError,  "inconsistent"),
        (Rism, "solute_lj", {1: "uff", 4: "opls-aa"}, "7.6.0", ValueError, "shape"),
        (Rism, "solute_lj", {1: "uff", 2: "none"},    "7.0.0", ValueError, "version"),
    ),
)
def test_check_param(phys_sys, namelist, name, value, version, raises, msg):
    if raises is None:
        namelist.check_param(
            name=name,
            value=value,
            version=version,
            phys_sys=phys_sys,
            strict=True
        )
    else:
        match msg:
            case "type":
                msg = f"The parameter `{name}` has the incorrect type"
            case "inconsistent":
                msg = f"Value for `{name}` has inconsistent types"
            case "value":
                msg = f"The parameter `{name}` has the incorrect value"
            case "shape":
                msg = f"The value for `{name}` does not have the correct shape"
            case "version":
                msg = fr"{name} was added in version .*?, but you are using version {version}"

        with pytest.raises(
            raises,
            match=msg,
        ):
            namelist.check_param(
                name=name,
                value=value,
                version=version,
                phys_sys=phys_sys,
                strict=True
            )

        with pytest.warns(
            NexusUserWarning,
            match=msg,
        ):
            namelist.check_param(
                name=name,
                value=value,
                version=version,
                phys_sys=phys_sys,
                strict=False,
                verbose=True
            )


def test_system_incorporate_phys_sys(phys_sys):
    system = System()
    assert(len(system.params) == 0)
    system.incorporate_phys_sys(phys_sys, overwrite=False)
    assert(system.params.keys() == {"ibrav", "nat", "ntyp", "tot_charge"})
    assert(system.params["ibrav"]      == 0)
    assert(system.params["nat"]        == phys_sys.n_ions)
    assert(system.params["ntyp"]       == phys_sys.n_species)
    assert(system.params["tot_charge"] == phys_sys.net_charge)

    system = System(
        check=True,
        version="7.6.0",
        phys_sys=phys_sys,
        verbose=False,
        strict=True,
        ibrav=15,
        nat=15,
        ntyp=15,
        tot_charge=15,
    )

    system.incorporate_phys_sys(phys_sys, overwrite=False)
    assert(system.params["ibrav"]      == 15)
    assert(system.params["nat"]        == 15)
    assert(system.params["ntyp"]       == 15)
    assert(system.params["tot_charge"] == 15)

    system.incorporate_phys_sys(phys_sys, overwrite=True)

    assert(system.params["ibrav"]      == 0)
    assert(system.params["nat"]        == phys_sys.n_ions)
    assert(system.params["ntyp"]       == phys_sys.n_species)
    assert(system.params["tot_charge"] == phys_sys.net_charge)


def test_atomic_species(phys_sys):
    pseudos = PseudoSet.from_dir(TEST_FILES["pseudos"], code="espresso")
    ref_at_spec = AtomicSpecies(
        atoms=["C", "H", "O"],
        masses={
            "C": Elements.Carbon.atomic_weight,
            "H": Elements.Hydrogen.atomic_weight,
            "O": Elements.Oxygen.atomic_weight,
        },
        pseudos=pseudos.pseudos, # Just get the internal dictionary
    )

    ref_text = """\
ATOMIC_SPECIES
    C  12.011 C.BFD.upf
    H   1.008 H.BFD.upf
    O  15.999 O.BFD.upf
"""

    assert(ref_at_spec.write() == ref_text)

    calc_at_spec = AtomicSpecies.read(ref_text.splitlines())
    assert_card_equals(calc_at_spec, ref_at_spec)

    calc_at_spec = AtomicSpecies(
        atoms=["O", "C", "H"],
        masses={
            "C": Elements.Carbon.atomic_weight,
            "H": Elements.Hydrogen.atomic_weight,
            "O": Elements.Oxygen.atomic_weight,
        },
        pseudos=pseudos.pseudos,
        elem_order=["C", "H", "O"],
    )
    assert_card_equals(calc_at_spec, ref_at_spec)

    # Passing no masses
    calc_at_spec = AtomicSpecies(
        atoms=["C", "H", "O"],
        masses=None,
        pseudos=pseudos.pseudos,
    )
    assert_card_equals(calc_at_spec, ref_at_spec)

    # Passing a PseudoSet object
    calc_at_spec = AtomicSpecies(
        atoms=["C", "H", "O"],
        masses=None,
        pseudos=pseudos,
    )
    assert_card_equals(calc_at_spec, ref_at_spec)

    calc_at_spec = AtomicSpecies.from_phys_sys(phys_sys, pseudos=pseudos)
    assert_card_equals(calc_at_spec, ref_at_spec)

    up_at_spec = deepcopy(ref_at_spec)
    up_at_spec.update(atoms=["C", "H", "O"])
    assert_card_equals(up_at_spec, ref_at_spec)

    with pytest.raises(
        NotAnElementError,
        match="Not all atoms supplied are elements!",
    ):
        AtomicSpecies(["Bean"], masses=None, pseudos=pseudos)

    with pytest.raises(
        ValueError,
        match="No pseudos provided!",
    ):
        AtomicSpecies(["B"], masses=None, pseudos=[])

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(
            "The following pseudopotential files are not present in PseudoSet.pseudo_files"
        ),
    ):
        AtomicSpecies(["B"], masses=None, pseudos=["not_a_file.upf"])

    with pytest.raises(
        ValueError,
        match="Atoms, masses, and pseudos must all have the same atoms!",
    ):
        AtomicSpecies(["C", "H", "O"], masses={"H": 1}, pseudos=pseudos)

    with pytest.raises(
        ValueError,
        match="`elem_order` is missing some atomic species!",
    ):
        AtomicSpecies(["C"], masses=None, pseudos=pseudos, elem_order=["H"])

    with pytest.raises(
        ValueError,
        match="Repeated elements in `elem_order` not allowed!",
    ):
        AtomicSpecies(["C"], masses=None, pseudos=pseudos, elem_order=["C", "C"])


def test_atomic_positions(frozen_phys_sys):
    ref_at_pos = AtomicPositions(
        atoms=["C", "H", "O"],
        positions=np.arange(9, dtype=float).reshape((3, 3)),
        force_multipliers=np.array(
            [
                [True,  False, False],
                [True,  True,  False],
                [False, True,  True],
            ],
            dtype=bool,
        ),
        specifier="angstrom",
    )

    ref_text = """\
ATOMIC_POSITIONS (angstrom)
    C     0.0000000000     1.0000000000     2.0000000000 1 0 0
    H     3.0000000000     4.0000000000     5.0000000000 1 1 0
    O     6.0000000000     7.0000000000     8.0000000000 0 1 1
"""

    assert(ref_at_pos.write() == ref_text)

    calc_at_pos = AtomicPositions.read(ref_text.splitlines())
    assert_card_equals(calc_at_pos, ref_at_pos)

    calc_at_pos = AtomicPositions.from_phys_sys(frozen_phys_sys)
    assert_card_equals(calc_at_pos, ref_at_pos)

    ref_bohr_at_pos = AtomicPositions(
        atoms=["C", "H", "O"],
        positions=convert(np.arange(9, dtype=float).reshape((3, 3)), "A", "B"),
        force_multipliers=np.array(
            [
                [True,  False, False],
                [True,  True,  False],
                [False, True,  True],
            ],
            dtype=bool,
        ),
        specifier="bohr",
    )

    # Since the function doesn't technically need an existing PwscfInput for
    # the specifier change, we can safely pass None here.
    calc_at_pos = ref_at_pos.change_specifier("bohr", None, in_place=False)
    assert_card_equals(calc_at_pos, ref_bohr_at_pos)

    ref_frac = AtomicPositions(
        atoms = ["Cd", "Cd"],
        positions = np.array(
            [
                [1/3, 2/3, 1/4],
                [2/3, 1/3, 3/4],
            ],
            dtype=float,
        ),
        specifier="crystal",
    )

    frac_text = """\
ATOMIC_POSITIONS (crystal)
    Cd    1/3    2/3    1/4
    Cd    2/3    1/3    3/4
"""

    assert_card_equals(AtomicPositions.read(frac_text.splitlines()), ref_frac)

    with pytest.raises(
        ValueError,
        match="Specifier 'bean' is not known to Nexus!",
    ):
        AtomicPositions(specifier="bean", atoms=["B"], positions=np.ones((1, 3)))

    with pytest.raises(
        NotAnElementError,
        match="Not all atoms supplied are elements!",
    ):
        AtomicPositions(specifier="bohr", atoms=["Bean"], positions=np.ones((1, 3)))

    with pytest.raises(
        ValueError,
        match="The pos array must be the same length as the number of atoms!",
    ):
        AtomicPositions(specifier="bohr", atoms=["C"], positions=np.ones((3, 3)))

    with pytest.raises(
        ValueError,
        match="The force_multipliers array must have the same dimension as the positions!",
    ):
        AtomicPositions(
            specifier="bohr",
            atoms=["C", "C", "C"],
            positions=np.ones((3, 3)),
            force_multipliers=np.ones((4, 3), dtype=int),
        )

    with pytest.raises(
        ValueError,
        match="`force_multipliers` must be an integer or boolean array!",
    ):
        AtomicPositions(
            specifier="bohr",
            atoms=["C", "C", "C"],
            positions=np.ones((3, 3)),
            force_multipliers=np.ones((3, 3), dtype=float),
        )

    fm = np.ones((3, 3), dtype=int)
    fm += np.triu(fm) * 2
    with pytest.raises(
        ValueError,
        match=re.escape(
            "All values in `force_multipliers` must be 0 or 1!\n"
            "The array is not 0 or 1 at these locations:\n"
            "  force_multipliers[0, 0] = 3\n"
            "  force_multipliers[0, 1] = 3\n"
            "  force_multipliers[0, 2] = 3\n"
            "  force_multipliers[1, 1] = 3\n"
            "  force_multipliers[1, 2] = 3\n"
            "  force_multipliers[2, 2] = 3\n"
        ),
    ):
        AtomicPositions(
            specifier="bohr",
            atoms=["C", "C", "C"],
            positions=np.ones((3, 3)),
            force_multipliers=fm,
        )


def test_atomic_forces():
    ref_atomic_forces = AtomicForces(
        atoms = ["C", "C", "C"],
        forces = np.zeros((3, 3)),
    )
    assert(ref_atomic_forces.n_atoms == 3)
    assert(ref_atomic_forces.n_species == 1)

    ref_text = """\
ATOMIC_FORCES
    C     0.0000000000     0.0000000000     0.0000000000
    C     0.0000000000     0.0000000000     0.0000000000
    C     0.0000000000     0.0000000000     0.0000000000
"""
    assert(ref_atomic_forces.write() == ref_text)
    read_atomic_forces = AtomicForces.read(ref_text.splitlines())
    assert_card_equals(read_atomic_forces, ref_atomic_forces)

    with pytest.warns(
        NexusUserWarning,
        match=re.escape("Sum of external forces (3.000000e+00) is greater than zero!"),
    ):
        AtomicForces(atoms = ["C"], forces = np.ones((1, 3)))

    with pytest.raises(
        NotAnElementError,
        match=re.escape("Not all atoms supplied are elements!"),
    ):
        AtomicForces(atoms = ["bean"], forces = np.zeros((1, 3)))

    with pytest.raises(
        ValueError,
        match=re.escape("`forces` must have shape (1, 3) but has shape (3, 3)!"),
    ):
        AtomicForces(atoms = ["C"], forces = np.zeros((3, 3)))

    with pytest.raises(
        RuntimeError,
        match="Can not change specifier for card type AtomicForces, no specifiers available!"
    ):
        ref_atomic_forces.change_specifier(None, None)

    with pytest.raises(
        RuntimeError,
        match="Can not create AtomicForces card from a PhysicalSystem/Structure!"
    ):
        ref_atomic_forces.from_phys_sys(None)


def test_atomic_velocities():
    ref_atomic_velocities = AtomicVelocities(
        atoms = ["C", "C", "C"],
        velocities = np.zeros((3, 3)),
    )
    assert(ref_atomic_velocities.n_atoms == 3)
    assert(ref_atomic_velocities.n_species == 1)

    ref_text = """\
ATOMIC_VELOCITIES
    C     0.0000000000     0.0000000000     0.0000000000
    C     0.0000000000     0.0000000000     0.0000000000
    C     0.0000000000     0.0000000000     0.0000000000
"""
    assert(ref_atomic_velocities.write() == ref_text)
    read_atomic_velocities = AtomicVelocities.read(ref_text.splitlines())
    assert_card_equals(read_atomic_velocities, ref_atomic_velocities)

    with pytest.raises(
        NotAnElementError,
        match=re.escape("Not all atoms supplied are elements!"),
    ):
        AtomicVelocities(atoms = ["bean"], velocities = np.zeros((1, 3)))

    with pytest.raises(
        ValueError,
        match=re.escape("`velocities` must have shape (1, 3) but has shape (3, 3)!"),
    ):
        AtomicVelocities(atoms = ["C"], velocities = np.zeros((3, 3)))


def test_kpoints(phys_sys):
    ref_kpoints = KPoints(
        specifier="crystal",
        nkpoints=8,
        kpoints=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            ],
            dtype=float,
        ),
        weights=np.ones((8,), dtype=int),
    )

    assert(ref_kpoints.nkpoints == 8)

    ref_text = """\
K_POINTS (crystal)
  8
    0.0000000000     0.0000000000     0.0000000000     1.0000000000
    0.5000000000     0.0000000000     0.0000000000     1.0000000000
    0.0000000000     0.5000000000     0.0000000000     1.0000000000
    0.5000000000     0.5000000000     0.0000000000     1.0000000000
    0.0000000000     0.0000000000     0.5000000000     1.0000000000
    0.5000000000     0.0000000000     0.5000000000     1.0000000000
    0.0000000000     0.5000000000     0.5000000000     1.0000000000
    0.5000000000     0.5000000000     0.5000000000     1.0000000000
"""

    assert(ref_kpoints.write() == ref_text)

    calc_kpoints = KPoints.read(ref_text.splitlines())
    assert_card_equals(calc_kpoints, ref_kpoints)

    calc_kpoints = KPoints.from_phys_sys(phys_sys, as_crystal=True)
    assert_card_equals(calc_kpoints, ref_kpoints)

    ref_kpoints_tpiba = KPoints(
        specifier="tpiba",
        nkpoints=8,
        kpoints=np.array(
            [
                [0.0000000000000, 0.0000000000000, 0.0000000000000],
                [0.1322943021475, 0.0000000000000, 0.0000000000000],
                [0.0000000000000, 0.1322943021475, 0.0000000000000],
                [0.1322943021475, 0.1322943021475, 0.0000000000000],
                [0.0000000000000, 0.0000000000000, 0.1322943021475],
                [0.1322943021475, 0.0000000000000, 0.1322943021475],
                [0.0000000000000, 0.1322943021475, 0.1322943021475],
                [0.1322943021475, 0.1322943021475, 0.1322943021475],
            ],
            dtype=float,
        ),
        weights=np.ones((8,), dtype=int),
    )
    calc_kpoints = KPoints.from_phys_sys(phys_sys, as_crystal=False)
    assert_card_equals(calc_kpoints, ref_kpoints_tpiba)

    assert(KPoints("automatic", grid=(2, 2, 2), shift=(0, 0, 0)).nkpoints == 8)

    ref_kpoints_gamma = KPoints(specifier="gamma")
    assert(ref_kpoints_gamma.nkpoints == 1)
    gamma_phys_sys = generate_physical_system(
        units = "A",
        axes = np.eye(3) * 2,
        elem = ["C", "H", "O"],
        pos = np.eye(3),
        kgrid = (1, 1, 1),
        kshift = (0, 0, 0),
    )

    calc_kpoints = KPoints.from_phys_sys(gamma_phys_sys, no_assume=False)
    assert_card_equals(calc_kpoints, ref_kpoints_gamma)


    ref_kpoints_L_point = KPoints(
        specifier="automatic", grid=(1, 1, 1), shift=(1, 1, 1)
    )
    L_point_phys_sys = generate_physical_system(
        units = "A",
        axes = np.eye(3) * 2,
        elem = ["C", "H", "O"],
        pos = np.eye(3),
        kgrid = (1, 1, 1),
        kshift = (0.5, 0.5, 0.5),
    )

    calc_kpoints = KPoints.from_phys_sys(L_point_phys_sys, no_assume=False)
    assert_card_equals(calc_kpoints, ref_kpoints_L_point)


    ref_kpoints_automatic = KPoints(
        specifier="automatic", grid=(1, 1, 1), shift=(0, 0, 0)
    )

    calc_kpoints = KPoints.from_phys_sys(
        gamma_phys_sys, no_assume=False, no_gamma=True
    )
    assert_card_equals(calc_kpoints, ref_kpoints_automatic)

    with pytest.raises(
        ValueError,
        match="Specifier 'bean' is not known to Nexus!",
    ):
        KPoints(specifier="bean")

    with pytest.raises(
        ValueError,
        match="Specifier 'bean' is not known to Nexus!",
    ):
        KPoints.read(["K_POINTS (bean)"])

    with pytest.raises(
        ValueError,
        match="Specifier 'bean' is not known to Nexus!",
    ):
        ref_kpoints.change_specifier("bean", None)

    s = Structure(units="B")
    with pytest.raises(
        ValueError,
        match="Structure has no k-points, can not create card!",
    ):
        KPoints.from_phys_sys(s)

    with pytest.raises(
        ValueError,
        match=re.escape("'grid' must have shape (3,)"),
    ):
        KPoints(specifier="automatic", grid=np.ones((4,)), shift=np.ones((3,)))

    with pytest.raises(
        ValueError,
        match=re.escape("'shift' must have shape (3,)"),
    ):
        KPoints(specifier="automatic", grid=np.ones((3,)), shift=np.ones((4,)))

    with pytest.raises(
        ValueError,
        match=re.escape("'kpoints' must have shape (4, 3)"),
    ):
        KPoints(
            specifier="crystal",
            nkpoints=4,
            kpoints=np.ones((5,5)),
            weights=np.ones((4,3)),
        )

    with pytest.raises(
        ValueError,
        match=re.escape("'weights' must have shape (4,)"),
    ):
        KPoints(
            specifier="crystal",
            nkpoints=4,
            kpoints=np.ones((4,3)),
            weights=np.ones((5,2)),
        )


def test_cell_parameters(phys_sys):
    ref_cell_params = CellParameters(
        axes=np.eye(3, dtype=float) * 2,
        specifier="angstrom",
    )

    ref_text = """\
CELL_PARAMETERS (angstrom)
    2.00000000       0.00000000       0.00000000
    0.00000000       2.00000000       0.00000000
    0.00000000       0.00000000       2.00000000
"""

    assert(ref_cell_params.write() == ref_text)

    calc_cell_params = CellParameters.read(ref_text.splitlines())
    assert_card_equals(calc_cell_params, ref_cell_params)

    # Should do nothing
    calc_cell_params = ref_cell_params.change_specifier("angstrom", None, in_place=False)
    assert_card_equals(calc_cell_params, ref_cell_params)

    calc_cell_params = CellParameters.from_phys_sys(phys_sys)
    assert_card_equals(calc_cell_params, ref_cell_params)

    with pytest.raises(
        ValueError,
        match="Specifier 'bean' is not known to Nexus!",
    ):
        CellParameters(axes=np.eye(3), specifier="bean")

    with pytest.raises(
        ValueError,
        match="Specifier 'bean' is not known to Nexus!",
    ):
        CellParameters.read(["CELL_PARAMETERS (bean)"])

    with pytest.raises(
        ValueError,
        match="Specifier 'bean' is not known to Nexus!",
    ):
        ref_cell_params.change_specifier("bean", None)

    with pytest.raises(
        ValueError,
        match=re.escape("'axes' must have shape (3, 3)"),
    ):
        CellParameters(axes=np.arange(3), specifier="angstrom")

    s = Structure()
    with pytest.raises(
        ValueError,
        match="Structure does not have axes!",
    ):
        CellParameters.from_phys_sys(s)


def test_constraints():
    ref_constraints=Constraints(
        constraints=[
            TypeCoordConstraint(1, 2, 2.0, 5.0),
            AtomCoordConstraint(1, 2, 2.0, 5.0),
            DistanceConstraint(1, 2, 5.0),
            PlanarAngleConstraint(1, 2, 3, 5.0),
            TorsionalAngleConstraint(1, 2, 3, 4, 5.0),
            BennettProjConstraint(1, np.array([1, 2, 3]), 5.0),
            PotentialWallConstraint(1.0, 2.0, 3.0),
        ],
        tolerance=1e-5,
    )
    ref_text = """\
CONSTRAINTS
  7  1.000000e-05
    'type_coord'      1          2          2.00000000 5.00000000
    'atom_coord'      1          2          2.00000000 5.00000000
    'distance'        1          2          5.00000000
    'planar_angle'    1          2          3          5.00000000
    'torsional_angle' 1          2          3          4          5.00000000
    'bennett_proj'    1          1.00000000 2.00000000 3.00000000 5.00000000
    'potential_wall'  1.00000000 2.00000000 3.00000000
"""
    assert(ref_constraints.write() == ref_text)
    read_constraints = Constraints.read(ref_text.splitlines())
    print(read_constraints)
    print(ref_constraints)
    assert_card_equals(read_constraints, ref_constraints)

    constr_dict = {
        "type_coord":      {"atom_type_1": 1, "atom_type_2": 2, "cutoff_radius": 2.0, "smoothing": 5.0},
        "atom_coord":      {"atom_idx": 1, "atom_type": 2, "cutoff_radius": 2.0, "smoothing": 5.0},
        "distance":        [{"atom_idx_1": 1, "atom_idx_2": 2, "constr_target": 5.0}],
        "planar_angle":    {"atom_idx_1": 1, "atom_idx_2": 2, "atom_idx_3": 3, "constr_target": 5.0},
        "torsional_angle": {"atom_idx_1": 1, "atom_idx_2": 2, "atom_idx_3": 3, "atom_idx_4": 4, "constr_target": 5.0},
        "bennett_proj":    {"atom_idx": 1, "vector": [1.0, 2.0, 3.0], "constr_target": 5.0},
        "potential_wall":  {"prefactor": 1.0, "exponent": 2.0, "cutoff": 3.0},
    }
    dict_constraints = Constraints.from_mapping(constr_dict, tolerance=1e-5)
    assert_card_equals(dict_constraints, ref_constraints)

    with pytest.raises(
        TypeError,
        match="Received non-constraint type in constraints list!"
    ):
        Constraints(["thing"])

    with pytest.raises(
        ValueError,
        match="Constraint 'bean' is not known to Nexus!"
    ):
        Constraints.read(["CONSTRAINTS", "7", "'bean', 1, 2, 3"])

    with pytest.raises(
        TypeError,
        match="Constraint initialization failed for constraint type 'DistanceConstraint'"
    ):
        Constraints.read(["CONSTRAINTS", "1", "'distance', 1, 2, 3, 4, 5"])

    with pytest.raises(
        ValueError,
        match="Expected 5 constraints but only found 1!"
    ):
        Constraints.read(["CONSTRAINTS", "5", "'distance', 1, 2, 3.00000000"])

    with pytest.raises(
        ValueError,
        match="Constraint 'bean' is not known to Nexus!"
    ):
        Constraints.from_mapping({"bean": []})

    with pytest.raises(
        TypeError,
        match="Must provide a dict of input arguments or a list of dicts of inputs!"
    ):
        Constraints.from_mapping({"distance": 1.0})

    with pytest.raises(
        TypeError,
        match=r"Input `.*?` should have type .*? but has type str!"
    ):
        TypeCoordConstraint("bean", 2, 2.0, 5.0)

    with pytest.raises(
        TypeError,
        match=r"Input `.*?` should have type .*? but has type str!"
    ):
        AtomCoordConstraint("bean", 2, 2.0, 5.0)

    with pytest.raises(
        TypeError,
        match=r"Input `.*?` should have type .*? but has type str!"
    ):
        DistanceConstraint("bean", 2, 5.0)

    with pytest.raises(
        TypeError,
        match=r"Input `.*?` should have type .*? but has type str!"
    ):
        PlanarAngleConstraint("bean", 2, 3, 5.0)

    with pytest.raises(
        TypeError,
        match=r"Input `.*?` should have type .*? but has type str!"
    ):
        TorsionalAngleConstraint("bean", 2, 3, 4, 5.0)

    with pytest.raises(
        TypeError,
        match=r"Input `.*?` should have type .*? but has type str!"
    ):
        BennettProjConstraint("bean", np.array([1, 2, 3]), 5.0)

    with pytest.raises(
        TypeError,
        match=r"Input `.*?` should have type .*? but has type str!"
    ):
        PotentialWallConstraint("bean", 2.0, 3.0)


def test_hubbard_u():
    ref_hubbard = HubbardU("Mn", "3d", 5.0)
    ref_text = "U  Mn-3d  5.0"
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = HubbardU.from_str(ref_text)
    dict_hubbard = HubbardU.from_mapping({"Mn-3d": 5.0})
    assert(read_hubbard == ref_hubbard)
    assert(dict_hubbard[0] == ref_hubbard)

    ref_hubbard = HubbardU("Mn", "3p", manifold_2="3s", u_val=3.0)
    ref_text = "U  Mn-3p-3s  3.0"
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = HubbardU.from_str(ref_text)
    dict_hubbard = HubbardU.from_mapping({"Mn-3p-3s": 3.0})
    assert(read_hubbard == ref_hubbard)
    assert(dict_hubbard[0] == ref_hubbard)

    ref_hubbard = HubbardU("Mn", "3d", 4.70, eigenstates=[3, 4, 5])
    ref_text = "U  Mn-3d  4.7   3 4 5"
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = HubbardU.from_str(ref_text)
    dict_hubbard = HubbardU.from_mapping({
        "Mn-3d": {"value": 4.70, "eigenstates": [3, 4, 5]}
    })
    assert(read_hubbard == ref_hubbard)
    assert(dict_hubbard[0] == ref_hubbard)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Perhaps you should use `HubbardJ0.from_str()` instead?\n"
            "Line: 'J0 Mn-3d 1.0'"
        )
    ):
        HubbardU.from_str("J0 Mn-3d 1.0")


def test_hubbard_j0():
    ref_hubbard = HubbardJ0("Mn", "3d", 1.0)
    ref_text = "J0 Mn-3d  1.0"
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = HubbardJ0.from_str(ref_text)
    dict_hubbard = HubbardJ0.from_mapping({"Mn-3d": 1.0})
    assert(read_hubbard == ref_hubbard)
    assert(dict_hubbard[0] == ref_hubbard)

    ref_hubbard = HubbardJ0("Mn", "3p", manifold_2="3s", j0_val=3.0)
    ref_text = "J0 Mn-3p-3s  3.0"
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = HubbardJ0.from_str(ref_text)
    dict_hubbard = HubbardJ0.from_mapping({"Mn-3p-3s": 3.0})
    assert(read_hubbard == ref_hubbard)
    assert(dict_hubbard[0] == ref_hubbard)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Perhaps you should use `HubbardU.from_str()` instead?\n"
            "Line: 'U Mn-3d 1.0'"
        )
    ):
        HubbardJ0.from_str("U Mn-3d 1.0")


def test_hubbard_alpha():
    ref_hubbard = HubbardAlpha("Ni", "3d", 0.05)
    ref_text = "ALPHA Ni-3d  0.05"
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = HubbardAlpha.from_str(ref_text)
    dict_hubbard = HubbardAlpha.from_mapping({"Ni-3d": 0.05})
    assert(read_hubbard == ref_hubbard)
    assert(dict_hubbard[0] == ref_hubbard)

    ref_hubbard = HubbardAlpha("Ni", "3d", 0.05, eigenstates=[1, 2])
    ref_text = "ALPHA Ni-3d  0.05  1 2"
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = HubbardAlpha.from_str(ref_text)
    dict_hubbard = HubbardAlpha.from_mapping({
        "Ni-3d": {"value": 0.05, "eigenstates": [1, 2]}
    })
    assert(read_hubbard == ref_hubbard)
    assert(dict_hubbard[0] == ref_hubbard)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Perhaps you should use `HubbardU.from_str()` instead?\n"
            "Line: 'U Mn-3d 1.0'"
        )
    ):
        HubbardAlpha.from_str("U Mn-3d 1.0")


def test_hubbard_j():
    ref_hubbard = HubbardJ("Mn", "3d", j_val=1.0)
    ref_text = "J  Mn-3d  1.0"
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = HubbardJ.from_str(ref_text)
    dict_hubbard = HubbardJ.from_mapping({"Mn-3d": 1.0}, itype="J")
    assert(read_hubbard == ref_hubbard)
    assert(dict_hubbard[0] == ref_hubbard)

    ref_hubbard = HubbardJ("Mn", "3d", b_val=1.1)
    ref_text = "B  Mn-3d  1.1"
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = HubbardJ.from_str(ref_text)
    dict_hubbard = HubbardJ.from_mapping({"Mn-3d": 1.1}, itype="B")
    assert(read_hubbard == ref_hubbard)
    assert(dict_hubbard[0] == ref_hubbard)

    ref_hubbard = HubbardJ("Mn", "3d", j_val=1.0, b_val=1.1)
    ref_text = (
        "J  Mn-3d  1.0\n"
        "B  Mn-3d  1.1"
    )
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = HubbardJ.from_str(ref_text)
    assert(read_hubbard == ref_hubbard)

    ref_hubbard = HubbardJ("Mn", "3d", j_val=1.0, b_val=1.1)
    ref_text = (
        "J  Mn-3d  1.0\n"
        "B  Mn-3d  1.1"
    )
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = HubbardJ.from_str(ref_text)
    assert(read_hubbard == ref_hubbard)

    ref_hubbard = HubbardJ("Ce", "4f", j_val=0.65, e2_val=0.12, e3_val=0.04)
    ref_text = (
        "J  Ce-4f  0.65\n"
        "E2 Ce-4f  0.12\n"
        "E3 Ce-4f  0.04"
    )
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = HubbardJ.from_str(ref_text)
    assert(read_hubbard == ref_hubbard)

    ref_hubbard = HubbardJ("Ce", "4f", j_val=0.65, e2_val=0.12, e3_val=0.04)
    joined_hubbard = HubbardJ.join(
        HubbardJ("Ce", "4f", j_val=0.65),
        HubbardJ("Ce", "4f", e2_val=0.12),
        HubbardJ("Ce", "4f", e3_val=0.04),
    )
    assert(joined_hubbard[0] == ref_hubbard)

    with pytest.raises(
        ValueError,
        match="Can not specify E2 and/or E3 value for d-orbital manifold!",
    ):
        HubbardJ("Mn", "3d", j_val=1.0, e2_val=3.0)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Perhaps you should use `HubbardU.from_str()` instead?\n"
            "Line: 'U Mn-3d 1.0'"
        )
    ):
        HubbardJ.from_str("U Mn-3d 1.0")


def test_hubbard_v(phys_sys):
    ref_hubbard = HubbardV("Co", "3d", 1, "Co", "3d", 1, 7.70)
    assert(ref_hubbard.is_onsite())
    assert(not ref_hubbard.is_intersite())
    ref_text = "V  Co-3d    Co-3d     1  1    7.7"
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = HubbardV.from_str(ref_text)
    dict_hubbard = HubbardV.from_mapping({
        ("Co-3d", "Co-3d"): [
            {"indices": (1, 1), "value": 7.70},
        ]
    })
    assert(read_hubbard == ref_hubbard)
    assert(dict_hubbard[0] == ref_hubbard)

    ref_hubbard = HubbardV("Co", "3d", 1, "O", "2p", 19, 0.75)
    assert(not ref_hubbard.is_onsite())
    assert(ref_hubbard.is_intersite())
    ref_text = "V  Co-3d    O-2p      1 19    0.75"
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = HubbardV.from_str(ref_text)
    dict_hubbard = HubbardV.from_mapping({
        ("Co-3d", "O-2p"): [
            {"indices": (1, 19), "value": 0.75},
        ]
    })
    assert(read_hubbard == ref_hubbard)
    assert(dict_hubbard[0] == ref_hubbard)

    ref_hubbard = HubbardV("C", "2p", 1, "C", "2p", 1, 1e-8)
    with pytest.warns(
        DeprecationWarning,
        match="Automatic determination of indices from a physical system is deprecated"
    ):
        dict_hubbard = HubbardV.from_mapping({("C-2p", "C-2p"): 1e-8}, phys_sys=phys_sys)
    assert(dict_hubbard[0] == ref_hubbard)

    ref_hubbard = HubbardV("C", "2p", 1, "O", "2p", 3, 1e-8)
    dict_hubbard = HubbardV.from_mapping(
        {
            ("C-2p", "O-2p"): [{"radius": 5.0, "value": 1e-8}]
        },
        phys_sys=phys_sys,
    )
    assert(dict_hubbard[0] == ref_hubbard)

    ref_hub_params = [
        HubbardU("Co", "3d", 7.70),
        HubbardV("Co", "3d", 1, "Co", "3p", 1, 1.00),
        HubbardV("Co", "3p", 1, "Co", "3p", 1, 2.00),
    ]

    for _ in range(1_000): # Run a lot of random samples
        hub_params = random.sample(ref_hub_params, k=len(ref_hub_params))
        sorted_params = HubbardV.sort_by_interaction_type(v_params=hub_params)
        assert(sorted_params == ref_hub_params)

    ref_hub_params = [
        HubbardV("Co", "3d", 1, "Co", "3d", 1, 7.70),
        HubbardV("Co", "3d", 1, "Co", "3p", 1, 1.00),
        HubbardV("Co", "3p", 1, "Co", "3p", 1, 2.00),
    ]

    for _ in range(1_000):
        hub_params = random.sample(ref_hub_params, k=len(ref_hub_params))
        sorted_params = HubbardV.sort_by_interaction_type(
            v_params=hub_params, standard="Co-3d"
        )
        assert(sorted_params == ref_hub_params)

    for _ in range(1_000):
        hub_params = random.sample(ref_hub_params, k=len(ref_hub_params))
        sorted_params = HubbardV.sort_by_interaction_type(
            v_params=hub_params, standard="Co-3d", background="Co-3p"
        )
        assert(sorted_params == ref_hub_params)

    with pytest.raises(
        ValueError,
        match="Must pass `phys_sys` if indices are not specified!",
    ):
        dict_hubbard = HubbardV.from_mapping(
            {("C-2p", "O-2p"): 1e-8},
        )

    with pytest.raises(
        ValueError,
        match="Must supply 'radius' or 'indices' for HubbardV input mapping!",
    ):
        dict_hubbard = HubbardV.from_mapping(
            {("C-2p", "O-2p"): [{"value": 1e-8}]},
        )

    with pytest.raises(
        TypeError,
        match="Must provide a list of indices/radii for HubbardV input mapping!",
    ):
        dict_hubbard = HubbardV.from_mapping(
            {("C-2p", "O-2p"): "wrong_thing"},
        )

    with pytest.raises(
        RuntimeError,
        match=re.escape("No atoms of specified types found within radius=5.0"),
    ):
        dict_hubbard = HubbardV.from_mapping(
            {("C-2p", "C-2p"): [{"radius": 5.0, "value": 1e-8}]},
            phys_sys=phys_sys,
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "invalid literal for int() with base 10: 'bean'\n"
            "HubbardV.from_mapping() failed with args:"
        ),
    ):
        dict_hubbard = HubbardV.from_mapping({
            ("Co-3d", "O-2p"): [
                {"indices": ("bean", "bean"), "value": 0.75},
            ]
        })


def test_hubbard():
    ref_hubbard = Hubbard(
        specifier="ortho-atomic",
        params=[
            HubbardU("Mn", "3d", 5.0),
            HubbardJ0("Mn", "3d", 1.0),
            HubbardU("Ni", "3d", 6.0),
            HubbardJ0("Ni", "3d", 1.2),
        ]
    )
    ref_text = """\
HUBBARD (ortho-atomic)
    U  Mn-3d  5.0
    J0 Mn-3d  1.0
    U  Ni-3d  6.0
    J0 Ni-3d  1.2
"""
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = Hubbard.read(ref_text.splitlines())
    dict_hubbard = Hubbard.from_mapping(
        {
            "U":  {"Mn-3d": 5.0, "Ni-3d": 6.0},
            "J0": {"Mn-3d": 1.0, "Ni-3d": 1.2},
        },
        specifier="ortho-atomic",
    )
    assert_card_equals(read_hubbard, ref_hubbard)
    assert_card_equals(dict_hubbard, ref_hubbard)

    ref_hubbard = Hubbard(
        specifier="atomic",
        params=[
            HubbardU("Ce", "4f", 5.8),
            HubbardJ("Ce", "4f", j_val=0.65, e2_val=0.12, e3_val=0.04),
            HubbardU("Fe", "3d", 3.7),
            HubbardJ("Fe", "3d", b_val=0.09)
        ]
    )
    ref_text = """\
HUBBARD (atomic)
    U  Ce-4f  5.8
    J  Ce-4f  0.65
    E2 Ce-4f  0.12
    E3 Ce-4f  0.04
    U  Fe-3d  3.7
    B  Fe-3d  0.09
"""
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = Hubbard.read(ref_text.splitlines())
    dict_hubbard = Hubbard.from_mapping(
        inputs={
            "U":  {"Ce-4f": 5.8, "Fe-3d": 3.7},
            "J":  {"Ce-4f": 0.65},
            "E2": {"Ce-4f": 0.12},
            "E3": {"Ce-4f": 0.04},
            "B":  {"Fe-3d": 0.09},
        },
        specifier="atomic"
    )
    assert_card_equals(read_hubbard, ref_hubbard)
    assert_card_equals(dict_hubbard, ref_hubbard)

    ref_hubbard = Hubbard(
        specifier="ortho-atomic",
        params=[
            HubbardU("Co", "3d", 7.70),
            HubbardV("Co", "3d", 1, "O", "2p", 19, 0.75),
            HubbardV("Co", "3d", 1, "O", "2p", 46, 0.75),
            HubbardV("Co", "3d", 1, "O", "2p", 43, 0.75),
            HubbardV("Co", "3d", 1, "O", "2p", 54, 0.75),
            HubbardV("Co", "3d", 1, "O", "2p", 11, 0.75),
            HubbardV("Co", "3d", 1, "O", "2p", 22, 0.75),
        ]
    )
    ref_text = """\
HUBBARD (ortho-atomic)
    U  Co-3d  7.7
    V  Co-3d    O-2p      1 19    0.75
    V  Co-3d    O-2p      1 46    0.75
    V  Co-3d    O-2p      1 43    0.75
    V  Co-3d    O-2p      1 54    0.75
    V  Co-3d    O-2p      1 11    0.75
    V  Co-3d    O-2p      1 22    0.75
"""
    assert(ref_hubbard.write() == ref_text)
    read_hubbard = Hubbard.read(ref_text.splitlines())
    dict_hubbard = Hubbard.from_mapping(
        inputs={
            "U": {"Co-3d": 7.7},
            "V": {
                ("Co-3d", "O-2p"): [
                    {"indices": (1, 19), "value": 0.75},
                    {"indices": (1, 46), "value": 0.75},
                    {"indices": (1, 43), "value": 0.75},
                    {"indices": (1, 54), "value": 0.75},
                    {"indices": (1, 11), "value": 0.75},
                    {"indices": (1, 22), "value": 0.75},
                ]
            },
        },
        specifier="ortho-atomic",
    )
    assert_card_equals(read_hubbard, ref_hubbard)
    assert_card_equals(dict_hubbard, ref_hubbard)


def test_pwscf_input(phys_sys):
    ref_control = Control(
        calculation   = "scf",
        restart_mode  = "from_scratch",
        outdir        = "./output",
        pseudo_dir    = "../pseudo/",
        etot_conv_thr = 1.0e-8,
        forc_conv_thr = 1.0e-6,
        tstress       = True,
        tprnfor       = True,
    )
    ref_system = System(
        ibrav           = 0,
        nat             = 3,
        ntyp            = 3,
        ecutwfc         = 100,
        tot_charge      = 0,
        occupations     = "smearing",
        degauss         = 0.0005 ,
        smearing        = "degauss",
        assume_isolated = "makov-payne",
    )
    ref_electrons = Electrons(
        conv_thr        = 1.0e-8,
        mixing_beta     = 0.7,
        diagonalization = "david",
    )
    ref_atomic_species = AtomicSpecies(
        atoms   = ["C", "H", "O"],
        masses  = {"C": 12.011, "H": 1.008, "O": 15.999},
        pseudos = {
            "C": PseudoSet.pseudo_files["C.BFD.upf"],
            "H": PseudoSet.pseudo_files["H.BFD.upf"],
            "O": PseudoSet.pseudo_files["O.BFD.upf"],
        },
    )
    ref_atomic_positions = AtomicPositions(
        specifier = "angstrom",
        atoms     = ["C", "H", "O"],
        positions = np.arange(9, dtype=float).reshape((3, 3)),
    )
    ref_k_points = KPoints(
        specifier="crystal",
        nkpoints=8,
        kpoints=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            ],
            dtype=float,
        ),
        weights=np.ones((8,), dtype=int),
    )
    ref_cell_parameters = CellParameters(
        specifier = "angstrom",
        axes = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=float,
        ),
    )
    pwscf_input = PwscfInput(
        phys_sys=phys_sys,
        control=ref_control,
        system=ref_system,
        electrons=ref_electrons,
        pseudos=PseudoSet(
            pseudos={
                "C": PseudoSet.pseudo_files["C.BFD.upf"],
                "H": PseudoSet.pseudo_files["H.BFD.upf"],
                "O": PseudoSet.pseudo_files["O.BFD.upf"],
            },
            codes="espresso",
        ),
    )

    assert(pwscf_input.has_required_elements())
    assert(
        pwscf_input.required_elements == {
            "control",
            "system",
            "electrons",
            "atomic_species",
            "atomic_positions",
            "k_points",
            "cell_parameters",
        }
    )
    assert_namelist_equals(pwscf_input.control, ref_control)
    assert_namelist_equals(pwscf_input.system, ref_system)
    assert_namelist_equals(pwscf_input.electrons, ref_electrons)
    assert_card_equals(pwscf_input.atomic_species, ref_atomic_species)
    assert_card_equals(pwscf_input.atomic_positions, ref_atomic_positions)
    assert_card_equals(pwscf_input.k_points, ref_k_points)
    assert_card_equals(pwscf_input.cell_parameters, ref_cell_parameters)
    assert(pwscf_input.ions is None)
    assert(pwscf_input.cell is None)
    assert(pwscf_input.fcp is None)
    assert(pwscf_input.rism is None)
    assert(pwscf_input.atomic_forces is None)
    assert(pwscf_input.atomic_velocities is None)
    assert(pwscf_input.constraints is None)
    assert(pwscf_input.occupations is None)
    assert(pwscf_input.hubbard is None)
    assert(pwscf_input.solvents is None)

    calc_phys_sys = pwscf_input.return_system(structure_only=False, C=4, H=1, O=6)
    ref_phys_sys = phys_sys.copy()
    ref_phys_sys.change_units("B")
    calc_structure = calc_phys_sys.structure
    ref_structure = ref_phys_sys.structure

    assert(dict(calc_phys_sys.valency) == dict(ref_phys_sys.valency))
    assert(calc_structure.elem.tolist() == ref_structure.elem.tolist())
    np.testing.assert_allclose(calc_structure.pos, ref_structure.pos)
    np.testing.assert_allclose(calc_structure.axes, ref_structure.axes)
    np.testing.assert_allclose(calc_structure.kpoints, ref_structure.kpoints)

    new_pwscf_input = deepcopy(pwscf_input)
    new_pwscf_input.incorporate_system(calc_phys_sys)
    new_pwscf_input.atomic_positions.change_specifier("angstrom", new_pwscf_input)
    new_pwscf_input.cell_parameters.change_specifier("angstrom", new_pwscf_input)
    assert_pwscf_input_equals(new_pwscf_input, pwscf_input)


def test_change_specifier():
    ref_system = System(ibrav=0, nat=3, ntyp=3, ecutwfc=100)
    ref_atomic_species = AtomicSpecies(
        atoms   = ["C", "H", "O"],
        masses  = {"C": 12.011, "H": 1.008, "O": 15.999},
        pseudos = {
            "C": PseudoSet.pseudo_files["C.BFD.upf"],
            "H": PseudoSet.pseudo_files["H.BFD.upf"],
            "O": PseudoSet.pseudo_files["O.BFD.upf"],
        },
    )
    ref_atomic_positions = AtomicPositions(
        specifier = "angstrom",
        atoms     = ["C", "H", "O"],
        positions = np.arange(9, dtype=float).reshape((3, 3)),
    )
    ref_k_points = KPoints(
        specifier="crystal",
        nkpoints=8,
        kpoints=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=float,
        ),
        weights=np.ones((8,), dtype=int),
    )
    ref_cell_parameters = CellParameters(
        specifier="angstrom",
        axes=2.0 * np.eye(3, dtype=float),
    )
    pwi = PwscfInput(
        phys_sys=phys_sys,
        control=Control(),
        system=ref_system,
        electrons=Electrons(),
        atomic_species=ref_atomic_species,
        atomic_positions=ref_atomic_positions,
        k_points=ref_k_points,
        cell_parameters=ref_cell_parameters,
    )

    ref_cell_param_bohr = CellParameters(
        specifier = "bohr",
        axes = 3.77945226577 * np.eye(3, dtype=float),
    )
    # alat is always in bohr, so this is the same as above but with a different specifier
    ref_cell_param_alat = CellParameters(
        specifier = "alat",
        axes = 3.77945226577 * np.eye(3, dtype=float),
    )
    # Angstrom to Bohr
    ang_to_bohr = ref_cell_parameters.change_specifier(
        "bohr", pwi, in_place=False,
    )
    assert_card_equals(ang_to_bohr, ref_cell_param_bohr)

    # Bohr to Angstrom
    bohr_to_ang = ref_cell_param_bohr.change_specifier(
        "angstrom", pwi, in_place=False,
    )
    assert_card_equals(bohr_to_ang, ref_cell_parameters)

    # Angstrom to alat
    ang_to_alat = ref_cell_parameters.change_specifier(
        "alat", pwi, in_place=False,
    )
    assert_card_equals(ang_to_alat, ref_cell_param_alat)

    # alat to Angstrom
    alat_to_ang = ref_cell_param_alat.change_specifier(
        "angstrom", pwi, in_place=False,
    )
    assert_card_equals(alat_to_ang, ref_cell_parameters)

    # Bohr to alat
    bohr_to_alat = ref_cell_param_bohr.change_specifier(
        "alat", pwi, in_place=False,
    )
    assert_card_equals(bohr_to_alat, ref_cell_param_alat)

    # alat to Bohr
    alat_to_bohr = ref_cell_param_alat.change_specifier(
        "bohr", pwi, in_place=False,
    )
    assert_card_equals(alat_to_bohr, ref_cell_param_bohr)

    ref_kpts_gamma = KPoints("gamma")
    ref_kpts_tpiba = KPoints(
        specifier="tpiba",
        nkpoints=8,
        kpoints=np.array(
            [
                [0.0000000000000, 0.0000000000000, 0.0000000000000],
                [0.1322943021475, 0.0000000000000, 0.0000000000000],
                [0.0000000000000, 0.1322943021475, 0.0000000000000],
                [0.1322943021475, 0.1322943021475, 0.0000000000000],
                [0.0000000000000, 0.0000000000000, 0.1322943021475],
                [0.1322943021475, 0.0000000000000, 0.1322943021475],
                [0.0000000000000, 0.1322943021475, 0.1322943021475],
                [0.1322943021475, 0.1322943021475, 0.1322943021475],
            ],
            dtype=float,
        ),
        weights=np.ones((8,), dtype=int),
    )

    # crystal to tpiba
    crystal_to_tpiba = ref_k_points.change_specifier("tpiba", pwi, in_place=False)
    assert_card_equals(crystal_to_tpiba, ref_kpts_tpiba)

    # tpiba to crystal
    tpiba_to_crystal = ref_kpts_tpiba.change_specifier("crystal", pwi, in_place=False)
    assert_card_equals(tpiba_to_crystal, ref_k_points)

    # automatic to tpiba
    kpts_automatic = KPoints("automatic", grid=(2, 2, 2), shift=(0, 0, 0))
    auto_to_tpiba = kpts_automatic.change_specifier("tpiba", pwi, in_place=False)
    assert_card_equals(auto_to_tpiba, ref_kpts_tpiba)

    # automatic to crystal
    auto_to_crystal = kpts_automatic.change_specifier("crystal", pwi, in_place=False)
    assert_card_equals(auto_to_crystal, ref_k_points)

    # automatic to gamma
    auto_to_gamma = kpts_automatic.change_specifier("gamma", pwi, in_place=False)
    assert_card_equals(auto_to_gamma, ref_kpts_gamma)

    # tpiba to gamma
    tpiba_to_gamma = ref_kpts_tpiba.change_specifier("gamma", pwi, in_place=False)
    assert_card_equals(tpiba_to_gamma, ref_kpts_gamma)

    # crystal to gamma
    crystal_to_gamma = ref_k_points.change_specifier("gamma", pwi, in_place=False)
    assert_card_equals(crystal_to_gamma, ref_kpts_gamma)

    ref_gamma_auto = KPoints("automatic", grid=(1, 1, 1), shift=(0, 0, 0))
    # gamma to automatic
    gamma_to_auto = ref_kpts_gamma.change_specifier("automatic", pwi, in_place=False)
    assert_card_equals(gamma_to_auto, ref_gamma_auto)

    ref_crystal_auto = KPoints("automatic", grid=(2, 2, 2), shift=(0, 0, 0))
    # crystal to auto
    crystal_to_auto = ref_k_points.change_specifier("automatic", pwi, in_place=False)
    assert_card_equals(crystal_to_auto, ref_crystal_auto)

    ref_shifted_tpiba_auto = KPoints("automatic", grid=(2, 2, 2), shift=(1, 1, 1))
    # shifted tpiba to auto
    kpts = kmesh(pwi.kaxes, (2, 2, 2), (0.5, 0.5, 0.5)) / ((2 * np.pi) / pwi.alat)
    kpts_shifted_tpiba = KPoints("tpiba", kpoints=kpts, weights=np.ones((len(kpts),)))
    shifted_tpiba_to_auto = kpts_shifted_tpiba.change_specifier(
        "automatic", pwi, in_place=False
    )
    assert_card_equals(shifted_tpiba_to_auto, ref_shifted_tpiba_auto)

    ref_gamma_crystal = KPoints(
        "crystal",
        nkpoints=1,
        kpoints=np.array([[0, 0, 0]], dtype=float),
        weights=np.array([1], dtype=float),
    )
    # gamma to automatic
    gamma_to_crystal = ref_kpts_gamma.change_specifier("crystal", pwi, in_place=False)
    assert_card_equals(gamma_to_crystal, ref_gamma_crystal)

    ref_gamma_tpiba = KPoints(
        "tpiba",
        nkpoints=1,
        kpoints=np.array([[0, 0, 0]], dtype=float),
        weights=np.array([1], dtype=float),
    )
    # gamma to tpiba
    gamma_to_tpiba = ref_kpts_gamma.change_specifier("tpiba", pwi, in_place=False)
    assert_card_equals(gamma_to_tpiba, ref_gamma_tpiba)

    tmp_kpts = KPoints(
        "tpiba", kpoints=np.arange(9).reshape(3,3), weights=np.ones((3,))
    )
    with pytest.raises(
        ValueError,
        match="Cannot map arbitrary kpoints into a Monkhorst-Pack mesh!",
    ):
        tmp_kpts.change_specifier("automatic", pwi)

    ref_at_pos_bohr = AtomicPositions(
        specifier = "bohr",
        atoms     = ["C", "H", "O"],
        positions = convert(np.arange(9, dtype=float).reshape((3, 3)), "A", "B"),
    )

    ang_to_bohr = ref_atomic_positions.change_specifier("bohr", pwi, in_place=False)
    assert_card_equals(ang_to_bohr, ref_at_pos_bohr)

    bohr_to_ang = ref_at_pos_bohr.change_specifier("angstrom", pwi, in_place=False)
    assert_card_equals(bohr_to_ang, ref_atomic_positions)

    ref_at_pos_crystal = AtomicPositions(
        specifier = "crystal",
        atoms     = ["C", "H", "O"],
        positions = np.dot(
            np.arange(9, dtype=float).reshape((3, 3)),
            np.linalg.inv(ref_cell_parameters.axes),
        ),
    )

    ang_to_crystal = ref_atomic_positions.change_specifier(
        "crystal", pwi, in_place=False
    )
    assert_card_equals(ang_to_crystal, ref_at_pos_crystal)

    bohr_to_crystal = ref_at_pos_bohr.change_specifier(
        "crystal", pwi, in_place=False
    )
    assert_card_equals(bohr_to_crystal, ref_at_pos_crystal)

    pwi_crystal = PwscfInput(
        control = Control(),
        system = System(
            ibrav = 4,
            a = 2.97887,
            c = 1.88583,
            nat = 2,
            ntyp = 1,
            ecutwfc = 80,
        ),
        electrons = Electrons(),
        atomic_species = AtomicSpecies(
            atoms=["C"], masses=None, pseudos={"C": "C.BFD.upf"},
        ),
        atomic_positions = AtomicPositions(
            specifier = "crystal",
            atoms = ["C", "C"],
            positions = np.array(
                [
                    [0.33333333, 0.66666667, 0.25      ],
                    [0.66666667, 0.33333333, 0.75      ],
                ],
                dtype=float,
            ),
        ),
        k_points = KPoints("gamma"),
    )
    ref_at_pos_bohr = AtomicPositions(
        specifier = "bohr",
        atoms = ["C", "C"],
        positions = np.array(
            [
                [-2.81462426e-08,  3.25004814, 2.65395142],
                [ 2.81462427e+00,  1.62502405, 7.96185425],
            ],
            dtype=float,
        ),
    )
    ref_at_pos_ang = ref_at_pos_bohr.change_specifier("angstrom", None, in_place=False)

    crys_to_bohr = pwi_crystal.atomic_positions.change_specifier(
        "bohr", pwi_crystal, in_place=False,
    )
    assert_card_equals(crys_to_bohr, ref_at_pos_bohr)

    crys_to_ang = pwi_crystal.atomic_positions.change_specifier(
        "angstrom", pwi_crystal, in_place=False,
    )
    assert_card_equals(crys_to_ang, ref_at_pos_ang)


def test_occupations():
    ref_occupations = Occupations(occupations=2 * np.ones((24,), dtype=float))
    ref_text = """\
OCCUPATIONS
    2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000
    2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000
    2.0000000000 2.0000000000 2.0000000000 2.0000000000
"""
    assert(ref_occupations.write() == ref_text)
    read_occupations = Occupations.read(ref_text.splitlines())
    assert_card_equals(read_occupations, ref_occupations)

    ref_min_occs = Occupations(
        occupations=2 * np.ones((12,), dtype=float),
        minority_spin_occs=2 * np.ones((12,), dtype=float),
    )
    pwscf_input = PwscfInput(
        control=Control(calculation="scf"),
        system=System(nbnd=12, nspin=2, occupations="from_input"),
        electrons=Electrons(),
        atomic_species=AtomicSpecies(
            atoms=["C"], masses=None, pseudos={"C": PseudoSet.pseudo_files["C.BFD.upf"]},
        ),
        atomic_positions=AtomicPositions(
            specifier="bohr", atoms=["C"], positions=np.array([[1, 2, 3]]),
        ),
        k_points=KPoints("gamma"),
        occupations=ref_min_occs,
    )
    ref_min_text = """\
OCCUPATIONS
    2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000
    2.0000000000 2.0000000000
    2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000 2.0000000000
    2.0000000000 2.0000000000
"""
    assert(ref_min_occs.write() == ref_min_text)
    read_min_occs = Occupations.read(ref_min_text.splitlines())
    # Before post-process, it should just be equal to the original occupations
    assert_card_equals(read_min_occs, ref_occupations)
    read_min_occs.post_process_read(pwscf_input)
    assert_card_equals(read_min_occs, ref_min_occs)

    with pytest.raises(
        ValueError,
        match=re.escape("Spin occupations must be a 1D array, but has shape (3, 3)!")
    ):
        Occupations(np.eye(3))

    with pytest.raises(
        ValueError,
        match="Majority and minority spin occupations must have the same length!"
    ):
        Occupations([1, 2], [1, 2, 3])

    with pytest.raises(
        ValueError,
        match=re.escape("Minority spin occupations must be a 1D array, but has shape (2, 2)!")
    ):
        Occupations([1, 2], np.eye(2))

    tmp = PwscfInput(
        control=Control(calculation="scf"),
        system=System(occupations="smearing", nbnd=24, nspin=1),
        electrons=Electrons(),
        atomic_species=AtomicSpecies(
            atoms=["C"], masses=None, pseudos={"C": PseudoSet.pseudo_files["C.BFD.upf"]},
        ),
        atomic_positions=AtomicPositions(
            specifier="bohr", atoms=["C"], positions=np.array([[1, 2, 3]]),
        ),
        k_points=KPoints("gamma"),
        occupations=ref_min_occs,
    )
    with pytest.raises(
        ValueError,
        match="Explicit occupations only available with `occupations=from_input`!",
    ):
        ref_occupations.post_process_read(tmp)

    tmp.system.params["occupations"] = "from_input"
    with pytest.raises(
        ValueError,
        match="The number of occupations must be equal to `nbnd`!",
    ):
        Occupations(occupations=np.ones((10,), dtype=float)).post_process_read(tmp)

    tmp.system.params["nspin"] = 2
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The number of occupations must be `2*nbnd` for a spin-polarized calculation!"
        ),
    ):
        Occupations(occupations=np.ones((10,), dtype=float)).post_process_read(tmp)

    del tmp.system.params["nbnd"]
    with pytest.raises(
        ValueError,
        match="Can not set Occupations when `nbnd` is not specified!",
    ):
        Occupations(occupations=np.ones((10,), dtype=float)).post_process_read(tmp)


def test_co2_constrained():
    register_pseudo_files(["C.pz-n-rrkjus_psl.0.1.UPF", "O.pz-n-rrkjus_psl.0.1.UPF"])
    control = Control(
        calculation = "relax",
        restart_mode = "from_scratch",
        prefix = "CO2",
        pseudo_dir = "./pseudo",
        outdir = "./tmp_CO2",
        tstress = True,
    )
    system = System(
        ibrav = 1,
        celldm = {1: 11},
        nat = 3,
        ntyp = 2,
        ecutwfc = 40,
        ecutrho = 320,
        occupations = "smearing",
        smearing = "mv",
        degauss = 0.001,
        nosym = True,
    )
    electrons = Electrons(
        diagonalization = "david",
        mixing_mode = "local-TF",
        mixing_beta = 0.1,
        conv_thr = 1e-06,
        startingwfc = "random",
    )
    ions = Ions(
        ion_dynamics = "damp",
    )
    atomic_species = AtomicSpecies(
        atoms = ["C", "O"],
        masses = {"C": 12.011, "O": 15.999},
        pseudos = {
            "C": PseudoSet.pseudo_files["C.pz-n-rrkjus_psl.0.1.UPF"],
            "O": PseudoSet.pseudo_files["O.pz-n-rrkjus_psl.0.1.UPF"],
        },
        require_pseudos_exist = False,
    )
    atomic_positions = AtomicPositions(
        specifier = "angstrom",
        atoms = ["C", "O", "O"],
        positions = np.array(
            [
                [0.0, 0.0,  0.0     ],
                [0.0, 0.0,  1.514076],
                [0.0, 0.0, -1.514076],
            ],
        ),
        force_multipliers = None,
    )
    k_points = KPoints(
        specifier = "gamma",
        kpoints = None,
        weights = None,
        grid = None,
        shift = None,
    )
    constraints = Constraints(
        constraints = [
            DistanceConstraint(
                atom_idx_1 = 1,
                atom_idx_2 = 2,
                constr_target = 3.1,
            ),
            DistanceConstraint(
                atom_idx_1 = 1,
                atom_idx_2 = 3,
                constr_target = 2.8,
            ),
        ],
        tolerance = None,
    )
    ref_pwscf_input = PwscfInput(
        control=control,
        system=system,
        electrons=electrons,
        ions=ions,
        atomic_species=atomic_species,
        atomic_positions=atomic_positions,
        k_points=k_points,
        constraints=constraints,
    )

    read_pwscf_input = PwscfInput(TEST_FILES["co2_constrained.in"])
    assert_pwscf_input_equals(read_pwscf_input, ref_pwscf_input)

    gen_pwscf_input = generate_pwscf_input(
        pseudos=["C.pz-n-rrkjus_psl.0.1.UPF", "O.pz-n-rrkjus_psl.0.1.UPF"],
        prefix="CO2",
        outdir="./tmp_CO2",
        pseudo_dir="./pseudo",
        restart_mode="from_scratch",
        calculation="relax",
        tstress=True,
        ibrav=1,
        celldm={1: 11.0},
        ecutwfc=40,
        ecutrho=320,
        occupations="smearing",
        smearing="mv",
        degauss=0.001,
        nosym=True,
        diagonalization="david",
        mixing_mode="local-TF",
        mixing_beta=0.1,
        conv_thr=1e-6,
        startingwfc="random",
        ion_dynamics="damp",
        atoms=["C", "O", "O"],
        pos=np.array(
            [
                [0.0000000, 0.0000000,  0.0000000],
                [0.0000000, 0.0000000,  1.5140760],
                [0.0000000, 0.0000000, -1.5140760],
            ],
        ),
        at_pos_specifier="angstrom",
        kpts_specifier="gamma",
        constraints_card=Constraints(
            constraints=[
                DistanceConstraint(1, 2, 3.1),
                DistanceConstraint(1, 3, 2.8),
            ]
        ),
    )
    assert_pwscf_input_equals(gen_pwscf_input, ref_pwscf_input)


def test_furfural_pd_surface():
    register_pseudo_files(
        [
            "Pd.pbe-n-kjpaw_psl.1.0.0.UPF.txt",
            "C.pbe-n-kjpaw_psl.1.0.0.UPF.txt",
            "O.pbe-n-kjpaw_psl.1.0.0.UPF.txt",
            "H.pbe-kjpaw_psl.1.0.0.UPF.txt"
        ]
    )
    atomic_positions = AtomicPositions(
        specifier = "angstrom",
        atoms = [
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "C",
            "C",
            "C",
            "O",
            "C",
            "H",
            "H",
            "H",
            "C",
            "O",
            "H",
            "H",
            "H",
        ],
        positions = np.array(
            [
                [ 1.3864322 ,  0.80045774, 21.80931399],
                [ 4.15923535,  0.80046709, 21.80929567],
                [ 6.93195165,  0.80046709, 21.80929567],
                [ 9.7047548 ,  0.80045774, 21.80931399],
                [ 0.        ,  3.20178607, 21.80930232],
                [ 2.77284237,  3.2017704 , 21.80929567],
                [ 5.545594  ,  3.20175   , 21.80928326],
                [ 8.31834563,  3.2017704 , 21.80929567],
                [-1.38640709,  5.60305775, 21.80929635],
                [ 1.38640709,  5.60305775, 21.80929635],
                [ 4.15920002,  5.60301351, 21.80929567],
                [ 6.93198698,  5.60301351, 21.80929567],
                [-2.77276576,  8.00435697, 21.80930232],
                [-0.        ,  8.00438549, 21.80929635],
                [ 2.77276576,  8.00435697, 21.80930232],
                [ 5.545594  ,  8.00433551, 21.80931399],
                [-0.        ,  1.60089499, 19.52685728],
                [ 2.7728343 ,  1.60089653, 19.52682511],
                [ 5.545594  ,  1.60089679, 19.52679652],
                [ 8.3183537 ,  1.60089653, 19.52682511],
                [-1.38642051,  4.00219831, 19.52682283],
                [ 1.38642051,  4.00219831, 19.52682283],
                [ 4.15921387,  4.0021771 , 19.52679652],
                [ 6.93197313,  4.0021771 , 19.52679652],
                [-2.77279932,  6.40347535, 19.52682283],
                [-0.        ,  6.4035    , 19.52682489],
                [ 2.77279932,  6.40347535, 19.52682283],
                [ 5.545594  ,  6.40345693, 19.52682511],
                [-4.15917769,  8.80480301, 19.52685728],
                [-1.38637781,  8.80482733, 19.52682283],
                [ 1.38637781,  8.80482733, 19.52682283],
                [ 4.15917769,  8.80480301, 19.52685728],
                [ 0.        ,  0.        , 17.263979  ],
                [ 5.545594  ,  0.        , 17.263979  ],
                [ 8.318391  ,  0.        , 17.263979  ],
                [-1.386398  ,  2.401313  , 17.263979  ],
                [ 1.386398  ,  2.401313  , 17.263979  ],
                [ 4.159195  ,  2.401313  , 17.263979  ],
                [ 6.931992  ,  2.401313  , 17.263979  ],
                [-2.772797  ,  4.802625  , 17.263979  ],
                [ 0.        ,  4.802625  , 17.263979  ],
                [ 2.772797  ,  4.802625  , 17.263979  ],
                [ 5.545594  ,  4.802625  , 17.263979  ],
                [-4.159195  ,  7.203938  , 17.263979  ],
                [-1.386398  ,  7.203938  , 17.263979  ],
                [ 1.386398  ,  7.203938  , 17.263979  ],
                [ 4.159195  ,  7.203938  , 17.263979  ],
                [-2.772797  ,  9.60525   , 17.263979  ],
                [ 1.386398  ,  0.800438  , 15.        ],
                [ 4.159195  ,  0.800438  , 15.        ],
                [ 6.931992  ,  0.800438  , 15.        ],
                [ 9.704789  ,  0.800438  , 15.        ],
                [ 0.        ,  3.20175   , 15.        ],
                [ 2.772797  ,  3.20175   , 15.        ],
                [ 5.545594  ,  3.20175   , 15.        ],
                [ 8.318391  ,  3.20175   , 15.        ],
                [-1.386398  ,  5.603063  , 15.        ],
                [ 1.386398  ,  5.603063  , 15.        ],
                [ 4.159195  ,  5.603063  , 15.        ],
                [ 6.931992  ,  5.603063  , 15.        ],
                [-2.772797  ,  8.004375  , 15.        ],
                [ 0.        ,  8.004375  , 15.        ],
                [ 2.772797  ,  8.004375  , 15.        ],
                [ 5.545594  ,  8.004375  , 15.        ],
                [ 2.56958127,  4.2620151 , 23.9110437 ],
                [ 1.91669314,  2.9921421 , 23.88867611],
                [ 0.48123209,  3.26164125, 23.90244359],
                [ 0.30334168,  4.61799222, 24.26578304],
                [ 1.5136284 ,  5.2537245 , 23.9703527 ],
                [ 3.59666269,  4.46019682, 24.20900172],
                [ 2.34549165,  2.04554165, 24.21547897],
                [-0.23585293,  2.60806547, 24.40282683],
                [ 1.57044743,  6.69520364, 24.14824088],
                [ 2.63368832,  7.37533837, 24.05260382],
                [ 0.60578387,  7.16866191, 24.4345088 ],
                [ 3.87682585,  7.04725423, 23.49855519],
                [ 5.66405619,  4.66869667, 22.67603584],
            ],
        ),
        force_multipliers = None,
    )
    atomic_species = AtomicSpecies(
        atoms = [
            "Pd",
            "C",
            "O",
            "H",
        ],
        masses = {"Pd": 106.42, "C": 12.011, "O": 15.999, "H": 1.0079},
        pseudos = {
            "Pd": PseudoSet.pseudo_files["Pd.pbe-n-kjpaw_psl.1.0.0.UPF.txt"],
            "C": PseudoSet.pseudo_files["C.pbe-n-kjpaw_psl.1.0.0.UPF.txt"],
            "O": PseudoSet.pseudo_files["O.pbe-n-kjpaw_psl.1.0.0.UPF.txt"],
            "H": PseudoSet.pseudo_files["H.pbe-kjpaw_psl.1.0.0.UPF.txt"],
        },
    )
    cell = Cell(
        cell_dofree = "ibrav",
    )
    constraints = Constraints(
        constraints = [
            DistanceConstraint(
                atom_idx_1 = 74,
                atom_idx_2 = 76,
                constr_target = 1.4,
            ),
        ],
        tolerance = None,
    )
    control = Control(
        prefix = "pwscf",
        outdir = "pwscf_output",
        calculation = "relax",
        forc_conv_thr = 0.001,
        etot_conv_thr = 0.001,
        max_seconds = 13000000,
        nstep = 200,
        tprnfor = True,
        tstress = True,
        pseudo_dir = "/home/levin-rojas/pseudopot",
    )
    electrons = Electrons(
        conv_thr = 1e-06,
        electron_maxstep = 300,
        mixing_beta = 0.7,
        mixing_mode = "local-TF",
        startingpot = "atomic",
        startingwfc = "atomic+random",
    )
    ions = Ions(
        ion_dynamics = "damp",
    )
    k_points = KPoints(
        specifier = "automatic",
        kpoints = None,
        weights = None,
        grid = np.array([3, 3, 1]),
        shift = np.array([0, 0, 0]),
    )
    system = System(
        a = 11.0912,
        c = 37.1834,
        ecutrho = 448,
        ecutwfc = 36,
        ibrav = 4,
        nat = 77,
        nspin = 2,
        ntyp = 4,
        degauss = 0.04,
        occupations = "smearing",
        smearing = "marzari-vanderbilt",
        starting_magnetization = {1: 0.2},
    )

    ref_pwscf_input = PwscfInput(
        atomic_positions=atomic_positions,
        atomic_species=atomic_species,
        cell=cell,
        constraints=constraints,
        control=control,
        electrons=electrons,
        ions=ions,
        k_points=k_points,
        system=system,
    )

    read_pwscf_input = PwscfInput(TEST_FILES["furfural_pd_surface.in"])
    assert_pwscf_input_equals(read_pwscf_input, ref_pwscf_input)

    pseudos = PseudoSet(
        pseudos={
            "Pd": PseudoSet.pseudo_files["Pd.pbe-n-kjpaw_psl.1.0.0.UPF.txt"],
            "C": PseudoSet.pseudo_files["C.pbe-n-kjpaw_psl.1.0.0.UPF.txt"],
            "O": PseudoSet.pseudo_files["O.pbe-n-kjpaw_psl.1.0.0.UPF.txt"],
            "H": PseudoSet.pseudo_files["H.pbe-kjpaw_psl.1.0.0.UPF.txt"],
        },
        codes="espresso",
    )

    gen_pwscf_input = generate_pwscf_input(
        calculation = "relax",
        pseudo_dir = "/home/levin-rojas/pseudopot",
        pseudos = pseudos,
        forc_conv_thr = 0.001,
        etot_conv_thr = 0.001,
        max_seconds = 13000000,
        nstep = 200,
        tprnfor = True,
        tstress = True,
        cell_dofree = "ibrav",
        ion_dynamics = "damp",
        a = 11.0912,
        c = 37.1834,
        ecutrho = 448,
        ecutwfc = 36,
        ibrav = 4,
        nat = 77,
        nspin = 2,
        ntyp = 4,
        degauss = 0.04,
        occupations = "smearing",
        smearing = "marzari-vanderbilt",
        starting_magnetization = {1: 0.2},
        conv_thr = 1e-06,
        electron_maxstep = 300,
        mixing_beta = 0.7,
        mixing_mode = "local-TF",
        startingpot = "atomic",
        startingwfc = "atomic+random",
        kpts_specifier = "automatic",
        kgrid = (3, 3, 1),
        kshift = (0, 0, 0),
        constraints_card = Constraints(
            constraints = [
                DistanceConstraint(
                    atom_idx_1 = 74,
                    atom_idx_2 = 76,
                    constr_target = 1.4,
                ),
            ],
            tolerance = None,
        ),
        atoms = [
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
            "C",
            "C",
            "C",
            "O",
            "C",
            "H",
            "H",
            "H",
            "C",
            "O",
            "H",
            "H",
            "H",
        ],
        positions = np.array(
            [
                [ 1.3864322 ,  0.80045774, 21.80931399],
                [ 4.15923535,  0.80046709, 21.80929567],
                [ 6.93195165,  0.80046709, 21.80929567],
                [ 9.7047548 ,  0.80045774, 21.80931399],
                [ 0.        ,  3.20178607, 21.80930232],
                [ 2.77284237,  3.2017704 , 21.80929567],
                [ 5.545594  ,  3.20175   , 21.80928326],
                [ 8.31834563,  3.2017704 , 21.80929567],
                [-1.38640709,  5.60305775, 21.80929635],
                [ 1.38640709,  5.60305775, 21.80929635],
                [ 4.15920002,  5.60301351, 21.80929567],
                [ 6.93198698,  5.60301351, 21.80929567],
                [-2.77276576,  8.00435697, 21.80930232],
                [-0.        ,  8.00438549, 21.80929635],
                [ 2.77276576,  8.00435697, 21.80930232],
                [ 5.545594  ,  8.00433551, 21.80931399],
                [-0.        ,  1.60089499, 19.52685728],
                [ 2.7728343 ,  1.60089653, 19.52682511],
                [ 5.545594  ,  1.60089679, 19.52679652],
                [ 8.3183537 ,  1.60089653, 19.52682511],
                [-1.38642051,  4.00219831, 19.52682283],
                [ 1.38642051,  4.00219831, 19.52682283],
                [ 4.15921387,  4.0021771 , 19.52679652],
                [ 6.93197313,  4.0021771 , 19.52679652],
                [-2.77279932,  6.40347535, 19.52682283],
                [-0.        ,  6.4035    , 19.52682489],
                [ 2.77279932,  6.40347535, 19.52682283],
                [ 5.545594  ,  6.40345693, 19.52682511],
                [-4.15917769,  8.80480301, 19.52685728],
                [-1.38637781,  8.80482733, 19.52682283],
                [ 1.38637781,  8.80482733, 19.52682283],
                [ 4.15917769,  8.80480301, 19.52685728],
                [ 0.        ,  0.        , 17.263979  ],
                [ 5.545594  ,  0.        , 17.263979  ],
                [ 8.318391  ,  0.        , 17.263979  ],
                [-1.386398  ,  2.401313  , 17.263979  ],
                [ 1.386398  ,  2.401313  , 17.263979  ],
                [ 4.159195  ,  2.401313  , 17.263979  ],
                [ 6.931992  ,  2.401313  , 17.263979  ],
                [-2.772797  ,  4.802625  , 17.263979  ],
                [ 0.        ,  4.802625  , 17.263979  ],
                [ 2.772797  ,  4.802625  , 17.263979  ],
                [ 5.545594  ,  4.802625  , 17.263979  ],
                [-4.159195  ,  7.203938  , 17.263979  ],
                [-1.386398  ,  7.203938  , 17.263979  ],
                [ 1.386398  ,  7.203938  , 17.263979  ],
                [ 4.159195  ,  7.203938  , 17.263979  ],
                [-2.772797  ,  9.60525   , 17.263979  ],
                [ 1.386398  ,  0.800438  , 15.        ],
                [ 4.159195  ,  0.800438  , 15.        ],
                [ 6.931992  ,  0.800438  , 15.        ],
                [ 9.704789  ,  0.800438  , 15.        ],
                [ 0.        ,  3.20175   , 15.        ],
                [ 2.772797  ,  3.20175   , 15.        ],
                [ 5.545594  ,  3.20175   , 15.        ],
                [ 8.318391  ,  3.20175   , 15.        ],
                [-1.386398  ,  5.603063  , 15.        ],
                [ 1.386398  ,  5.603063  , 15.        ],
                [ 4.159195  ,  5.603063  , 15.        ],
                [ 6.931992  ,  5.603063  , 15.        ],
                [-2.772797  ,  8.004375  , 15.        ],
                [ 0.        ,  8.004375  , 15.        ],
                [ 2.772797  ,  8.004375  , 15.        ],
                [ 5.545594  ,  8.004375  , 15.        ],
                [ 2.56958127,  4.2620151 , 23.9110437 ],
                [ 1.91669314,  2.9921421 , 23.88867611],
                [ 0.48123209,  3.26164125, 23.90244359],
                [ 0.30334168,  4.61799222, 24.26578304],
                [ 1.5136284 ,  5.2537245 , 23.9703527 ],
                [ 3.59666269,  4.46019682, 24.20900172],
                [ 2.34549165,  2.04554165, 24.21547897],
                [-0.23585293,  2.60806547, 24.40282683],
                [ 1.57044743,  6.69520364, 24.14824088],
                [ 2.63368832,  7.37533837, 24.05260382],
                [ 0.60578387,  7.16866191, 24.4345088 ],
                [ 3.87682585,  7.04725423, 23.49855519],
                [ 5.66405619,  4.66869667, 22.67603584],
            ],
        ),
        masses = {"Pd": 106.42, "C": 12.011, "O": 15.999, "H": 1.0079},
        elem_order = ["Pd", "C", "O", "H"],
        at_pos_specifier = "angstrom",
    )
    assert_pwscf_input_equals(gen_pwscf_input, ref_pwscf_input)


def test_h2o_nacl_rism():
    register_pseudo_files(
        ["O.pbe-n-rrkjus_psl.1.0.0.UPF", "H.pbe-rrkjus_psl.1.0.0.UPF"]
    )
    atomic_positions = AtomicPositions(
        specifier = "angstrom",
        atoms = ["O", "H", "H"],
        positions = np.array(
            [
                [10.00000, 10.00000, 10.00000],
                [10.81649, 10.57736, 10.00000],
                [ 9.18351, 10.57736, 10.00000],
            ],
            dtype=float,
        ),
        force_multipliers = None,
    )
    atomic_species = AtomicSpecies(
        atoms = ["O", "H"],
        masses = {"O": -1.0, "H": -1.0},
        pseudos = {
            "O": PseudoSet.pseudo_files["O.pbe-n-rrkjus_psl.1.0.0.UPF"],
            "H": PseudoSet.pseudo_files["H.pbe-rrkjus_psl.1.0.0.UPF"],
        },
    )
    control = Control(
        calculation = "relax",
        prefix = "H2O.NaCl_aq.1mol",
        pseudo_dir = "./pseudo",
        outdir = "./output",
        trism = True,
        nstep = 100,
    )
    electrons = Electrons(
        electron_maxstep = 200,
    )
    ions = Ions(
        ion_dynamics = "bfgs",
    )
    k_points = KPoints(
        specifier = "gamma",
        kpoints = None,
        weights = None,
        grid = None,
        shift = None,
    )
    rism = Rism(
        nsolv = 3,
        closure = "kh",
        tempv = 300,
        ecutsolv = 160,
        solute_lj = {1: "none"},
        solute_epsilon = {1: 0.1554, 2: 0.046},
        solute_sigma = {1: 3.166, 2: 1},
        starting1d = "zero",
        rism1d_conv_thr = 1e-08,
        rism1d_maxstep = 10000,
        mdiis1d_size = 20,
        mdiis1d_step = 0.5,
        starting3d = "zero",
        rism3d_maxstep = 10000,
        rism3d_conv_thr = 1e-06,
    )
    solvents = Solvents(
        specifier = "mol/l",
        labels = [
            "H2O",
            "Na+",
            "Cl-",
        ],
        mol_files = [
            Path("H2O.spc.MOL"),
            Path("Na+.aq.MOL"),
            Path("Cl-.aq.MOL"),
        ],
        densities = [-1.0, 1.0, 1.0],
        densities_left = None,
        densities_right = None,
    )
    system = System(
        ibrav = 1,
        a = 20,
        ntyp = 2,
        nat = 3,
        ecutwfc = 40,
        ecutrho = 320,
        nbnd = 8,
        occupations = "smearing",
        smearing = "gauss",
        degauss = 0.01,
    )
    ref_pwscf_input = PwscfInput(
        atomic_positions=atomic_positions,
        atomic_species=atomic_species,
        control=control,
        electrons=electrons,
        ions=ions,
        k_points=k_points,
        rism=rism,
        solvents=solvents,
        system=system,
    )

    read_pwscf_input = PwscfInput(filepath=TEST_FILES["h2o_nacl_rism.in"])
    assert_pwscf_input_equals(read_pwscf_input, ref_pwscf_input)

    pseudos = PseudoSet(
        pseudos = {
            "O": PseudoSet.pseudo_files["O.pbe-n-rrkjus_psl.1.0.0.UPF"],
            "H": PseudoSet.pseudo_files["H.pbe-rrkjus_psl.1.0.0.UPF"],
        },
    )

    gen_pwscf_input = generate_pwscf_input(
        calculation = "relax",
        prefix = "H2O.NaCl_aq.1mol",
        pseudo_dir = "./pseudo",
        outdir = "./output",
        trism = True,
        nstep = 100,
        ibrav = 1,
        a = 20,
        ntyp = 2,
        nat = 3,
        ecutwfc = 40,
        ecutrho = 320,
        nbnd = 8,
        occupations = "smearing",
        smearing = "gauss",
        degauss = 0.01,
        electron_maxstep = 200,
        ion_dynamics = "bfgs",
        kpts_specifier = "gamma",
        nsolv = 3,
        closure = "kh",
        tempv = 300,
        ecutsolv = 160,
        solute_lj = {1: "none"},
        solute_epsilon = {1: 0.1554, 2: 0.046},
        solute_sigma = {1: 3.166, 2: 1},
        starting1d = "zero",
        rism1d_conv_thr = 1e-08,
        rism1d_maxstep = 10000,
        mdiis1d_size = 20,
        mdiis1d_step = 0.5,
        starting3d = "zero",
        rism3d_maxstep = 10000,
        rism3d_conv_thr = 1e-06,
        solvents_card = Solvents(
            specifier = "mol/l",
            labels = [
                "H2O",
                "Na+",
                "Cl-",
            ],
            mol_files = [
                Path("H2O.spc.MOL"),
                Path("Na+.aq.MOL"),
                Path("Cl-.aq.MOL"),
            ],
            densities = [-1.0, 1.0, 1.0],
            densities_left = None,
            densities_right = None,
        ),
        at_pos_specifier = "angstrom",
        atoms = ["O", "H", "H"],
        positions = np.array(
            [
                [10.00000, 10.00000, 10.00000],
                [10.81649, 10.57736, 10.00000],
                [ 9.18351, 10.57736, 10.00000],
            ],
            dtype=float,
        ),
        masses = {"O": -1.0, "H": -1.0},
        pseudos = pseudos,
    )
    assert_pwscf_input_equals(gen_pwscf_input, ref_pwscf_input)


def test_water_forces():
    register_pseudo_files(["o_pbesol_v1.2.uspp.F.UPF", "h_pbesol_v1.4.uspp.F.UPF"])
    control = Control(
        calculation = "relax",
        nstep = 100,
        tprnfor = True,
        outdir = "relax_output",
        prefix = "relax",
        max_seconds = 1200,
        etot_conv_thr = 1e-06,
        forc_conv_thr = 0.0001,
        pseudo_dir = "./",
    )
    system = System(
        ibrav = 0,
        nat = 3,
        ntyp = 2,
        tot_charge = 0,
        ecutwfc = 80,
        ecutrho = 320,
        input_dft = "pbe",
        assume_isolated = "mp",
    )
    electrons = Electrons(conv_thr = 1e-07)
    ions = Ions(ion_dynamics = "bfgs")
    atomic_species = AtomicSpecies(
        atoms = ["O", "H"],
        masses = {"O": 15.999, "H": 1.008},
        pseudos = {
            "O": PseudoSet.pseudo_files["o_pbesol_v1.2.uspp.F.UPF"],
            "H": PseudoSet.pseudo_files["h_pbesol_v1.4.uspp.F.UPF"],
        },
    )
    atomic_positions = AtomicPositions(
        specifier = "bohr",
        atoms = ["O", "H", "H"],
        positions = np.array(
            [
                [7.9534469 , 7.77734499, 8.68592097],
                [9.81293742, 7.77734499, 8.68592097],
                [7.55890453, 9.59449696, 8.68592097],
            ],
        ),
        force_multipliers = None,
    )
    k_points = KPoints(specifier = "gamma")
    cell_parameters = CellParameters(
        specifier = "bohr",
        axes = np.array(
            [
                [17.37184195,  0.00000000,  0.00000000],
                [ 0.00000000, 17.37184195,  0.00000000],
                [ 0.00000000,  0.00000000, 17.37184195],
            ],
        ),
    )
    atomic_forces = AtomicForces(
        atoms = ["O", "H", "H"],
        forces = np.array(
            [
                [ 0.01117272,  0.01387508,  0.00000000],
                [-0.00940373, -0.00388720,  0.00000000],
                [-0.00176899, -0.00998788,  0.00000000],
            ],
        ),
    )

    ref_pwscf_input = PwscfInput(
        control=control,
        system=system,
        electrons=electrons,
        ions=ions,
        atomic_species=atomic_species,
        atomic_positions=atomic_positions,
        k_points=k_points,
        cell_parameters=cell_parameters,
        atomic_forces=atomic_forces,
    )

    read_pwscf_input = PwscfInput(filepath=TEST_FILES["nexus_water_forces.in"])
    assert_pwscf_input_equals(read_pwscf_input, ref_pwscf_input)

    phys_sys = generate_physical_system(
        units = "B",
        elem = ["O", "H", "H"],
        pos = np.array(
            [
                [7.9534469 , 7.77734499, 8.68592097],
                [9.81293742, 7.77734499, 8.68592097],
                [7.55890453, 9.59449696, 8.68592097],
            ],
        ),
        axes = 17.37184195 * np.eye(3),
        kgrid = (1, 1, 1),
        kshift = (0, 0, 0),
        O = 6,
        H = 1,
    )

    pseudos = PseudoSet(
        pseudos={
            "O": PseudoSet.pseudo_files["o_pbesol_v1.2.uspp.F.UPF"],
            "H": PseudoSet.pseudo_files["h_pbesol_v1.4.uspp.F.UPF"],
        },
        codes="espresso",
    )

    gen_pwscf_input = generate_pwscf_input(
        system = phys_sys,
        pseudos = pseudos,
        calculation = "relax",
        nstep = 100,
        tprnfor = True,
        outdir = "relax_output",
        prefix = "relax",
        max_seconds = 1200,
        etot_conv_thr = 1e-06,
        forc_conv_thr = 0.0001,
        pseudo_dir = "./",
        ibrav = 0,
        nat = 3,
        ntyp = 2,
        tot_charge = 0,
        ecutwfc = 80,
        ecutrho = 320,
        input_dft = "pbe",
        assume_isolated = "mp",
        conv_thr = 1e-07,
        ion_dynamics = "bfgs",
        forces = np.array(
            [
                [ 0.01117272,  0.01387508,  0.00000000],
                [-0.00940373, -0.00388720,  0.00000000],
                [-0.00176899, -0.00998788,  0.00000000],
            ],
        ),
    )
    assert_pwscf_input_equals(gen_pwscf_input, ref_pwscf_input)


def test_cd_fractions():
    register_pseudo_files(["cd_pbesol_v1.uspp.F.UPF"])
    control = Control(
        calculation = "scf",
        nstep = 100,
        tprnfor = True,
        outdir = "scf_output",
        prefix = "scf",
        max_seconds = 1200,
        etot_conv_thr = 1e-06,
        forc_conv_thr = 0.0001,
        pseudo_dir = "./",
    )
    system = System(
        ibrav = 4,
        a = 2.97887,
        c = 1.88583,
        nat = 2,
        ntyp = 1,
        tot_charge = 0,
        ecutwfc = 80,
        ecutrho = 320,
        input_dft = "pbe",
    )
    electrons = Electrons(conv_thr=1e-07)
    atomic_species = AtomicSpecies(
        atoms = ["Cd"],
        masses = {"Cd": 112.414},
        pseudos = {"Cd": PseudoSet.pseudo_files["cd_pbesol_v1.uspp.F.UPF"]},
    )
    atomic_positions = AtomicPositions(
        specifier = "crystal",
        atoms = ["Cd", "Cd"],
        positions = np.array(
            [
                [1/3, 2/3, 1/4],
                [2/3, 1/3, 3/4],
            ],
            dtype=float
        ),
    )
    k_points = KPoints("gamma")

    ref_pwscf_input = PwscfInput(
        control=control,
        system=system,
        electrons=electrons,
        atomic_species=atomic_species,
        atomic_positions=atomic_positions,
        k_points=k_points,
    )

    read_pwscf_input = PwscfInput(filepath=TEST_FILES["nexus_cd_fractions.in"])
    assert_pwscf_input_equals(read_pwscf_input, ref_pwscf_input)

    pseudos = PseudoSet(
        pseudos={"Cd": PseudoSet.pseudo_files["cd_pbesol_v1.uspp.F.UPF"]},
        codes="espresso",
    )

    gen_pwscf_input = generate_pwscf_input(
        pseudos = pseudos,
        calculation = "scf",
        nstep = 100,
        tprnfor = True,
        outdir = "scf_output",
        prefix = "scf",
        max_seconds = 1200,
        etot_conv_thr = 1e-06,
        forc_conv_thr = 0.0001,
        pseudo_dir = "./",
        ibrav = 4,
        a = 2.97887,
        c = 1.88583,
        nat = 2,
        ntyp = 1,
        tot_charge = 0,
        ecutwfc = 80,
        ecutrho = 320,
        input_dft = "pbe",
        conv_thr = 1e-07,
        atoms = ["Cd", "Cd"],
        positions = np.array(
            [
                [1/3, 2/3, 1/4],
                [2/3, 1/3, 3/4],
            ],
            dtype=float
        ),
        at_pos_specifier = "crystal",
        kpts_specifier = "gamma",
    )
    assert_pwscf_input_equals(gen_pwscf_input, ref_pwscf_input)


def test_oxygen_occupations():
    control = Control(
        calculation = "scf",
        prefix = "Oxygen",
        outdir = "pwscf_output",
        pseudo_dir = "./pseudo_dir",
        verbosity = "high",
    )
    system = System(
        ibrav = 1,
        celldm = {1: 14},
        nat = 1,
        ntyp = 1,
        ecutwfc = 50,
        nosym = True,
        occupations = "from_input",
        nbnd = 6,
        nspin = 2,
        starting_magnetization = {1: 0.5},
    )
    electrons = Electrons(
        conv_thr = 1e-07,
    )
    atomic_species = AtomicSpecies(
        atoms = ["O"],
        masses = {"O": 15.999},
        pseudos = {"O": PseudoSet.pseudo_files["O.BFD.upf"]},
    )
    atomic_positions = AtomicPositions(
        specifier = "bohr",
        atoms = ["O"],
        positions = np.array(
            [[0., 0., 0.]],
        ),
        force_multipliers = None,
    )
    k_points = KPoints(
        specifier = "gamma",
        kpoints = None,
        weights = None,
        grid = None,
        shift = None,
    )
    occupations = Occupations(
        occupations = np.array(
            [1, 1, 1, 1, 0, 0],
            dtype=float,
        ),
        minority_spin_occs = np.array(
            [1, 0.3333333333, 0.3333333333, 0.3333333333, 0, 0],
            dtype=float,
        ),
    )

    ref_pwscf_input = PwscfInput(
        control=control,
        system=system,
        electrons=electrons,
        atomic_species=atomic_species,
        atomic_positions=atomic_positions,
        k_points=k_points,
        occupations=occupations,
    )

    read_pwscf_input = PwscfInput(TEST_FILES["nexus_oxygen_occupations.in"])
    assert_pwscf_input_equals(read_pwscf_input, ref_pwscf_input)

    pseudos = PseudoSet(
        pseudos={"O": PseudoSet.pseudo_files["O.BFD.upf"]},
        codes="espresso",
    )

    gen_pwscf_input = generate_pwscf_input(
        calculation = "scf",
        prefix = "Oxygen",
        outdir = "pwscf_output",
        pseudo_dir = "./pseudo_dir",
        pseudos = pseudos,
        verbosity = "high",
        ibrav = 1,
        celldm = {1: 14},
        ecutwfc = 50,
        nosym = True,
        nbnd = 6,
        nspin = 2,
        starting_magnetization = {1: 0.5},
        conv_thr = 1e-07,
        kpts_specifier="gamma",
        occupations = [1, 1, 1, 1, 0, 0],
        minority_spin_occs = [1, 0.3333333333, 0.3333333333, 0.3333333333, 0, 0],
        at_pos_specifier = "bohr",
        atoms = ["O"],
        positions = np.array(
            [[0., 0., 0.]],
        ),
    )
    assert_pwscf_input_equals(gen_pwscf_input, ref_pwscf_input)


@isolate_nexus_core
def test_input(tmp_path):
    register_pseudo_files(
        [
            "V.opt.upf",
            "O.opt.upf",
            "Fe.pbe-nd-rrkjus.UPF",
            "Ar.nexus-test.UPF",
            "H.nexus-test.UPF",
            "Ce.nexus-test.UPF",
            "Fe.nexus-test.UPF",
        ]
    )
    # test compose
    compositions = {}

    # based on sample_inputs/Fe_start_ns_eig.in
    control = Control(
        calculation   = 'scf' ,
        restart_mode  = 'from_scratch' ,
        wf_collect    = True ,
        outdir        = './output' ,
        pseudo_dir    = '../pseudo/' ,
        prefix        = 'fe' ,
        etot_conv_thr = 1.0e-9 ,
        forc_conv_thr = 1.0e-6 ,
        tstress       = True ,
        tprnfor       = True ,
    )
    system = System(
        ibrav           = 1,
        nat             = 2,
        ntyp            = 1,
        ecutwfc         = 100 ,
        ecutrho         = 300 ,
        nbnd            = 18,
        occupations     = 'smearing',
        degauss         = 0.0005 ,
        smearing        = 'methfessel-paxton' ,
        nspin           = 2 ,
        assume_isolated = 'martyna-tuckerman',
        lda_plus_u      = True ,
        **{
            'celldm(1)' : 15,
            'starting_magnetization(1)' : 0.9,
            'hubbard_u(1)' : 3.1,
            'starting_ns_eigenvalue(1,2,1)' : 0.0,
            'starting_ns_eigenvalue(2,2,1)' : 0.0476060,
            'starting_ns_eigenvalue(3,2,1)' : 0.0476060,
            'starting_ns_eigenvalue(4,2,1)' : 0.9654373,
            'starting_ns_eigenvalue(5,2,1)' : 0.9954307,
        }
    )
    electrons = Electrons(
        conv_thr        = 1.0e-9 ,
        mixing_beta     = 0.7 ,
        diagonalization = 'david' ,
        mixing_fixed_ns = 500,
    )
    atomic_species = AtomicSpecies(
        atoms   = ['Fe'],
        masses  = obj(Fe=58.69000),
        pseudos = obj(Fe=PseudoSet.pseudo_files['Fe.pbe-nd-rrkjus.UPF']),
    )
    atomic_positions = AtomicPositions(
        specifier = 'angstrom',
        atoms     = ['Fe','Fe'],
        positions = np.array([
            [2.070000000,   0.000000000,   0.000000000],
            [0.000000000,   0.000000000,   0.000000000],
        ]),
    )
    k_points = KPoints(
        specifier = 'automatic',
        grid      = np.array((1,1,1)),
        shift     = np.array((1,1,1)),
    )
    pwscf_input = PwscfInput(
        control = control,
        system = system,
        electrons = electrons,
        atomic_species = atomic_species,
        atomic_positions = atomic_positions,
        k_points = k_points,
    )
    compositions['Fe_start_ns_eig.in'] = pwscf_input

    # Nexus-authored representative scf input
    control = Control(
        calculation = 'scf',
        prefix      = 'nexus_argon',
        outdir      = './nexus_test_tmp',
        pseudo_dir  = './pseudo',
    )
    system = System(
        ibrav       = 1,
        nat         = 1,
        ntyp        = 1,
        nspin       = 1,
        ecutwfc     = 32.0,
        occupations = 'fixed',
        **{"celldm(1)": 9.25}
    )
    electrons = Electrons(
        conv_thr        = 2.0e-9,
        diagonalization = 'david',
    )
    atomic_species = AtomicSpecies(
        atoms   = ['Ar'],
        masses  = obj(Ar=39.948),
        pseudos = obj(Ar=PseudoSet.pseudo_files['Ar.nexus-test.UPF']),
    )
    atomic_positions = AtomicPositions(
        specifier = 'crystal',
        atoms     = ['Ar'],
        positions = np.array([[0.0,0.0,0.0]]),
    )
    k_points = KPoints(
        specifier = 'automatic',
        grid      = np.array((5,5,5)),
        shift     = np.array((1,1,1)),
    )
    pwscf_input = PwscfInput(
        control = control,
        system = system,
        electrons = electrons,
        atomic_species = atomic_species,
        atomic_positions = atomic_positions,
        k_points = k_points,
    )
    compositions['nexus_argon_scf.in'] = pwscf_input

    # Nexus-authored representative relax input
    control = Control(
        calculation   = 'relax',
        prefix        = 'nexus_h2',
        outdir        = './nexus_test_tmp',
        pseudo_dir    = './pseudo',
        forc_conv_thr = 3.0e-5,
        nstep         = 40,
    )
    system = System(
        ibrav           = 1,
        nat             = 2,
        ntyp            = 1,
        nspin           = 1,
        ecutwfc         = 28.0,
        assume_isolated = 'martyna-tuckerman',
        **{"celldm(1)": 18.0},
    )
    electrons = Electrons(
        conv_thr    = 4.0e-10,
        mixing_beta = 0.55,
    )
    ions = Ions(
        ion_dynamics='bfgs',
    )
    atomic_species = AtomicSpecies(
        atoms   = ['H'],
        masses  = obj(H=1.00794),
        pseudos = obj(H=PseudoSet.pseudo_files['H.nexus-test.UPF']),
    )
    atomic_positions = AtomicPositions(
        specifier        = 'angstrom',
        atoms            = ['H','H'],
        positions        = np.array([
            [4.50,4.50,4.09],
            [4.50,4.50,4.91],
        ]),
        force_multipliers = np.array([
            [0,0,1],
            [0,0,1],
        ], dtype=bool),
    )
    k_points = KPoints(specifier = 'gamma')
    pwscf_input = PwscfInput(
        control = control,
        system = system,
        electrons = electrons,
        ions = ions,
        atomic_species = atomic_species,
        atomic_positions = atomic_positions,
        k_points = k_points,
    )
    compositions['nexus_h2_relax.in'] = pwscf_input

    # Nexus-authored representative bands input
    control = Control(
        calculation = 'bands',
        prefix      = 'nexus_argon',
        outdir      = './nexus_test_tmp',
        pseudo_dir  = './pseudo',
    )
    system = System(
        ibrav       = 1,
        nat         = 1,
        ntyp        = 1,
        nspin       = 1,
        ecutwfc     = 32.0,
        nbnd        = 8,
        occupations = 'fixed',
        **{"celldm(1)": 9.25}
    )
    electrons = Electrons(
        conv_thr        = 2.0e-9,
        diagonalization = 'david',
    )
    atomic_species = AtomicSpecies(
        atoms   = ['Ar'],
        masses  = obj(Ar=39.948),
        pseudos = obj(Ar=PseudoSet.pseudo_files['Ar.nexus-test.UPF']),
    )
    atomic_positions = AtomicPositions(
        specifier = 'crystal',
        atoms     = ['Ar'],
        positions = np.array([[0.0,0.0,0.0]]),
    )
    k_points = KPoints(
        specifier = 'crystal_b',
        nkpoints  = 4,
        kpoints   = np.array([
            [0.0,0.0,0.0],
            [0.5,0.0,0.0],
            [0.5,0.5,0.0],
            [0.0,0.0,0.0],
        ]),
        weights   = np.array((16,16,16,1)),
    )
    pwscf_input = PwscfInput(
        control = control,
        system = system,
        electrons = electrons,
        atomic_species = atomic_species,
        atomic_positions = atomic_positions,
        k_points = k_points,
    )
    compositions['nexus_argon_bands.in'] = pwscf_input

    # Nexus-authored representative Hubbard B/E2/E3 input
    control = Control(
        calculation = 'scf',
        prefix       = 'nexus_hubbard',
        outdir       = './nexus_test_tmp',
        pseudo_dir   = './pseudo',
    )
    system = System(
        ibrav       = 1,
        nat         = 2,
        ntyp        = 2,
        nspin       = 1,
        ecutwfc     = 36.0,
        occupations = 'fixed',
        **{"celldm(1)": 10.5}
    )
    electrons = Electrons(
        conv_thr        = 3.0e-9,
        diagonalization = 'david',
    )
    atomic_species = AtomicSpecies(
        atoms   = ['Ce','Fe'],
        masses  = obj(Ce=140.116,Fe=55.845),
        pseudos = obj(
            Ce = PseudoSet.pseudo_files['Ce.nexus-test.UPF'],
            Fe = PseudoSet.pseudo_files['Fe.nexus-test.UPF'],
        ),
    )
    atomic_positions = AtomicPositions(
        specifier = 'crystal',
        atoms     = ['Ce','Fe'],
        positions = np.array([
            [0.0,0.0,0.0],
            [0.5,0.5,0.5],
        ]),
    )
    k_points = KPoints(
        specifier = 'automatic',
        grid      = np.array((4,4,4)),
        shift     = np.array((0,0,0)),
    )
    hubbard = Hubbard(
        specifier = 'atomic',
        params = [
            HubbardU("Ce", "4f", 5.8),
            HubbardU("Fe", "3d", 3.7),
            HubbardJ("Fe", "3d", b_val=0.09),
            HubbardJ("Ce", "4f", j_val=0.65, e2_val=0.12, e3_val=0.04),
        ]
    )
        # hubbard   = {
        #     'U' : {'Ce-4f':5.8,'Fe-3d':3.7},
        #     'J' : {'Ce-4f':0.65},
        #     'B' : {'Fe-3d':0.09},
        #     'E2': {'Ce-4f':0.12},
        #     'E3': {'Ce-4f':0.04},
        #     },
    pwscf_input = PwscfInput(
        control = control,
        system = system,
        electrons = electrons,
        atomic_species = atomic_species,
        atomic_positions = atomic_positions,
        k_points = k_points,
        hubbard = hubbard,
    )
    compositions['nexus_hubbard_e2_e3.in'] = pwscf_input


    # test read
    for input_file,pw in compositions.items():
        pwr = PwscfInput(TEST_FILES[input_file])
        pwc = deepcopy(pw)
        assert_pwscf_input_equals(pwc, pwr)
    #end for


    # test write
    for input_file,pw in compositions.items():
        write_path = tmp_path / ('composed_'+input_file)
        pw.write(write_path)
        pw2 = PwscfInput(write_path)
        pwr = PwscfInput(TEST_FILES[input_file])
        assert_pwscf_input_equals(pw2,pwr)
    #end for


    # test read/write/read
    reads = obj()
    for input_file, file_path in TEST_FILES.items():
        if file_path.suffix != ".in":
            continue
        read_path = file_path
        write_path = tmp_path / input_file
        pw = PwscfInput(read_path)
        pw.write(write_path)
        pw2 = PwscfInput(write_path)
        assert_pwscf_input_equals(pw,pw2)
        reads[input_file] = pw
    #end for


    # test generate
    generations = obj()

    # based on sample_inputs/VO2_M1_afm.in
    infile      = 'VO2_M1_afm.in'
    struct_file = TEST_FILES['VO2_M1_afm.xsf']
    read_path   = TEST_FILES[infile]
    write_path  = tmp_path / infile

    s = read_structure(struct_file)
    s.elem[0] = 'V1'
    s.elem[1] = 'V2'
    s.elem[2] = 'V1'
    s.elem[3] = 'V2'

    vo2 = generate_physical_system(
        structure = s,
        V1        = 13,
        V2        = 13,
        O         =  6,
        )

    pw = generate_pwscf_input(
        calculation      = 'scf',
        disk_io          = 'low',
        verbosity        = 'high',
        wf_collect       = True,
        input_dft        = 'lda',
        hubbard_u        = obj(V1=3.5,V2=3.5),
        ecutwfc          = 350,
        bandfac          = 1.3,
        nosym            = True,
        occupations      = 'smearing',
        smearing         = 'fermi-dirac',
        degauss          = 0.0001,
        nspin            = 2,
        start_mag        = obj(V1=1.0,V2=-1.0),
        diagonalization  = 'david',
        conv_thr         = 1e-8,
        mixing_beta      = 0.2,
        electron_maxstep = 1000,
        system           = vo2,
        pseudos          = ['V.opt.upf','O.opt.upf'],
        kgrid            = (6,6,6),
        kshift           = (0,0,0),
        # added for reverse compatibility
        celldm           = {1:1.0},
        cell_params_specifier = 'alat',
        at_pos_specifier = 'alat',
        # Previous version sorted alphabetically, but we sort by order in structure.
        # This causes the order of the labels in ATOMIC_SPECIES to be different
        # The Hubbard_U and starting_magnetization indices are correct in both
        # versions, but they are not the same.
        # To get around this, we are manually providing the order.
        elem_order = ["O", "V1", "V2"]
        )

    # Our atomic mass for Vanadium has 1 extra digit of precision, 50.9415,
    # whereas the user input used 50.942. The previous code used a very high
    # absolute tolerance for float comparison (5e-4), which masked this
    # difference. To avoid this with our higher sensitivity to differences,
    # I am going to change the masses manually for this test.
    pw.atomic_species.masses["V1"] = 50.942
    pw.atomic_species.masses["V2"] = 50.942

    generations[infile] = pw

    pw.write(write_path)
    pw2 = PwscfInput(read_path)
    pw3 = PwscfInput(write_path)
    assert_pwscf_input_equals(pw2,pw3)

    # based on sample_inputs/Fe_start_ns_eig.in
    infile     = 'Fe_start_ns_eig.in'
    read_path  = TEST_FILES[infile]
    write_path = tmp_path / infile

    pw = generate_pwscf_input(
        calculation     = 'scf',
        restart_mode    = 'from_scratch',
        wf_collect      = True,
        outdir          = './output',
        pseudo_dir      = '../pseudo/',
        prefix          = 'fe',
        etot_conv_thr   = 1.0e-9,
        forc_conv_thr   = 1.0e-6,
        tstress         = True,
        tprnfor         = True,
        ibrav           = 1,
        nat             = 2,
        ntyp            = 1,
        ecutwfc         = 100,
        ecutrho         = 300,
        nbnd            = 18,
        occupations     = 'smearing',
        degauss         = 0.0005,
        smearing        = 'methfessel-paxton',
        nspin           = 2,
        assume_isolated = 'martyna-tuckerman',
        lda_plus_u      = True,
        conv_thr        = 1.0e-9,
        mixing_beta     = 0.7,
        diagonalization = 'david',
        mixing_fixed_ns = 500,
        mass            = obj(Fe=58.69000),
        pseudos         = ['Fe.pbe-nd-rrkjus.UPF'],
        elem            = ['Fe','Fe'],
        pos             = [[2.070000000, 0.000000000, 0.000000000],
                           [0.000000000, 0.000000000, 0.000000000]],
        at_pos_specifier   = 'angstrom',
        kgrid           = np.array((1,1,1)),
        kshift          = np.array((1,1,1)),
        **{
            'celldm(1)' : 15,
            'starting_magnetization(1)' : 0.9,
            'hubbard_u(1)' : 3.1,
            'starting_ns_eigenvalue(1,2,1)' : 0.0,
            'starting_ns_eigenvalue(2,2,1)' : 0.0476060,
            'starting_ns_eigenvalue(3,2,1)' : 0.0476060,
            'starting_ns_eigenvalue(4,2,1)' : 0.9654373,
            'starting_ns_eigenvalue(5,2,1)' : 0.9954307,
        }
    )

    generations[infile] = pw

    pw2 = compositions[infile]
    assert_pwscf_input_equals(pw,pw2)
    pw.write(write_path)
    pw3 = PwscfInput(write_path)
    pw4 = reads[infile]
    assert_pwscf_input_equals(pw3,pw4)


    # based on sample_inputs/Fe_start_ns_eig.in
    #  variant that uses direct pwscf array input
    pw = generate_pwscf_input(
        calculation     = 'scf',
        restart_mode    = 'from_scratch',
        wf_collect      = True,
        outdir          = './output',
        pseudo_dir      = '../pseudo/',
        prefix          = 'fe',
        etot_conv_thr   = 1.0e-9,
        forc_conv_thr   = 1.0e-6,
        tstress         = True,
        tprnfor         = True,
        ibrav           = 1,
        nat             = 2,
        ntyp            = 1,
        ecutwfc         = 100,
        ecutrho         = 300,
        nbnd            = 18,
        occupations     = 'smearing',
        degauss         = 0.0005,
        smearing        = 'methfessel-paxton',
        nspin           = 2,
        assume_isolated = 'martyna-tuckerman',
        lda_plus_u      = True,
        conv_thr        = 1.0e-9,
        mixing_beta     = 0.7,
        diagonalization = 'david',
        mixing_fixed_ns = 500,
        mass            = obj(Fe=58.69000),
        pseudos         = ['Fe.pbe-nd-rrkjus.UPF'],
        elem            = ['Fe','Fe'],
        pos             = [[2.070000000, 0.000000000, 0.000000000],
                           [0.000000000, 0.000000000, 0.000000000]],
        at_pos_specifier   = 'angstrom',
        kgrid           = np.array((1,1,1)),
        kshift          = np.array((1,1,1)),
        starting_ns_eigenvalue = {(1,2,1) : 0.0,
                                  (2,2,1) : 0.0476060,
                                  (3,2,1) : 0.0476060,
                                  (4,2,1) : 0.9654373,
                                  (5,2,1) : 0.9954307,},
        celldm                 = {1 : 15 },
        starting_magnetization = {1 : 0.9},
        hubbard_u              = {1 : 3.1},
        )

    pwg = deepcopy(pw)

    generations[infile] = pw

    pw2 = deepcopy(compositions[infile])
    assert_pwscf_input_equals(pwg,pw2)
    pw3 = reads[infile]
    assert_pwscf_input_equals(pwg,pw3)
    pw.write(write_path)
    pw4 = PwscfInput(write_path)
    assert_pwscf_input_equals(pwg,pw3)

    # Nexus-authored representative scf input
    infile     = 'nexus_argon_scf.in'
    write_path = tmp_path / infile
    pw = generate_pwscf_input(
        calculation     = 'scf',
        prefix          = 'nexus_argon',
        outdir          = './nexus_test_tmp',
        pseudo_dir      = './pseudo',
        ibrav           = 1,
        celldm          = {1:9.25},
        nat             = 1,
        ntyp            = 1,
        nspin           = 1,
        ecutwfc         = 32.0,
        occupations     = 'fixed',
        conv_thr        = 2.0e-9,
        diagonalization = 'david',
        mass            = obj(Ar=39.948),
        pseudos         = ['Ar.nexus-test.UPF'],
        elem            = ['Ar'],
        pos             = [[0.0,0.0,0.0]],
        at_pos_specifier   = 'crystal',
        kgrid           = (5,5,5),
        kshift          = (1,1,1),
        )
    generations[infile] = pw
    pw.write(write_path)
    pw2 = PwscfInput(write_path)
    assert_pwscf_input_equals(pw2,reads[infile])

    # Nexus-authored representative relax input
    infile     = 'nexus_h2_relax.in'
    write_path = tmp_path / infile
    pw = generate_pwscf_input(
        calculation     = 'relax',
        prefix          = 'nexus_h2',
        outdir          = './nexus_test_tmp',
        pseudo_dir      = './pseudo',
        forc_conv_thr   = 3.0e-5,
        nstep           = 40,
        ibrav           = 1,
        celldm          = {1:18.0},
        nat             = 2,
        ntyp            = 1,
        nspin           = 1,
        ecutwfc         = 28.0,
        assume_isolated = 'martyna-tuckerman',
        conv_thr        = 4.0e-10,
        mixing_beta     = 0.55,
        ion_dynamics    = 'bfgs',
        mass            = obj(H=1.00794),
        pseudos         = ['H.nexus-test.UPF'],
        elem            = ['H','H'],
        pos             = [[4.50,4.50,4.09],
                           [4.50,4.50,4.91]],
        at_pos_specifier   = 'angstrom',
        kgrid           = (1,1,1),
        kshift          = (0,0,0),
        force_multipliers = np.array([
            [0,0,1],
            [0,0,1],
        ]),
    )
    generations[infile] = pw
    pw.write(write_path)
    pw2 = PwscfInput(write_path)
    assert_pwscf_input_equals(pw2,reads[infile])

    # Nexus-authored representative bands input.  The explicit path is set on
    # the generated input because the generic interface only accepts regular
    # k-point grids directly.
    infile     = 'nexus_argon_bands.in'
    write_path = tmp_path / infile
    k_points = KPoints(
        specifier = 'crystal_b',
        nkpoints  = 4,
        kpoints   = np.array([
            [0.0,0.0,0.0],
            [0.5,0.0,0.0],
            [0.5,0.5,0.0],
            [0.0,0.0,0.0],
            ]),
        weights   = np.array((16,16,16,1)),
    )
    pw = generate_pwscf_input(
        calculation     = 'bands',
        prefix          = 'nexus_argon',
        outdir          = './nexus_test_tmp',
        pseudo_dir      = './pseudo',
        ibrav           = 1,
        celldm          = {1:9.25},
        nat             = 1,
        ntyp            = 1,
        nspin           = 1,
        ecutwfc         = 32.0,
        nbnd            = 8,
        occupations     = 'fixed',
        conv_thr        = 2.0e-9,
        diagonalization = 'david',
        mass            = obj(Ar=39.948),
        pseudos         = ['Ar.nexus-test.UPF'],
        elem            = ['Ar'],
        pos             = [[0.0,0.0,0.0]],
        at_pos_specifier   = 'crystal',
        kgrid           = (1,1,1),
        kshift          = (0,0,0),
        k_points_card   = k_points,
        )
    generations[infile] = pw
    pw.write(write_path)
    pw2 = PwscfInput(write_path)
    assert_pwscf_input_equals(pw2,reads[infile])

    # Nexus-authored representative Hubbard B/E2/E3 input
    infile     = 'nexus_hubbard_e2_e3.in'
    write_path = tmp_path / infile
    pw = generate_pwscf_input(
        calculation     = 'scf',
        prefix          = 'nexus_hubbard',
        outdir          = './nexus_test_tmp',
        pseudo_dir      = './pseudo',
        ibrav           = 1,
        celldm          = {1:10.5},
        nat             = 2,
        ntyp            = 2,
        nspin           = 1,
        ecutwfc         = 36.0,
        occupations     = 'fixed',
        conv_thr        = 3.0e-9,
        diagonalization = 'david',
        mass            = obj(Ce=140.116,Fe=55.845),
        pseudos         = ['Ce.nexus-test.UPF','Fe.nexus-test.UPF'],
        elem            = ['Ce','Fe'],
        pos             = [[0.0,0.0,0.0],
                           [0.5,0.5,0.5]],
        at_pos_specifier   = 'crystal',
        kgrid           = (4,4,4),
        kshift          = (0,0,0),
        hubbard         = {
            'U' : {'Ce-4f':5.8,'Fe-3d':3.7},
            'J' : {'Ce-4f':0.65},
            'B' : {'Fe-3d':0.09},
            'E2': {'Ce-4f':0.12},
            'E3': {'Ce-4f':0.04},
            },
        hubbard_specifier = 'atomic',
        )
    generations[infile] = pw
    pw.write(write_path)
    pw2 = PwscfInput(write_path)
    assert_pwscf_input_equals(pw2,reads[infile])


@isolate_nexus_core
def test_supplemental_input_generation(tmp_path):
    """Generate representative inputs for supplemental analyzer run styles."""
    def check_round_trip(name,pw):
        write_path = tmp_path / f'{name}.in'
        pw.write(write_path)
        pwr = PwscfInput(write_path)
        assert_pwscf_input_equals(pw, pwr)
    #end def check_round_trip
    register_pseudo_files(["C.UPF", "B.ccECP.upf", "N.ccECP.upf"])

    carbon = dict(
        prefix        = 'carbon',
        outdir        = './tmp',
        pseudo_dir    = './pseudo',
        tstress       = True,
        tprnfor       = True,
        ibrav         = 2,
        celldm        = {1:6.80},
        nat           = 2,
        ntyp          = 1,
        ecutwfc       = 25.0,
        ecutrho       = 100.0,
        occupations   = 'fixed',
        nosym         = True,
        noinv         = True,
        conv_thr      = 1e-8,
        mixing_beta   = 0.5,
        diagonalization = 'david',
        mass          = obj(C=12.0107),
        pseudos       = ['C.UPF'],
        elem          = ['C','C'],
        pos           = [[0.0,0.0,0.0],[0.255,0.245,0.250]],
        at_pos_specifier = 'crystal',
        kgrid         = np.array((2,2,2)),
        kshift        = np.array((0,0,0)),
        )

    md = generate_pwscf_input(
        calculation     = 'md',
        nstep           = 5,
        iprint          = 1,
        dt              = 5.0,
        ion_dynamics    = 'verlet',
        ion_temperature = 'initial',
        tempw           = 300.0,
        **carbon,
        )
    assert md.control.calculation == 'md'
    assert md.control.iprint == 1
    assert md.ions.ion_dynamics == 'verlet'
    check_round_trip('md',md)

    vc_md = generate_pwscf_input(
        calculation     = 'vc-md',
        nstep           = 5,
        iprint          = 1,
        dt              = 5.0,
        ion_dynamics    = 'beeman',
        ion_temperature = 'initial',
        tempw           = 300.0,
        cell_dynamics   = 'pr',
        press           = 0.0,
        cell_dofree     = 'all',
        **carbon,
        )
    assert vc_md.control.calculation == 'vc-md'
    assert vc_md.ions.ion_dynamics == 'beeman'
    assert vc_md.cell.cell_dynamics == 'pr'
    check_round_trip('vc_md',vc_md)

    kpts_card = KPoints(
        specifier = 'crystal',
        nkpoints  = 4,
        kpoints   = np.array([
            [0.00,0.00,0.00],
            [0.25,0.00,0.00],
            [0.25,0.25,0.00],
            [0.25,0.25,0.25],
        ]),
        weights   = np.array([0.10,0.20,0.30,0.40]),
    )
    crystal_kpoints = generate_pwscf_input(
        calculation='scf',
        k_points_card=kpts_card,
        **carbon
    )
    assert crystal_kpoints.k_points.specifier == 'crystal'
    assert np.allclose(crystal_kpoints.k_points.weights,[0.10,0.20,0.30,0.40])
    check_round_trip('crystal_kpoints',crystal_kpoints)

    cbn = dict(
        calculation   = 'scf',
        prefix        = 'cbn',
        outdir        = './tmp',
        pseudo_dir    = './pseudo',
        tstress       = True,
        tprnfor       = True,
        ibrav         = 2,
        celldm        = {1:6.82},
        nat           = 2,
        ntyp          = 2,
        ecutwfc       = 300.0,
        ecutrho       = 1200.0,
        occupations   = 'smearing',
        smearing      = 'gaussian',
        degauss       = 0.02,
        nbnd          = 8,
        conv_thr      = 1e-8,
        mixing_beta   = 0.5,
        diagonalization = 'david',
        mass          = obj(B=10.81,N=14.01),
        pseudos       = ['B.ccECP.upf','N.ccECP.upf'],
        elem          = ['B','N'],
        pos           = [[0.0,0.0,0.0],[0.25,0.25,0.25]],
        at_pos_specifier = 'crystal',
        kgrid         = np.array((3,3,3)),
        kshift        = np.array((0,0,0)),
        )
    cbn_spin = generate_pwscf_input(
        nspin                    = 2,
        starting_magnetization   = {1:0.5,2:-0.5},
        **cbn,
        )
    assert cbn_spin.atomic_species.atoms == ['B','N']
    assert cbn_spin.system.nspin == 2
    assert cbn_spin.system.ecutwfc == 300.0
    check_round_trip('cbn_spin',cbn_spin)

    cbn_vc_relax_kwargs = dict(cbn)
    cbn_vc_relax_kwargs.update(
        calculation     = 'vc-relax',
        occupations     = 'fixed',
        ion_dynamics    = 'bfgs',
        cell_dynamics   = 'bfgs',
        press           = 0.0,
        press_conv_thr  = 0.5,
        cell_dofree     = 'all',
        )
    cbn_vc_relax = generate_pwscf_input(**cbn_vc_relax_kwargs)
    assert cbn_vc_relax.control.calculation == 'vc-relax'
    assert cbn_vc_relax.cell.press_conv_thr == 0.5
    check_round_trip('cbn_vc_relax',cbn_vc_relax)
