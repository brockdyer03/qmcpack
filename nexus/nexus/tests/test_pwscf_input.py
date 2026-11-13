from copy import deepcopy

import numpy as np
import pytest

from ..developer import obj
from ..periodic_table import Elements
from ..physical_system import generate_physical_system
from ..pseudoset import PseudoSet
from ..pwscf_input import (
    AtomicForces,
    AtomicPositions,
    AtomicSpecies,
    CardBase,
    Cell,
    CellParameters,
    Constraints,
    Control,
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
    PwscfInput,
    Rism,
    System,
    generate_pwscf_input,
)
from ..pwscf_input_defs import SystemDefinitions
from ..structure import read_structure
from ..unit_converter import convert
from . import TEST_DIR, NexusTestOrder, isolate_nexus_core, register_pseudo_files

pytestmark = pytest.mark.order(NexusTestOrder.PWSCF_INPUT)

TEST_FILES = {
    "Cr_noncolin.in":         TEST_DIR / "test_pwscf_input_files/Cr_noncolin.in",
    "Fe_start_ns_eig.in":     TEST_DIR / "test_pwscf_input_files/Fe_start_ns_eig.in",
    "LiI_vc_relax.in":        TEST_DIR / "test_pwscf_input_files/LiI_vc_relax.in",
    "Ni_surface.in":          TEST_DIR / "test_pwscf_input_files/Ni_surface.in",
    "nexus_argon_scf.in":     TEST_DIR / "test_pwscf_input_files/nexus_argon_scf.in",
    "nexus_h2_relax.in":      TEST_DIR / "test_pwscf_input_files/nexus_h2_relax.in",
    "nexus_argon_bands.in":   TEST_DIR / "test_pwscf_input_files/nexus_argon_bands.in",
    "nexus_hubbard_e2_e3.in": TEST_DIR / "test_pwscf_input_files/nexus_hubbard_e2_e3.in",
    "TiO2_band_structure.in": TEST_DIR / "test_pwscf_input_files/TiO2_band_structure.in",
    "TiO2_relax_freeze.in":   TEST_DIR / "test_pwscf_input_files/TiO2_relax_freeze.in",
    "VO2_M1_afm.in":          TEST_DIR / "test_pwscf_input_files/VO2_M1_afm.in",
    "VO2_M1_afm.xsf":         TEST_DIR / "test_pwscf_input_files/VO2_M1_afm.xsf",
    "WSe2_band_structure.in": TEST_DIR / "test_pwscf_input_files/WSe2_band_structure.in",
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
#end for

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
    assert(calc_nml.params.keys() == ref_nml.params.keys())
    for key, value_1 in calc_nml.params.items():
        value_2 = ref_nml.params[key]
        assert(type(value_1) is type(value_2))
        if isinstance(value_1, np.ndarray):
            np.testing.assert_allclose(value_1, value_2)
        else:
            assert(value_1 == value_2)

def assert_card_equals[T: CardBase](calc_card: T, ref_card: T) -> None:
    assert(type(calc_card) is type(ref_card))
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
        assert(len(calc_card.hubbard_params) == len(ref_card.hubbard_params))
        # The classes make equality checks a bit difficult, so we check instead
        # that their output is the same.
        hub1 = set(calc_card.write().splitlines())
        hub2 = set(ref_card.write().splitlines())
        assert(hub1 == hub2)

    elif isinstance(calc_card, Occupations) and isinstance(ref_card, Occupations):
        np.testing.assert_allclose(calc_card.occupations, ref_card.occupations)
        assert(type(calc_card.minority_spin_occs) is type(ref_card.minority_spin_occs))
        if calc_card.minority_spin_occs is not None:
            np.testing.assert_allclose(calc_card.minority_spin_occs, ref_card.minority_spin_occs)

    elif isinstance(calc_card, AtomicForces) and isinstance(ref_card, AtomicForces):
        assert(calc_card.atoms == ref_card.atoms)
        np.testing.assert_allclose(calc_card.forces, ref_card.forces)

    elif isinstance(calc_card, Constraints) and isinstance(ref_card, Constraints):
        assert(calc_card.tolerance == ref_card.tolerance)
        # The classes make equality checks a bit difficult, so we check instead
        # that their output is the same.
        con1 = set(calc_card.write().splitlines())
        con2 = set(ref_card.write().splitlines())
        assert(con1 == con2)

def assert_pwscf_input_equals(calc_pwi: PwscfInput, ref_pwi: PwscfInput) -> None:
    for nml in PwscfInput.namelists:
        assert(hasattr(calc_pwi, nml))
        assert(hasattr(ref_pwi, nml))
        calc_nml = getattr(calc_pwi, nml)
        ref_nml = getattr(ref_pwi, nml)
        assert(type(calc_nml) is type(ref_nml))
        if not (calc_nml is None and ref_nml is None):
            assert_card_equals(calc_nml, ref_nml)

    for card in PwscfInput.cards:
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
                [0.0,  0.0,  0.0],
                [0.25, 0.0,  0.0],
                [0.0,  0.25, 0.0],
                [0.25, 0.25, 0.0],
                [0.0,  0.0,  0.25],
                [0.25, 0.0,  0.25],
                [0.0,  0.25, 0.25],
                [0.25, 0.25, 0.25],
            ],
            dtype=float,
        ),
        weights=np.ones((8,), dtype=int),
    )
    calc_kpoints = KPoints.from_phys_sys(phys_sys, as_crystal=False)
    assert_card_equals(calc_kpoints, ref_kpoints_tpiba)


    ref_kpoints_gamma = KPoints(specifier="gamma")
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
        hubbard_params = [
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
