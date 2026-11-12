from copy import deepcopy

import numpy as np
import pytest

from ..developer import obj
from ..physical_system import generate_physical_system
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
from ..structure import read_structure
from ..testing import object_eq
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

def assert_namelist_equals[T: NamelistBase](nml1: T, nml2: T) -> None:
    assert(type(nml1) is type(nml2))
    assert(nml1.params.keys() == nml2.params.keys())
    for key, value_1 in nml1.params.items():
        value_2 = nml2.params[key]
        assert(type(value_1) is type(value_2))
        if isinstance(value_1, np.ndarray):
            np.testing.assert_allclose(value_1, value_2)
        else:
            assert(value_1 == value_2)

def assert_card_equals[T: CardBase](card1: T, card2: T) -> None:
    assert(type(card1) is type(card2))
    if isinstance(card1, AtomicSpecies) and isinstance(card2, AtomicSpecies):
        assert(card1.atoms == card2.atoms)
        assert(card1.masses == card2.masses)
        assert(card1.pseudos == card2.pseudos)

    elif isinstance(card1, AtomicPositions) and isinstance(card2, AtomicPositions):
        assert(card1.specifier == card2.specifier)
        assert(card1.atoms == card2.atoms)
        np.testing.assert_allclose(card1.positions, card2.positions, rtol=1e-6)
        assert(type(card1.force_multipliers) is type(card2.force_multipliers))
        if card1.force_multipliers is not None:
            np.testing.assert_allclose(card1.force_multipliers, card2.force_multipliers)

    elif isinstance(card1, KPoints) and isinstance(card2, KPoints):
        assert(card1.specifier == card2.specifier)
        if card1.specifier == "automatic":
            np.testing.assert_array_equal(card1.grid, card2.grid)
            np.testing.assert_array_equal(card1.shift, card2.shift)
        elif card1.specifier != "gamma":
            assert(card1.nkpoints == card2.nkpoints)
            np.testing.assert_allclose(card1.kpoints, card2.kpoints, rtol=1e-6)
            assert(type(card1.weights) is type(card2.weights))
            if card1.weights is not None:
                np.testing.assert_allclose(card1.weights, card2.weights)

    elif isinstance(card1, CellParameters) and isinstance(card2, CellParameters):
        assert(card1.specifier == card2.specifier)
        np.testing.assert_allclose(card1.axes, card2.axes)

    elif isinstance(card1, Hubbard) and isinstance(card2, Hubbard):
        assert(card1.specifier == card2.specifier)
        assert(len(card1.hubbard_params) == len(card2.hubbard_params))
        # The classes make equality checks a bit difficult, so we check instead
        # that their output is the same.
        hub1 = set(card1.write().splitlines())
        hub2 = set(card2.write().splitlines())
        assert(hub1 == hub2)

    elif isinstance(card1, Occupations) and isinstance(card2, Occupations):
        np.testing.assert_allclose(card1.occupations, card2.occupations)
        assert(type(card1.minority_spin_occs) is type(card2.minority_spin_occs))
        if card1.minority_spin_occs is not None:
            np.testing.assert_allclose(card1.minority_spin_occs, card2.minority_spin_occs)

    elif isinstance(card1, AtomicForces) and isinstance(card2, AtomicForces):
        assert(card1.atoms == card2.atoms)
        np.testing.assert_allclose(card1.forces, card2.forces)

    elif isinstance(card1, Constraints) and isinstance(card2, Constraints):
        assert(card1.tolerance == card2.tolerance)
        # The classes make equality checks a bit difficult, so we check instead
        # that their output is the same.
        con1 = set(card1.write().splitlines())
        con2 = set(card2.write().splitlines())
        assert(con1 == con2)

def assert_pwscf_input_equals(pwi1: PwscfInput, pwi2: PwscfInput) -> None:
    for nml in PwscfInput.namelists:
        assert(hasattr(pwi1, nml))
        assert(hasattr(pwi2, nml))
        pwi1_nml = getattr(pwi1, nml)
        pwi2_nml = getattr(pwi2, nml)
        assert(type(pwi1_nml) is type(pwi2_nml))
        if not (pwi1_nml is None and pwi2_nml is None):
            assert_card_equals(pwi1_nml, pwi2_nml)

    for card in PwscfInput.cards:
        assert(hasattr(pwi1, card))
        assert(hasattr(pwi2, card))
        pwi1_card = getattr(pwi1, card)
        pwi2_card = getattr(pwi2, card)
        assert(type(pwi1_card) is type(pwi2_card))
        if not (pwi1_card is None and pwi2_card is None):
            assert_card_equals(pwi1_card, pwi2_card)


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
        pseudos = obj(Fe='Fe.pbe-nd-rrkjus.UPF'),
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
        pseudos = obj(Ar='Ar.nexus-test.UPF'),
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
        pseudos = obj(H='H.nexus-test.UPF'),
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
        pseudos = obj(Ar='Ar.nexus-test.UPF'),
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
            Ce = 'Ce.nexus-test.UPF',
            Fe = 'Fe.nexus-test.UPF',
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
#end def test_input


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
