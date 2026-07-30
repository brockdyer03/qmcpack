import pytest
from copy import deepcopy

from . import NexusTestOrder
pytestmark = pytest.mark.order(NexusTestOrder.PHYSICAL_SYSTEM)

from ..generic import generic_settings
generic_settings.raise_error = True

import numpy as np
from ..testing import value_eq,object_eq
from nexus.physical_system import generate_physical_system, Electrons, IonSpecies
from nexus.periodic_table import Elements
from nexus.unit_converter import convert
from nexus.structure import Structure

from .test_structure import structure_same


def sub_obj(s,keys):
    from ..developer import obj
    return obj({k:s[k] for k in keys})


def system_same(s1,s2,pseudized=True,tiled=False):
    same = True
    keys = ('net_charge','net_spin','pseudized')
    o1 = sub_obj(s1,keys)
    o2 = sub_obj(s2,keys)
    qsame = object_eq(o1,o2)
    vsame = True
    if pseudized:
        vsame = dict(**s1.valency)==dict(**s2.valency)
    #end if
    ssame = structure_same(s1.structure,s2.structure)
    fsame = True
    if tiled:
        fsame = system_same(s1.folded_system,s2.folded_system)
    #end if
    same = qsame and vsame and ssame and fsame
    return same
#end def system_same


def test_physical_system_initialization(tmp_path):
    from ..developer import obj
    from ..structure import generate_structure
    from ..physical_system import generate_physical_system
    from ..physical_system import PhysicalSystem

    d2 = generate_structure(
        structure = 'diamond',
        cell      = 'prim',
        )
    d2_path = tmp_path / 'diamond2.xsf'
    d2.write(d2_path)

    d8 = generate_structure(
        structure = 'diamond',
        cell      = 'conv',
        )
    d8_path = tmp_path / 'diamond8.xsf'
    d8.write(d8_path)


    d8_tile = d2.tile([[ 1, -1,  1],
                       [ 1,  1, -1],
                       [-1,  1,  1]])

    d8_tile_pos_ref = np.array([
        [0.  , 0.  , 0.  ],
        [0.25, 0.25, 0.25],
        [0.5 , 0.5 , 0.  ],
        [0.75, 0.75, 0.25],
        [0.  , 0.5 , 0.5 ],
        [0.25, 0.75, 0.75],
        [0.5 , 0.  , 0.5 ],
        [0.75, 0.25, 0.75]])

    assert(value_eq(d8_tile.pos_unit(),d8_tile_pos_ref,atol=1e-8))


    direct_notile = generate_physical_system(
        units = 'A',
        axes  = [[3.57, 0.00, 0.00],
                 [0.00, 3.57, 0.00],
                 [0.00, 0.00, 3.57]],
        elem  = 8*['C'],
        posu  = [[0.00, 0.00, 0.00],
                 [0.25, 0.25, 0.25],
                 [0.00, 0.50, 0.50],
                 [0.25, 0.75, 0.75],
                 [0.50, 0.00, 0.50],
                 [0.75, 0.25, 0.75],
                 [0.50, 0.50, 0.00],
                 [0.75, 0.75, 0.25]],
        C     = 4,
        )

    direct_tile = generate_physical_system(
        units  = 'A',
        axes   = [[1.785, 1.785, 0.   ],
                  [0.   , 1.785, 1.785],
                  [1.785, 0.   , 1.785]],
        elem   = 2*['C'],
        posu   = [[0.00, 0.00, 0.00],
                  [0.25, 0.25, 0.25]],
        tiling = [[ 1, -1,  1],
                  [ 1,  1, -1],
                  [-1,  1,  1]],
        C      = 4,
        )

    struct_notile = generate_physical_system(
        structure = d8,
        C         = 4,
        )

    struct_tile = generate_physical_system(
        structure = d2,
        tiling    = [[ 1, -1,  1],
                     [ 1,  1, -1],
                     [-1,  1,  1]],
        C         = 4,
        )

    read_notile = generate_physical_system(
        structure = d8_path,
        C         = 4,
        )

    read_tile = generate_physical_system(
        structure = d2_path,
        tiling    = [[ 1, -1,  1],
                     [ 1,  1, -1],
                     [-1,  1,  1]],
        C         = 4,
        )

    gen_notile = generate_physical_system(
        lattice   = 'cubic',        # cubic tetragonal orthorhombic rhombohedral
                                    # hexagonal triclinic monoclinic
        cell      = 'conventional', # primitive, conventional
        centering = 'F',            # P A B C I R F
        constants = 3.57,           # a,b,c,alpha,beta,gamma
        units     = 'A',            # A or B
        atoms     = 'C',            # species in primitive cell
        basis     = [[0,0,0],       # basis vectors (optional)
                     [.25,.25,.25]],
        C         = 4,
        )

    gen_tile = generate_physical_system(
        lattice   = 'cubic',        # cubic tetragonal orthorhombic rhombohedral
                                    # hexagonal triclinic monoclinic
        cell      = 'primitive',    # primitive, conventional
        centering = 'F',            # P A B C I R F
        constants = 3.57,           # a,b,c,alpha,beta,gamma
        units     = 'A',            # A or B
        atoms     = 'C',            # species in primitive cell
        basis     = [[0,0,0],       # basis vectors (optional)
                     [.25,.25,.25]],
        tiling    = [[ 1, -1,  1],
                     [ 1,  1, -1],
                     [-1,  1,  1]],
        C         = 4,
        )

    lookup_notile = generate_physical_system(
        structure = 'diamond',
        cell      = 'conv',
        C         = 4,
        )

    lookup_tile = generate_physical_system(
        structure = 'diamond',
        cell      = 'prim',
        tiling    = [[ 1, -1,  1],
                     [ 1,  1, -1],
                     [-1,  1,  1]],
        C         = 4,
        )

    # check direct system w/o tiling
    ref = direct_notile
    sref = ref.structure
    assert(ref.net_charge==0)
    assert(ref.net_spin==0)
    assert(ref.pseudized)
    assert(object_eq(ref.valency,obj(C=4)))
    assert(structure_same(sref,d8))
    assert(value_eq(sref.axes,3.57*np.eye(3)))
    assert(tuple(sref.bconds)==tuple('ppp'))
    assert(list(sref.elem)==8*['C'])
    assert(value_eq(tuple(sref.pos[-1]),(2.6775,2.6775,0.8925)))
    assert(sref.units=='A')

    # check direct system w/ tiling
    ref = direct_tile
    sref = ref.structure
    assert(ref.net_charge==0)
    assert(ref.net_spin==0)
    assert(ref.pseudized)
    assert(object_eq(ref.valency,obj(C=4)))
    assert(structure_same(sref,d8_tile))
    assert(value_eq(sref.axes,3.57*np.eye(3)))
    assert(tuple(sref.bconds)==tuple('ppp'))
    assert(list(sref.elem)==8*['C'])
    assert(value_eq(tuple(sref.pos[-1]),(2.6775,0.8925,2.6775)))
    assert(sref.units=='A')
    ref = direct_tile.folded_system
    sref = ref.structure
    assert(ref.net_charge==0)
    assert(ref.net_spin==0)
    assert(ref.pseudized)
    assert(object_eq(ref.valency,obj(C=4)))
    assert(structure_same(sref,d2))
    assert(value_eq(sref.axes,1.785*np.array([[1.,1,0],[0,1,1],[1,0,1]])))
    assert(tuple(sref.bconds)==tuple('ppp'))
    assert(list(sref.elem)==2*['C'])
    assert(value_eq(tuple(sref.pos[-1]),(0.8925,0.8925,0.8925)))
    assert(sref.units=='A')


    ref_notile = direct_notile
    ref_tile   = direct_tile

    assert(system_same(struct_notile,ref_notile))
    assert(system_same(read_notile  ,ref_notile))
    assert(system_same(gen_notile   ,ref_notile))
    assert(system_same(lookup_notile,ref_notile))

    assert(system_same(struct_tile,ref_tile,tiled=True))
    assert(system_same(read_tile  ,ref_tile,tiled=True))
    assert(system_same(gen_tile   ,ref_tile,tiled=True))
    assert(system_same(lookup_tile,ref_tile,tiled=True))

    systems_notile = [
        direct_notile,
        struct_notile,
        read_notile  ,
        gen_notile   ,
        lookup_notile,
        ]
    systems_tile = [
        direct_tile,
        struct_tile,
        read_tile  ,
        gen_tile   ,
        lookup_tile,
        ]
    systems = systems_notile+systems_tile
    for sys in systems:
        assert(sys.is_valid())
    #end for

    # test has_folded
    for sys in systems_notile:
        assert(not sys.has_folded())
    #end for
    for sys in systems_tile:
        assert(sys.has_folded())
    #end for

    # test copy
    for sys in systems:
        c = deepcopy(sys)
        assert(id(c)!=id(sys))
        assert(c.is_valid())
        assert(system_same(c,sys,tiled=sys.has_folded()))
    #end for

    # test load
    for i,sys in enumerate(systems):
        path = tmp_path / 'system_{}'.format(i)
        sys.save(path)
        sys2 = PhysicalSystem()
        sys2.load(path)
        assert(sys2.is_valid())
        assert(system_same(sys2,sys,tiled=sys.has_folded()))
    #end for

    # test particle counts
    assert(direct_notile.n_ions    ==  8)
    assert(direct_notile.n_species ==  1)
    assert(direct_notile.n_elec    == 32)
    assert(direct_notile.n_up      == 16)
    assert(direct_notile.n_down    == 16)
#end def test_physical_system_initialization


def test_change_units():
    from ..physical_system import generate_physical_system
    
    sys = generate_physical_system(
        units = 'A',
        axes  = [[3.57, 0.00, 0.00],
                 [0.00, 3.57, 0.00],
                 [0.00, 0.00, 3.57]],
        elem  = 8*['C'],
        posu  = [[0.00, 0.00, 0.00],
                 [0.25, 0.25, 0.25],
                 [0.00, 0.50, 0.50],
                 [0.25, 0.75, 0.75],
                 [0.50, 0.00, 0.50],
                 [0.75, 0.25, 0.75],
                 [0.50, 0.50, 0.00],
                 [0.75, 0.75, 0.25]],
        C     = 4,
        )

    s = sys.structure

    assert(value_eq(s.pos[-1],np.array([2.6775,2.6775,0.8925])))
    sys.change_units('B')
    assert(value_eq(s.pos[-1],np.array([5.05974172,5.05974172,1.68658057])))
#end def test_change_units   



def test_rename():
    from ..developer import obj
    from ..physical_system import generate_physical_system

    sys = generate_physical_system(
        units  = 'A',
        axes   = [[1.785, 1.785, 0.   ],
                  [0.   , 1.785, 1.785],
                  [1.785, 0.   , 1.785]],
        elem   = ['C1','C2'],
        posu   = [[0.00, 0.00, 0.00],
                  [0.25, 0.25, 0.25]],
        tiling = [[ 1, -1,  1],
                  [ 1,  1, -1],
                  [-1,  1,  1]],
        C1     = 4,
        C2     = 4,
        )

    ref = sys
    assert(object_eq(ref.valency,obj(C1=4,C2=4)))
    assert(list(ref.structure.elem)==4*['C1','C2'])
    assert(ref.n_ions==8)
    assert(ref.n_species==2)
    ref = sys.folded_system
    assert(object_eq(ref.valency,obj(C1=4,C2=4)))
    assert(list(ref.structure.elem)==['C1','C2'])
    assert(ref.n_ions==2)
    assert(ref.n_species==2)

    sys.rename(C1='C',C2='C')

    ref = sys
    assert(object_eq(ref.valency,obj(C=4)))
    assert(list(ref.structure.elem)==8*['C'])
    assert(ref.n_ions==8)
    assert(ref.n_species==1)
    ref = sys.folded_system
    assert(object_eq(ref.valency,obj(C=4)))
    assert(list(ref.structure.elem)==2*['C'])
    assert(ref.n_ions==2)
    assert(ref.n_species==1)

#end def test_rename



def test_tile():
    from ..physical_system import generate_physical_system

    d2_ref = generate_physical_system(
        units  = 'A',
        axes   = [[1.785, 1.785, 0.   ],
                  [0.   , 1.785, 1.785],
                  [1.785, 0.   , 1.785]],
        elem   = 2*['C'],
        posu   = [[0.00, 0.00, 0.00],
                  [0.25, 0.25, 0.25]],
        C      = 4,
        )

    d8_ref = generate_physical_system(
        units  = 'A',
        axes   = [[1.785, 1.785, 0.   ],
                  [0.   , 1.785, 1.785],
                  [1.785, 0.   , 1.785]],
        elem   = 2*['C'],
        posu   = [[0.00, 0.00, 0.00],
                  [0.25, 0.25, 0.25]],
        tiling = [[ 1, -1,  1],
                  [ 1,  1, -1],
                  [-1,  1,  1]],
        C      = 4,
        )

    d8 = d2_ref.tile([[ 1, -1,  1],
                      [ 1,  1, -1],
                      [-1,  1,  1]])

    assert(system_same(d8,d8_ref,tiled=True))
#end def test_tile



def test_kf_rpa():
    from .test_structure import example_structure_h4
    s1 = example_structure_h4()
    ps = generate_physical_system(
        structure = s1,
        net_charge = 1,
        net_spin = 1,
        H = 1
        )
    kfs = ps.kf_rpa()
    assert np.isclose(kfs[0], 1.465, atol=1e-3)
    assert np.isclose(kfs[1], 1.465/2**(1./3), atol=1e-3)
#end def test_kf_rpa


def test_electrons():
    ref_charge       = -16
    ref_multiplicity = 3
    ref_n_up         = 9
    ref_n_down       = 7

    electrons = Electrons(count=16, n_unpaired=2, spin_orbit=False)

    assert(electrons.total_charge    == ref_charge)
    assert(electrons.multiplicity    == ref_multiplicity)
    assert(electrons.n_up            == ref_n_up)
    assert(electrons.n_down          == ref_n_down)
    assert(not electrons.is_fractional())

    ref_charge       = -15
    ref_multiplicity = 3
    ref_n_up         = 8.5
    ref_n_down       = 6.5

    electrons = Electrons(count=15, n_unpaired=2, spin_orbit=False)

    assert(electrons.total_charge == ref_charge)
    assert(not electrons.is_fractional())
    assert(electrons.multiplicity == ref_multiplicity)
    assert(electrons.n_up         == ref_n_up)
    assert(electrons.n_down       == ref_n_down)

    ref_charge       = -16
    ref_multiplicity = 2
    ref_n_up         = 8.5
    ref_n_down       = 7.5

    electrons = Electrons(count=16, n_unpaired=1, spin_orbit=False)

    assert(electrons.total_charge == ref_charge)
    assert(not electrons.is_fractional())
    assert(electrons.multiplicity == ref_multiplicity)
    assert(electrons.n_up         == ref_n_up)
    assert(electrons.n_down       == ref_n_down)

    ref_charge       = -15
    ref_multiplicity = 2
    ref_n_up         = 8
    ref_n_down       = 7

    electrons = Electrons(count=15, n_unpaired=1, spin_orbit=False)

    assert(electrons.total_charge == ref_charge)
    assert(not electrons.is_fractional())
    assert(electrons.multiplicity == ref_multiplicity)
    assert(electrons.n_up         == ref_n_up)
    assert(electrons.n_down       == ref_n_down)

    ref_charge       = -15
    ref_multiplicity = 2
    ref_n_up         = 7
    ref_n_down       = 8

    electrons = Electrons(count=15, n_unpaired=-1, spin_orbit=False)

    assert(electrons.total_charge == ref_charge)
    assert(not electrons.is_fractional())
    assert(electrons.multiplicity == ref_multiplicity)
    assert(electrons.n_up         == ref_n_up)
    assert(electrons.n_down       == ref_n_down)
#end def test_electrons


def test_electrons_eq():
    ref_charge       = -16
    ref_multiplicity = 3
    ref_n_up         = 9
    ref_n_down       = 7

    electrons1 = Electrons(count=16, n_unpaired=2, spin_orbit=False)

    assert(electrons1.total_charge == ref_charge)
    assert(electrons1.multiplicity == ref_multiplicity)
    assert(electrons1.n_up         == ref_n_up)
    assert(electrons1.n_down       == ref_n_down)
    assert(not electrons1.is_fractional())

    electrons2 = Electrons(count=16, n_unpaired=2, spin_orbit=False)

    assert(electrons2.total_charge == ref_charge)
    assert(electrons2.multiplicity == ref_multiplicity)
    assert(electrons2.n_up         == ref_n_up)
    assert(electrons2.n_down       == ref_n_down)
    assert(not electrons2.is_fractional())

    assert(electrons1 == electrons2)
#end def test_electrons_eq


def test_electrons_repr():
    ref_repr = "Electrons(count=16, n_unpaired=2, spin_orbit=False)"
    electrons = Electrons(count=16, n_unpaired=2, spin_orbit=False)
    assert(repr(electrons) == ref_repr)
#end def test_electrons_repr


def test_custom_ion_species():
    ion = IonSpecies(
        element       = Elements.Iron,
        count         = 12,
        label         = "Fe1",
        formal_charge = 2,
        magnetization = 1,
        Zeff          = 16,
        )

    assert(ion.element              is Elements.Iron)
    assert(ion.count                == 12)
    assert(ion.label                == "Fe1")
    assert(ion.formal_charge        == 2)
    assert(ion.magnetization        == 1)
    assert(ion.Zeff                 == 16)
    assert(ion.pseudized()          is True)
    assert(ion.is_ghost()           is False)
    assert(ion.symbol               == "Fe")
    assert(ion.total_mass           == 670.14)
    assert(ion.total_magnetization  == 12)
    assert(ion.total_charge_deficit == 168)
#end def test_custom_ion_species


def test_minimal_ion_species():
    """Test to make sure the defaults are populated correctly."""
    ion = IonSpecies(element="Fe", count=12)

    assert(ion.element              is Elements.Iron)
    assert(ion.label                == "Fe")
    assert(ion.count                == 12)
    assert(ion.formal_charge        == 0)
    assert(ion.magnetization        == 0)
    assert(ion.Zeff                 == Elements.Iron.atomic_number)
    assert(ion.pseudized()          is False)
    assert(ion.is_ghost()           is False)
    assert(ion.symbol               == Elements.Iron.symbol)
    assert(ion.total_mass           == 670.14)
    assert(ion.total_magnetization  == 0)
    assert(ion.total_charge_deficit == 312)
#end def test_minimal_ion_species


def test_ion_species_eq():
    ref_element               = Elements.Iron
    ref_label                 = "Fe1"
    ref_count                 = 12
    ref_formal_charge         = 0
    ref_magnetization         = 0
    ref_Zeff                  = Elements.Iron.atomic_number
    ref_is_pseudo             = False
    ref_is_ghost              = False
    ref_symbol                = Elements.Iron.symbol
    ref_total_mass            = 670.14
    ref_total_charge_deficit  = 312
    ref_total_spin            = 0

    ion1 = IonSpecies(element="Fe", label="Fe1", count=12)

    assert(ion1.element               is ref_element)
    assert(ion1.label                 == ref_label)
    assert(ion1.count                 == ref_count)
    assert(ion1.formal_charge         == ref_formal_charge)
    assert(ion1.magnetization         == ref_magnetization)
    assert(ion1.Zeff                  == ref_Zeff)
    assert(ion1.pseudized()           is ref_is_pseudo)
    assert(ion1.is_ghost()            is ref_is_ghost)
    assert(ion1.symbol                == ref_symbol)
    assert(ion1.total_mass            == ref_total_mass)
    assert(ion1.total_charge_deficit  == ref_total_charge_deficit)
    assert(ion1.total_magnetization   == ref_total_spin)

    ion2 = IonSpecies(element=Elements.Iron, label="Fe1", count=12)

    assert(ion2.element               is ref_element)
    assert(ion2.label                 == ref_label)
    assert(ion2.count                 == ref_count)
    assert(ion2.formal_charge         == ref_formal_charge)
    assert(ion2.magnetization         == ref_magnetization)
    assert(ion2.Zeff                  == ref_Zeff)
    assert(ion2.pseudized()           is ref_is_pseudo)
    assert(ion2.is_ghost()            is ref_is_ghost)
    assert(ion2.symbol                == ref_symbol)
    assert(ion2.total_mass            == ref_total_mass)
    assert(ion2.total_charge_deficit  == ref_total_charge_deficit)
    assert(ion2.total_magnetization   == ref_total_spin)

    assert(ion1 == ion2)
#end def test_ion_species_eq


def test_ion_species_repr():
    ref_repr = "IonSpecies(element=Fe, count=12, label=Fe, formal_charge=0, magnetization=0, Zeff=26)"
    ion = IonSpecies(element="Fe", count=12)
    assert(repr(ion) == ref_repr)
#end def test_ion_species_repr


def test_ion_species_hash():
    ion1 = IonSpecies(
        element       = Elements.Carbon,
        count         = 1,
        label         = "C2",
        formal_charge = 0,
        magnetization     = 0,
        Zeff          = 6,
        )
    ion2 = IonSpecies(
        element       = "C", # Use a slightly different element specifier to make sure it resolves.
        count         = 1,
        label         = "C2",
        formal_charge = 0,
        magnetization     = 0,
        Zeff          = 6,
        )
    assert(ion1 == ion2) # Make sure they're actually equal.
    assert(hash(ion1) == hash(ion2))
#end def test_ion_species_hash


def test_ion_species_from_structure():
    ref_ions = {
        "C1": IonSpecies(element=Elements.Carbon,   count=1, label="C1", formal_charge=0, magnetization=0, Zeff=6),
        "C2": IonSpecies(element=Elements.Carbon,   count=1, label="C2", formal_charge=0, magnetization=0, Zeff=6),
        "H" : IonSpecies(element=Elements.Hydrogen, count=5, label="H",  formal_charge=0, magnetization=0, Zeff=1),
        "N" : IonSpecies(element=Elements.Nitrogen, count=1, label="N",  formal_charge=0, magnetization=0, Zeff=7),
        "O" : IonSpecies(element=Elements.Oxygen,   count=2, label="O",  formal_charge=0, magnetization=0, Zeff=8),
        }

    structure = Structure(
        elem = ["N", "C1", "C2", "O", "O", "H", "H", "H", "H", "H"],
        pos = np.zeros((10,3)),
        )
    ions = IonSpecies.from_structure(
        structure=structure,
        )

    assert(ions == ref_ions)

    ref_ions = {
        "C1": IonSpecies(element=Elements.Carbon,   count=1, label="C1", formal_charge=0, magnetization=0,   Zeff=4),
        "C2": IonSpecies(element=Elements.Carbon,   count=1, label="C2", formal_charge=0, magnetization=0.5, Zeff=6),
        "H" : IonSpecies(element=Elements.Hydrogen, count=5, label="H",  formal_charge=0, magnetization=0.5, Zeff=1),
        "N" : IonSpecies(element=Elements.Nitrogen, count=1, label="N",  formal_charge=0, magnetization=1,   Zeff=5),
        "O" : IonSpecies(element=Elements.Oxygen,   count=2, label="O",  formal_charge=0, magnetization=0,   Zeff=6),
        }

    structure = Structure(
        elem = ["N", "C1", "C2", "O", "O", "H", "H", "H", "H", "H"],
        pos = np.zeros((10,3)),
        )
    ions = IonSpecies.from_structure(
        structure   = structure,
        elem_charge = dict(N=0, C1=0, C2=0,   O=0, H=0),
        elem_mag    = dict(N=1, C1=0, C2=0.5, O=0, H=0.5),
        elem_Zeff   = dict(N=5, C1=4, C2=6,   O=6, H=1),
        )

    assert(ions == ref_ions)
#end def test_ion_species_from_structure


def test_electrons_neutralize_to():
    ions = {
        "C1": IonSpecies(element=Elements.Carbon,   count=1, label="C1", formal_charge=0, magnetization=0,   Zeff=4),
        "C2": IonSpecies(element=Elements.Carbon,   count=1, label="C2", formal_charge=0, magnetization=0.5, Zeff=6),
        "H" : IonSpecies(element=Elements.Hydrogen, count=5, label="H",  formal_charge=0, magnetization=0.5, Zeff=1),
        "N" : IonSpecies(element=Elements.Nitrogen, count=1, label="N",  formal_charge=0, magnetization=1,   Zeff=5),
        "O" : IonSpecies(element=Elements.Oxygen,   count=2, label="O",  formal_charge=0, magnetization=0,   Zeff=6),
        }

    ref_electrons = Electrons(count=32, n_unpaired=0, spin_orbit=False)

    electrons = Electrons.neutralize_to(
        ions       = ions,
        net_charge = 0,
        n_unpaired = 0,
        spin_orbit = False,
        )
    assert(electrons == ref_electrons)
    assert(electrons.n_up   == 16)
    assert(electrons.n_down == 16)
    assert(isinstance(electrons.count, int))
    assert(isinstance(electrons.spin,  int))

    ref_electrons = Electrons(count=31, n_unpaired=1, spin_orbit=False)

    electrons = Electrons.neutralize_to(
        ions       = ions,
        net_charge = 1,
        n_unpaired = 1,
        spin_orbit = False,
        )
    assert(electrons == ref_electrons)
    assert(electrons.n_up   == 16)
    assert(electrons.n_down == 15)
    assert(isinstance(electrons.count, int))
    assert(isinstance(electrons.spin,  int))

    # Test with integer-value floats for count and spin.
    # Should be turned into ints.
    ref_electrons = Electrons(count=32, n_unpaired=2, spin_orbit=False)

    electrons = Electrons.neutralize_to(
        ions       = ions,
        net_charge = 0.0,
        n_unpaired = 2.0,
        spin_orbit = False,
        )
    assert(electrons == ref_electrons)
    assert(electrons.n_up   == 17)
    assert(electrons.n_down == 15)
    assert(isinstance(electrons.count, int))
    assert(isinstance(electrons.spin,  int))

    # See if it can make the right spin with floats
    ref_electrons = Electrons(count=31.5, n_unpaired=1.5, spin_orbit=False)

    electrons = Electrons.neutralize_to(
        ions       = ions,
        net_charge = 0.5,
        n_unpaired = None,
        spin_orbit = False,
        )
    assert(electrons == ref_electrons)
    assert(electrons.n_up   == 16.5)
    assert(electrons.n_down == 15)
    assert(isinstance(electrons.count, float))
    assert(isinstance(electrons.spin,  float))
#end def test_electrons_neutralize_to