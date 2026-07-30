##################################################################
##  (c) Copyright 2015-  by Jaron T. Krogel                     ##
##################################################################


#====================================================================#
#  physical_system.py                                                #
#    Representations of particles collected together in complete     #
#    systems.                                                        #
#                                                                    #
#  Content summary:                                                  #
#    PhysicalSystem                                                  #
#      Class representing electrons+ions for a simulation.           #
#                                                                    #
#    generate_physical_system                                        #
#      User function to create arbitrary physical systems.           #
#                                                                    #
#====================================================================#

from __future__ import annotations
from collections.abc import Mapping, Iterable
from copy import deepcopy
import os
from pathlib import Path
from typing import TypeAlias, Literal

import numpy as np
import numpy.typing as npt

from .developer import obj, DevBase, error, warn
from .periodic_table import Elements, ElementLike
from .structure import Structure, generate_structure, read_structure

LabelNumMap: TypeAlias = Mapping[str, int | float | npt.NDArray[np.floating]]
"""Mapping (e.g. ``dict`` or ``obj``) from an ion label to a number or array."""

ElemNumMap: TypeAlias = Mapping[ElementLike, int | float | npt.NDArray[np.floating]]
"""Mapping (e.g. ``dict`` or ``obj``) from an element to a number or array."""


def _as_int_if_close(value: int | float, tol: float = 1e-8) -> int | float:
    """Return an int if value is close to an integer within a given tolerance."""
    if isinstance(value, int):
        return value # Early exit
    elif abs((int_value := round(value)) - value) < tol:
        return int_value
    else:
        return value


class Electrons:
    """Class representing a collection of electrons

    Note that this class does not make guarantees about having an
    integer amount of electrons, but provides ``is_fractional`` to check
    for non-integer numbers of electrons. If the class is created with
    integer-value floats for ``count`` and ``spin``, they will be
    converted into ints.

    Attributes
    ----------
    count : int or float, property
        The total number of electrons.
    spin : int or float, property
        The total spin of the system.

        An up-spin electron has a spin of +1, a down-spin electron has
        a spin of -1.
    n_up : int or float, read-only property
        The number of up-spin electrons.
        Not defined for spin-orbit systems.
    n_down : int or float, read-only property
        The number of down-spin electrons.
        Not defined for spin-orbit systems.
    total_charge : int or float, read-only property
        The total charge of the electrons, equal to
        ``self.unit_charge * self.count``.
    multiplicity : int or float, read-only property
        The spin multiplicity of the electrons, equal to :math:`2S+1`
        where :math:`S` is the spin.
    """
    
    @property
    def unit_charge(self) -> Literal[-1]:
        return -1

    @property
    def unit_mass(self) -> float:
        return 1.0

    def __init__(
        self,
        count     : int | float,
        n_unpaired: int | float,
        *,
        spin_orbit: bool = False,
        ):
        self.count      = count
        self.n_unpaired = n_unpaired
        self.spin_orbit = spin_orbit

    @property
    def count(self) -> int | float:
        return self._count

    @count.setter
    def count(self, new_count: int | float) -> None:
        self._count = _as_int_if_close(new_count)

    @property
    def n_unpaired(self) -> int | float:
        """The number of unpaired electrons.

        This is essentially the number of up-spin electrons minus the
        number of down-spin electrons.
        """
        return self._n_unpaired

    @n_unpaired.setter
    def n_unpaired(self, n_unpaired: int | float) -> None:
        self._n_unpaired = _as_int_if_close(n_unpaired)

    @property
    def spin(self) -> int | float:
        """Alias for ``self.n_unpaired``."""
        return self.n_unpaired

    @property
    def quantum_spin(self) -> int | float:
        """The spin quantum number of the electrons collection.

        This uses the definition of spin where up-spin is +1/2 and
        down-spin is -1/2.
        """
        return _as_int_if_close(self.n_unpaired / 2)

    @quantum_spin.setter
    def quantum_spin(self, new_spin: int | float) -> None:
        self.n_unpaired = new_spin * 2 # Delegate setting to n_unpaired

    def n_up_down(self) -> tuple[int, int] | tuple[float, float]:
        """Return a tuple representing the number of up- and down-spin electrons.

        Examples
        --------
        >>> Electrons(count=16, n_unpaired=2).n_up_down()
        (9, 7) # (up, down)
        >>> Electrons(count=15, n_unpaired=2).n_up_down()
        (8.5, 6.5)
        >>> Electrons(count=16, n_unpaired=1).n_up_down()
        (8.5, 7.5)
        >>> Electrons(count=15, n_unpaired=1).n_up_down()
        (8, 7)
        >>> Electrons(count=15, n_unpaired=-1).n_up_down()
        (7, 8)
        """
        if self.spin_orbit:
            # Use type(self).__name__ to get name of subclass, not base class
            raise RuntimeError(
                f"{type(self).__name__} can not be split into up- and down-spin with a spin-orbit system!"
                )

        n_up   = _as_int_if_close((self.count + self.spin) / 2)
        n_down = _as_int_if_close((self.count - self.spin) / 2)

        return n_up, n_down

    @property
    def n_up(self) -> int | float:
        if self.spin_orbit:
            raise RuntimeError(
                f"Up-spin {type(self).__name__} are not defined with a spin-orbit system!"
                )
        return self.n_up_down()[0]

    @property
    def n_down(self) -> int | float:
        if self.spin_orbit:
            raise RuntimeError(
                f"Down-spin {type(self).__name__} are not defined with a spin-orbit system!"
                )
        return self.n_up_down()[1]

    def is_fractional(self) -> bool:
        """Returns ``True`` if the count of electrons is not an ``int``."""
        return isinstance(self.count, float)

    @property
    def total_charge(self) -> int | float:
        return self.unit_charge * self.count

    @property
    def multiplicity(self) -> int | float:
        """Defined as :math:`2S+1` where :math:`S` is ``self.spin``.

        Undefined for spin-orbit systems and fractional counts.
        """
        if self.spin_orbit:
            raise RuntimeError("Multiplicity is undefined for spin-orbit systems!")
        elif self.is_fractional():
            raise RuntimeError("Multiplicity is undefined for fractional counts!")
        else:
            return abs(self.n_unpaired) + 1

    @classmethod
    def neutralize_to(
        cls,
        ions      : Iterable[IonSpecies],
        net_charge: int | float,
        n_unpaired: int | float | None = None,
        *,
        spin_orbit: bool = False,
    ) -> Electrons:
        """Neutralize the charge of ``ions`` to ``net_charge``.

        This will prioritize creating an integer number of electrons
        with the specified spin, however it will fall back to a
        fractional number of electrons if necessary.

        Parameters
        ----------
        ions : Iterable of IonSpecies
            A ``dict`` or ``obj`` with ``IonSpecies`` as values or a
            list of ``IonSpecies``.
        net_charge : int or float
            The desired net charge of the combined ion-electron system.
        n_unpaired : int or float, optional
            The desired total number of unpaired electrons. If this is
            not specified, then this function sets it to zero for an
            even number of electrons and one for an odd number. If the
            number of electrons is not an integer, then it sets it such
            that the number of down-spin electrons is an integer and the
            number of up-spin electrons is a float.
        spin_orbit : bool, default=False
            Specify whether or not the system is a spin-orbit system.
            Passed to the class constructor.
        """
        if isinstance(ions, Mapping):
            ions = ions.values()

        ions_charge = 0
        for ion in ions:
            ions_charge += ion.total_charge_deficit

        n_electrons = _as_int_if_close(ions_charge - net_charge)
        if n_unpaired is None:
            n_unpaired = _as_int_if_close(n_electrons % 2)

        return Electrons(
            count      = n_electrons,
            n_unpaired = n_unpaired,
            spin_orbit = spin_orbit,
            )

    def __eq__(self, other) -> bool:
        return (
            self.unit_charge    == other.unit_charge
            and self.count      == other.count
            and self.n_unpaired == other.n_unpaired
            and self.spin_orbit is other.spin_orbit
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"count={self.count}, "
            f"n_unpaired={self.n_unpaired}, "
            f"spin_orbit={self.spin_orbit})"
            )
#end class Electrons


class IonSpecies:
    """Class representing a collection of ions of the same type.

    Attributes
    ----------
    element : Elements
        The element for this ion collection.
    count : int or float
        The number of ions in this collection.
    label : str
        The label for the ion collection.
    formal_charge : int or float
        The formal charge associated with a single one of the ions.
    magnetization : int or float or NDArray
        The magnetization of a single one of the ions.
    Zeff : int or float
        The effective nuclear charge of one of the ions.
    symbol : str, read-only property
        The atomic symbol of the element.
    total_formal_charge : int or float, read-only property
        Formal charge multiplied by count.
    total_nuclear_charge : int or float, read-only property
        Zeff multiplied by count.
    electron_deficit : int or float, read-only property
        Number of electrons required to reach the formal charge.
    total_electron_deficit : int or float, read-only property
        Number of electrons required to reach the formal charge
        multiplied by count.
    total_magnetization : int or float or NDArray, read-only property
        Magnetization multiplied by count.
    total_mass : float, read-only property
        Atomic weight multiplied by count.

    Parameters
    ----------
    element : ElementLike
        A member of the ``Elements`` enum, atomic symbol, or atomic
        number.
    count : int or float
        The number of ions in this collection.
    label : str, optional
        The label for the ion. If not given, defaults to
        ``element.symbol``.
    formal_charge : int or float, default=0
        The formal charge associated with a single one of the ions.
    magnetization : int or float or ArrayLike, default=0
        The magnetization of a single one of the ions.
    Zeff : int, optional
        The effective nuclear charge of the ion. Defaults to the atomic
        number (a.k.a. all-electron).
    """

    def __init__(
        self,
        element      : ElementLike,
        count        : int | float,
        label        : str | None                  = None,
        formal_charge: int | float                 = 0,
        magnetization: int | float | npt.ArrayLike = 0,
        Zeff         : int | float | None          = None,
        ):
        self.element       = Elements(element)
        self.count         = _as_int_if_close(count)
        self.label         = label if label is not None else self.element.symbol
        self.formal_charge = _as_int_if_close(formal_charge)
        self.Zeff          = _as_int_if_close(Zeff) if Zeff is not None else self.element.atomic_number

        if isinstance(magnetization, int | float):
            self.magnetization = magnetization
        elif isinstance(magnetization, list | tuple | np.ndarray):
            self.magnetization = np.asarray(magnetization, dtype=np.float64)

    def pseudized(self) -> bool:
        """Check if this ion is pseudized."""
        return self.Zeff != self.element.atomic_number

    def pseudize(self, Zeff: int):
        """Equivalent to setting ``self.Zeff = Zeff``."""
        self.Zeff = Zeff

    def is_ghost(self) -> bool:
        """Check if this collection of ions represents ghost atoms."""
        return self.element is Elements.Unknown

    @property
    def symbol(self) -> str:
        """Atomic symbol of the element."""
        return self.element.symbol

    @property
    def total_mass(self) -> float:
        """Total mass of all ions in the collection."""
        return self.element.atomic_weight * self.count

    @property
    def charge_deficit(self) -> int | float:
        """The required number of electrons to get to the formal charge.

        Equal to the effective nuclear charge minus the formal charge.

        For example, a single all-electron oxygen (``Zeff=8``) that has 
        ``formal_charge=-1`` needs :math:`8 - (-1) = 8 + 1 = 9`
        electrons to achieve the desired format charge.
        """
        return self.Zeff - self.formal_charge

    @property
    def total_nuclear_charge(self) -> int | float:
        """Nuclear charge times count."""
        return self.Zeff * self.count

    @property
    def total_formal_charge(self) -> int | float:
        """Unit charge times count."""
        return self.formal_charge * self.count

    @property
    def total_charge_deficit(self) -> int | float:
        """Number of electrons to achieve the desired formal charge times count."""
        return self.charge_deficit * self.count

    @property
    def total_magnetization(self) -> int | float | npt.NDArray[np.floating]:
        """Total magnetization of all ions in the collection."""
        return self.magnetization * self.count

    def __eq__(self, other) -> bool:
        # Use np.all to handle cases where magnetization is an array
        return bool(np.all(
            self.element is other.element
            and self.count         == other.count
            and self.label         == other.label
            and self.formal_charge == other.formal_charge
            and self.magnetization == other.magnetization
            and self.Zeff          == other.Zeff
            ))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"element={self.element}, "
            f"count={self.count}, "
            f"label={self.label}, "
            f"formal_charge={self.formal_charge}, "
            f"magnetization={self.magnetization}, "
            f"Zeff={self.Zeff})"
            )

    def __hash__(self) -> int: # Enables making unordered sets or using as dict keys
        return hash((
            self.element,
            self.count,
            self.label,
            self.formal_charge,
            self.magnetization,
            self.Zeff,
            ))

    @classmethod
    def from_structure(
        cls,
        structure  : Structure,
        elem_charge: LabelNumMap | None = None,
        elem_mag   : LabelNumMap | None = None,
        elem_Zeff  : LabelNumMap | None = None,
        ) -> dict[str, IonSpecies]:
        """Create a dict with ``IonSpecies`` from a ``Structure`` object.

        It is important to note that this class only represents the ions
        in a structure, not any other particles (e.g. electrons). Thus,
        this will not capture the background charge of the structure if
        it is defined.

        Parameters
        ----------
        structure : Structure
            The structure from which to pull ions.
        elem_charge : Mapping[str, int or float], optional
            A dict or ``obj`` mapping the elements to formal charges.
            Defaults to 0 if not given.
        elem_mag : Mapping[str, int or float], optional
            A dict or ``obj`` mapping the elements to magnetizations.
            Defaults to 0 if not given.
        elem_Zeff : Mapping[str, int or float], optional
            A dict or ``obj`` mapping the elements to effective nuclear
            charges.
            Defaults to the atomic number if not supplied.

        Returns
        -------
        ions : dict[str, IonSpecies]
            A dictionary of the ion species found in the structure.
            The ion labels are used for the keys, sorted in alphabetical
            order.

        Examples
        --------
        Minimal call signature, populated with defaults.

        >>> structure = Structure(
        ...     elem = ["N", "C1", "C2", "O", "O", "H", "H", "H", "H", "H"],
        ...     pos = np.zeros((10,3)),
        ...     )
        >>> ions = IonSpecies.from_structure(
        ...     structure=structure,
        ...     )
        >>> for label, ion in ions.items():
        ...     print(f"{label:2}: {repr(ion)}")
        C1: IonSpecies(element=C, count=1, label=C1, formal_charge=0, magnetization=0, Zeff=6)
        C2: IonSpecies(element=C, count=1, label=C2, formal_charge=0, magnetization=0, Zeff=6)
         H: IonSpecies(element=H, count=5, label=H, formal_charge=0, magnetization=0, Zeff=1)
         N: IonSpecies(element=N, count=1, label=N, formal_charge=0, magnetization=0, Zeff=7)
         O: IonSpecies(element=O, count=2, label=O, formal_charge=0, magnetization=0, Zeff=8)

        Full call signature, all values specified.

        >>> structure = Structure(
        ...     elem = ["N", "C1", "C2", "O", "O", "H", "H", "H", "H", "H"],
        ...     pos = np.zeros((10,3)),
        ...     )
        >>> ions = IonSpecies.from_structure(
        ...     structure   = structure,
        ...     elem_charge = dict(N=3, C1=2, C2=4,   O=2, H=1  ),
        ...     elem_mag    = dict(N=1, C1=0, C2=0.5, O=0, H=0.5),
        ...     elem_Zeff   = dict(N=5, C1=4, C2=6,   O=6, H=1  ),
        ...     )
        >>> for label, ion in ions.items():
        ...     print(f"{label:2}: {repr(ion)}")
        C1: IonSpecies(element=C, count=1, label=C1, formal_charge=2, magnetization=0, Zeff=4)
        C2: IonSpecies(element=C, count=1, label=C2, formal_charge=4, magnetization=0.5, Zeff=6)
         H: IonSpecies(element=H, count=5, label=H, formal_charge=1, magnetization=0.5, Zeff=1)
         N: IonSpecies(element=N, count=1, label=N, formal_charge=3, magnetization=1, Zeff=5)
         O: IonSpecies(element=O, count=2, label=O, formal_charge=2, magnetization=0, Zeff=6)
        """
        elem_charge = dict(elem_charge) if elem_charge is not None else {}
        elem_mag = dict(elem_mag) if elem_mag is not None else {}
        elem_Zeff = dict(elem_Zeff) if elem_Zeff is not None else {}

        ions = {}
        elem_list = list(structure.elem)
        elem_set = set(elem_list)
        for label in elem_set:
            is_elem, element = Elements.is_element(label, return_element=True)
            if not is_elem:
                raise ValueError(
                    f"Can not determine element from label {label}!"
                    )

            ion = cls(
                element       = element,
                count         = elem_list.count(label),
                label         = label,
                formal_charge = elem_charge.get(label, 0),
                magnetization = elem_mag.get(label, 0),
                Zeff          = elem_Zeff.get(label, element.atomic_number),
                )
            ions[label] = ion
        # Sort so keys (ion labels) are in alphabetical order.
        # This is ever so slightly better than the random order
        # that comes from using set(elem_list) above.
        ions = {lbl: ions[lbl] for lbl in sorted(ions.keys())}
        return ions
#end class IonSpecies


class PhysicalSystem(DevBase):

    ghost_aliases = ["Xx"]

    def __init__(self,structure=None,net_charge=0,net_spin=0,**valency):
        self.pseudized = False
        if structure is None:
            self.structure = Structure()
        else:
            self.structure = structure
        #end if

        self.folded_system = None
        if self.structure.has_folded():
            if self.structure.is_tiled():
                vratio = structure.volume()/structure.folded_structure.volume()
                ncells = int(round(vratio))
                if abs(vratio-ncells)>1e-4:
                    self.error('volume of system does not divide evenly into folded system')
                #end if
                if net_charge%ncells!=0:
                    self.error('net charge of system does not divide evenly into folded system')
                #end if
                if isinstance(net_spin,str):
                    net_spin_fold = net_spin
                elif net_spin%ncells!=0:
                    self.error('net_spin of system does not divide evenly into folded system')
                else:
                    net_spin_fold = net_spin//ncells 
                #end if
                net_charge_fold = net_charge//ncells
            elif not self.structure.has_axes(): # folded molecule
                # net charge/spin are not physically meaningful
                # for a point group folded molecule
                # set them to safe values; they will not be used later
                net_charge_fold = 0
                net_spin_fold   = 'low'
            else:
                self.error('folded structure is not correctly integrated with full structure\nfolded physical system cannot be constructed')
            #end if
                
            self.folded_system = PhysicalSystem(
                structure  = structure.folded_structure,
                net_charge = net_charge_fold,
                net_spin   = net_spin_fold,
                **valency
                )
        #end if
        if valency is not None and len(valency) > 0:
            self.pseudize(**valency)
        else:
            self.valency = None
        self.net_charge = net_charge
        self.net_spin   = net_spin

        self.check_folded_system()
    #end def __init__


    def pseudize(self,**valency):
        for ion in valency.keys():
            if ion not in self.ion_labels:
                self.error(ion+' is not in the physical system',exit=False)

        self.valency = obj(**valency)
        self.pseudized = True
    #end def pseudize

        
    def check_folded_system(self,exit=True,message=False):
        msg = ''
        sys_folded    = self.folded_system is not None
        struct_folded = self.structure.folded_structure is not None
        if sys_folded!=struct_folded:
            msg+='folding of physical system and structure is not consistent\nsystem folded: {0}\nstructure folded: {1}\n'.format(sys_folded,struct_folded)
        #end if
        if sys_folded and id(self.structure.folded_structure)!=id(self.folded_system.structure):
            msg+='structure of folded system and folded structure are distinct\nthis is not allowed and may be a developer error'
        #end if
        success = len(msg)==0
        if not success and exit:
            self.error(msg)
        #end if
        if not message:
            return success
        else:
            return success,msg
        #end if
    #end def check_folded_system


    def check_consistent(self,tol=1e-8,exit=True,message=False):
        fs,fm = self.check_folded_system(exit=False,message=True)
        cs,cm = self.structure.check_consistent(tol,exit=False,message=True)
        msg = ''
        if not fs:
            msg += fm+'\n'
        #end if
        if not cs:
            msg += cm+'\n'
        #end if
        consistent = len(msg)==0
        if not consistent and exit:
            self.error(msg)
        #end if
        if not message:
            return consistent
        else:
            return consistent,msg
        #end if
    #end def check_consistent


    def is_valid(self):
        return self.check_consistent(exit=False)
    #end def is_valid


    def change_units(self,units):
        self.structure.change_units(units,folded=False)
        if self.folded_system is not None:
            self.folded_system.change_units(units)
        #end if
    #end def change_units


    def group_atoms(self):
        self.structure.group_atoms(folded=False)
        if self.folded_system is not None:
            self.folded_system.group_atoms()
        #end if
    #end def group_atoms


    def rename(self,folded=True,**name_pairs):
        self.structure.rename(folded=False,**name_pairs)
        if self.pseudized:
            for old,new in name_pairs.items():
                if old in self.valency:
                    if new not in self.valency:
                        self.valency[new] = self.valency[old]
                    #end if
                    del self.valency[old]
                #end if
            #end for
        #end if
        if self.folded_system is not None and folded:
            self.folded_system.rename(folded=folded,**name_pairs)
        #end if
    #end def rename


    def copy(self):
        cp = deepcopy(self)
        if self.folded_system is not None and self.structure.folded_structure is not None:
            del cp.folded_system.structure
            cp.folded_system.structure = cp.structure.folded_structure
        #end if
        return cp
    #end def copy


    def load(self,filepath):
        DevBase.load(self,filepath)
        if self.folded_system is not None and self.structure.folded_structure is not None:
            del self.folded_system.structure
            self.folded_system.structure = self.structure.folded_structure
        #end if
    #end def load


    def tile(self,*td,**kwargs):
        extensive = True
        net_spin  = None
        if 'extensive' in kwargs:
            extensive = kwargs['extensive']
        #end if
        if 'net_spin' in kwargs:
            net_spin = kwargs['net_spin']
        #end if
        supercell = self.structure.tile(*td)
        supercell.remove_folded()
        if extensive:
            ncells = int(round(supercell.volume()/self.structure.volume()))
            net_charge = ncells*self.net_charge
            if net_spin is None:
                net_spin = ncells*self.net_spin
            #end if
        else:
            net_charge = self.net_charge
            if net_spin is None:
                net_spin   = self.net_spin
            #end if
        #end if
        system = deepcopy(self)
        supersystem = PhysicalSystem(
            structure  = supercell,
            net_charge = net_charge,
            net_spin   = net_spin,
            **self.valency
            )
        supersystem.folded_system = system
        supersystem.structure.set_folded(system.structure)
        return supersystem
    #end def tile


    def has_folded(self):
        return self.folded_system is not None
    #end def has_folded


    def remove_folded_system(self):
        self.folded_system = None
        self.structure.remove_folded_structure()
    #end def remove_folded_system


    def remove_folded(self):
        self.remove_folded_system()
    #end def remove_folded


    def get_smallest(self):
        if self.has_folded():
            return self.folded_system
        else:
            return self
        #end if
    #end def get_smallest

    
    def is_magnetic(self):
        return self.net_spin!=0 or self.structure.is_magnetic()
    #end def is_magnetic

    
    def spin_polarized_orbitals(self):
        return self.is_magnetic()
    #end def spin_polarized_orbitals


    # test needed
    def large_Zeff_elem(self,Zmin):
        elem = []
        for atom,Zeff in self.valency.items():
            if Zeff>Zmin:
                elem.append(atom)
            #end if
        #end for
        return elem
    #end def large_Zeff_elem


    # test needed
    def ae_pp_species(self):
        species = set(self.structure.elem)
        if self.pseudized:
            pp_species = set(self.valency.keys())
            ae_species = species-pp_species
        else:
            pp_species = set()
            ae_species = species
        #end if
        return ae_species,pp_species
    #end def ae_pp_species


    def kf_rpa(self):
      nelecs = (self.n_up, self.n_down)
      volume = self.structure.volume()
      kvol1 = (2*np.pi)**3/volume  # k-space volume per particle
      kfs = [(3*nelec*kvol1/(4*np.pi))**(1./3) for nelec in nelecs]
      return np.array(kfs)
    #end def kf_rpa


    @property
    def n_elec(self):
        ions = self.structure.elem.tolist()
        tot_charge = 0
        for ion in ions:
            if self.valency is not None:
                if ion in self.valency:
                    tot_charge += self.valency[ion]
                else:
                    _, element = Elements.is_element(ion, return_element=True)
                    tot_charge += element.atomic_number
            else:
                _, element = Elements.is_element(ion, return_element=True)
                tot_charge += element.atomic_number

        return tot_charge - self.net_charge

    @property
    def n_up(self):
        return (self.n_elec + self.net_spin) // 2

    @property
    def n_down(self):
        return (self.n_elec - self.net_spin) // 2

    @property
    def n_species(self):
        return len(set(self.structure.elem))

    @property
    def n_ions(self):
        return len(self.structure.elem)

    @property
    def ion_labels(self):
        return set(self.structure.elem)

    @property
    def Zeff(self):
        if self.valency is not None:
            return self.valency

        Zeff = {}
        for ion in self.ion_labels:
            _, element = Elements.is_element(ion, return_element=True)
            Zeff[ion] = element.atomic_number

        return Zeff
#end class PhysicalSystem


ps_defaults = dict(
    type='crystal',
    kshift = (0,0,0),
    net_charge=0,
    net_spin=0,
    pretile=None,
    tiling=None,
    tiled_spin=None,
    extensive=True
    )
def generate_physical_system(**kwargs):
    for var,val in ps_defaults.items():
        if var not in kwargs:
            kwargs[var] = val
        #end if
    #end for
    type = kwargs['type']
    if type=='atom' or type=='dimer' or type=='trimer':
        del kwargs['kshift']
        del kwargs['tiling']
        #if not 'units' in kwargs:
        #    kwargs['units'] = 'B'
        ##end if
        tiling = None
    else:
        tiling = kwargs['tiling']
    #end if

    if 'structure' in kwargs:
        s = kwargs['structure']
        is_str = isinstance(s,str)
        if is_str or isinstance(s, Path):
            if os.path.exists(s):
                if 'elem' in kwargs:
                    s = read_structure(s,elem=kwargs['elem'])
                else:
                    s = read_structure(s)
                #end if
                if 'axes' in kwargs:
                    s.reset_axes(kwargs['axes'])
                #end if
                kwargs['structure'] = s
            else:
                slow = s.lower()
                format = None
                if '.' in slow:
                    format = slow.rsplit('.')[1]
                elif 'poscar' in slow:
                    format = 'poscar'
                #end if
                is_path = '/' in s
                is_file = format in set('xyz xsf poscar cif fhi-aims'.split())
                if is_path or is_file:
                    error('user provided structure file does not exist\nstructure file path: '+s,'generate_physical_system')
                #end if
            #end if
        #end if
    #end if

    generation_info = obj(**deepcopy(kwargs))

    net_charge = kwargs['net_charge']
    net_spin   = kwargs['net_spin']
    tiled_spin = kwargs['tiled_spin']
    extensive  = kwargs['extensive']
    del kwargs['net_spin']
    del kwargs['net_charge']
    del kwargs['tiled_spin']
    del kwargs['extensive']
    if 'particles' in kwargs:
        warn("The keyword `particles` is no longer valid. Please remove from your scripts!")
        del kwargs['particles']

    pretile = kwargs['pretile']
    del kwargs['pretile']
    valency = dict()
    remove = []
    for var in kwargs:
        #if var in Matter.elements:
        if Elements.is_element(var):
            valency[var] = kwargs[var]
            remove.append(var)
        #end if
    #end if
    generation_info.valency = deepcopy(valency)
    for var in remove:
        del kwargs[var]
    #end for

    if pretile is None:
        structure = generate_structure(**kwargs)
    else:
        for d in range(len(pretile)):
            if tiling[d]%pretile[d]!=0:
                error('pretile does not divide evenly into tiling\n  tiling provided: {0}\n  pretile provided: {1}'.format(tiling,pretile),'generate_physical_system')
            #end if
        #end for
        tiling = tuple(np.array(tiling)//np.array(pretile))
        kwargs['tiling'] = pretile
        pre = generate_structure(**kwargs)
        pre.remove_folded_structure()
        structure = pre.tile(tiling)
    #end if
    if isinstance(tiling, tuple):
        tiling_mat = np.diag(tiling)
    elif tiling is None:
        tiling_mat = np.eye(3)
    else:
        tiling_mat = tiling

    if not np.array_equal(tiling_mat, np.eye(3)) and structure.has_folded():
        # Has some supercell tiling
        fps = PhysicalSystem(
            structure  = structure.folded_structure,
            net_charge = net_charge,
            net_spin   = net_spin,
            **valency
            )
        structure.remove_folded()
        folded_structure = fps.structure
        if extensive:
            ncells = int(round(structure.volume()/folded_structure.volume()))
            net_charge = ncells*net_charge
            if not isinstance(net_spin,str):
                net_spin   = ncells*net_spin
            #end if
        #end if
        if tiled_spin is not None:
            net_spin = tiled_spin
        #end if
        ps = PhysicalSystem(
            structure  = structure,
            net_charge = net_charge,
            net_spin   = net_spin,
            **valency
            )
        structure.set_folded(folded_structure)
        ps.folded_system = fps
    else:
        # No supercell tiling
        ps = PhysicalSystem(
            structure  = structure,
            net_charge = net_charge,
            net_spin   = net_spin,
            **valency
            )
    #end if
    
    ps.generation_info = generation_info

    return ps
#end def generate_physical_system



# test needed
def ghost_atoms(*particles):
    for particle in particles:
        PhysicalSystem.ghost_aliases.append(particle)
#end def ghost_atoms
