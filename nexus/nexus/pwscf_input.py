##################################################################
##  (c) Copyright 2015-  by Jaron T. Krogel                     ##
##################################################################

"""Supports I/O for PWSCF input files.

PwscfInput
    SimulationInput class for PWSCF.
    Can read/write most PWSCF input files.

generate_pwscf_input
    User-facing function to create arbitrary input files.

Element
    Base class for two different types of PWSCF input sections:
    namelists (Section class) and 'cards' (Card class)

Section
    Class representing keyword (namelist) input sections.
    Base class for other keyword input section classes.
    See control, system, electrons, ions, cell, phonon, and ee.

Card
    Class representing formatted (card) input sections.
    Base class for other formatted section classes.
    See atomic_species, atomic_positions, k_points,
      cell_parameters, climbing_images, constraints,
      collective_vars, occupations and hubbard.
"""
from __future__ import annotations

import difflib
import inspect
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from functools import cache, cached_property
from itertools import batched, chain, groupby, product
from os import PathLike
from pathlib import Path
from re import Pattern
from textwrap import indent
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Self, overload

import numpy as np
import numpy.typing as npt
from numpy import pi
from numpy.linalg import inv

from .developer import FileFormatError, NexusError, NotAnElementError, obj, warn
from .developer_tools import unset
from .generic import nxs_deprecate
from .periodic_table import Elements
from .physical_system import PhysicalSystem
from .pseudoset import PseudoSet, pp_elem_label
from .pwscf_input_defs import (
    CellDefinitions,
    ControlDefinitions,
    ElectronsDefinitions,
    FcpDefinitions,
    IonsDefinitions,
    PwscfInputType,
    RismDefinitions,
    SystemDefinitions,
)
from .simulation import SimulationInput
from .structure import Structure, kmesh
from .unit_converter import convert

type NamelistDefType = type[
    CellDefinitions
    | ControlDefinitions
    | ElectronsDefinitions
    | FcpDefinitions
    | IonsDefinitions
    | RismDefinitions
    | SystemDefinitions
]

type NamelistName = Literal[
    "control",
    "system",
    "electrons",
    "ions",
    "cell",
    "fcp",
    "rism",
]

type CardName = Literal[
    "atomic_species",
    "atomic_positions",
    "atomic_forces",
    "k_points",
    "cell_parameters",
    "constraints",
    "occupations",
    "hubbard",
]

type AtomicPositionsSpecifier = Literal["alat", "bohr", "angstrom", "crystal"]

type KPointsSpecifier = Literal[
    "tpiba",
    "crystal",
    "tpiba_b",
    "crystal_b",
    "tpiba_c",
    "crystal_c",
    "automatic",
    "gamma"
]

type ConstraintName = Literal[
    "type_coord",
    "atom_coord",
    "distance",
    "planar_angle",
    "torsional_angle",
    "bennett_proj",
    "potential_wall",
]

type CellParametersSpecifier = Literal["alat", "bohr", "angstrom"]

type HubbardSpecifier = Literal["atomic", "ortho-atomic", "norm-atomic", "wf", "pseudo"]

type HubbardInteraction = Literal["U", "ALPHA", "J0", "J", "B", "E2", "E3", "V"]

LOGGER = logging.getLogger(__name__)

@cache
def _with_accepted_types(
    dtype: type[str | bool | int | float]
) -> set[type]:
    if dtype is float:
        return {float, int, np.floating, np.integer}
    elif dtype is int:
        return {int, np.integer}
    elif dtype is bool:
        return {bool, np.bool_}
    elif dtype is str:
        return {str, np.str_}
    else:
        msg = f"Unhandled type {dtype} found!"
        raise ValueError(msg)


class NamelistBase:
    """Base class for instances of namelists."""

    name: ClassVar[
        Literal["cell", "control", "electrons", "fcp", "ions", "rism", "system"]
    ]
    namelist_def: ClassVar[NamelistDefType]
    required: ClassVar[bool]
    params: dict[str, PwscfInputType]

    def __init__(
        self,
        *,
        check: bool = False,
        version: str | None = None,
        phys_sys: PhysicalSystem | None = None,
        verbose: bool = False,
        strict: bool = False,
        **params: PwscfInputType,
    ):
        self.params = {}
        params = self.join_array_params(params)
        for name, value in params.items():
            if check:
                self.__class__.check_param(
                    name     = name,
                    value    = value,
                    version  = version,
                    phys_sys = phys_sys,
                    verbose  = verbose,
                    strict   = strict,
                )
            self.params[name.lower()] = value

    @classmethod
    def check_param(
        cls,
        name: str,
        value: PwscfInputType,
        version: str | None = None,
        phys_sys: PhysicalSystem | None = None,
        *,
        verbose: bool = False,
        strict: bool = False,
    ) -> None:
        """Check that a parameter meets the specification from its namelist enum.

        Parameters
        ----------
        name : str
            The parameter name, case-insensitive.
        value : str or bool or int or float or Sequence or Mapping
            The value of the parameter.
        version : str, optional
            The version to check against, if it is requested.
        phys_sys : PhysicalSystem, optional
            The ``PhysicalSystem`` to check species arrays against.
        verbose : bool, default=False
            Whether or not to emit a warning for a failed check.
            Overridden by ``strict``.
        strict : bool, default=False
            Set to ``True`` to raise an error instead of emit a warning.

        Raises
        ------
        ValueError
            If the value is not known, or if it is not in the known set
            of allowed values.
        TypeError
            If the value is not of the correct type.
        """
        try:
            param_def = cls.namelist_def(name)
        except ValueError:
            msg = f"The parameter `{name}` is not known to Nexus"
            if strict:
                close_matches = difflib.get_close_matches(
                    name, cls.namelist_def.__members__
                )
                if len(close_matches) > 0:
                    msg += f", did you mean '{close_matches[0]}'?"
                raise ValueError(msg) from None
            elif verbose:
                msg += ", no checks will be performed"
                warn(msg)
                return # Return early since we can't do the rest of the checks.

        msg = (
            "The parameter `" + name + "` has the incorrect {attr}!\n"
            "Expected {attr}: {known}\n"
            "Given {attr}:    {given}"
        )

        def_shape     = param_def.shape
        def_allowed   = param_def.allowed_values
        def_v_added   = param_def.version_added
        def_v_removed = param_def.version_removed
        def_types     = _with_accepted_types(param_def.datatype)

        if def_shape is None: # Scalar, includes strings
            comp_type = type(value)
        elif isinstance(value, Mapping | Sequence | np.ndarray):
            if isinstance(value, Mapping):
                types = {type(v) for v in value.values()}
            else:
                types = {type(v) for v in value}

            if not types.issubset(def_types):
                msg = f"Value for `{name}` has inconsistent types!"
                if strict:
                    raise TypeError(msg)
                elif verbose:
                    warn(msg)
            comp_type = next(iter(types))

        if not any(issubclass(comp_type, dtype) for dtype in def_types):
            msg = msg.format(
                attr="type",
                known=", ".join(t.__name__ for t in def_types),
                given=comp_type.__name__,
            )
            if strict:
                raise TypeError(msg)
            elif verbose:
                warn(msg)

        if def_allowed is not None and value not in def_allowed:
            msg = msg.format(attr="value", known=def_allowed, given=value)
            if strict:
                raise ValueError(msg)
            elif verbose:
                warn(msg)

        if def_shape is not None:
            valid, msg = cls._check_param_shape(
                name     = name,
                value    = value,
                shape    = def_shape,
                phys_sys = phys_sys,
            )
            if not valid:
                if strict:
                    raise ValueError(msg)
                elif verbose:
                    warn(msg)

        if version is not None:
            valid, msg = cls._check_param_version(
                name    = name,
                current = version,
                added   = def_v_added,
                removed = def_v_removed,
            )
            if not valid:
                if strict:
                    raise ValueError(msg)
                elif verbose:
                    warn(msg)

    @staticmethod
    def _check_param_shape(
        *,
        name: str,
        value: Mapping | Sequence,
        shape: tuple,
        phys_sys: PhysicalSystem,
        verbose: bool = False,
    ) -> tuple[Literal[False], Literal[""]] | tuple[Literal[True], str]:
        """Check that a non-scalar parameter has the correct shape.

        Notes
        -----
        This function will take all values which are not mappings and internally
        create a 1-index mapping, since that is the form expected by QE.

        This is what the conversion looks like:

        .. code-block:: python
            value = [
                "a",
                "b",
                "c",
            ]
            internal_rep = {
                1: "a",
                2: "b",
                3: "c",
            }
        """
        valid = True
        msg = (
            "The value for `{name}` does not have the correct shape!\n"
            "Given indices: {given}\n"
            "Known shape:   {known}"
        )
        if isinstance(value, Sequence | np.ndarray):
            val_dict = {}
            for idx, val in np.ndenumerate(value):
                idx = idx[0] + 1 if len(idx) == 1 else tuple(i + 1 for i in idx)
                val_dict[idx] = val
            value = val_dict

        if (
            isinstance(shape[1], int)
            and (
                len(value) > shape[1]
                or max(value.keys()) > shape[1]
            )
        ):
            # Case: Fixed-shape arrays like `celldm`
            valid = False
            msg = msg.format(
                name  = name,
                given = tuple(value.keys()),
                known = shape,
            )

        elif isinstance(shape[1], str):
            # Case: Species arrays like `starting_charge`
            if shape[1] in {"ntyp", "ntype"} and max(value.keys()) > phys_sys.n_species:
                # old hubbard_j0 had `ntype` instead of `ntyp`
                valid = False
                msg = msg.format(
                    name  = name,
                    given = tuple(value.keys()),
                    known = (1, phys_sys.n_species),
                )
            elif shape[1] == "natx" and max(value.keys()) > phys_sys.n_ions:
                # Unused as of QE 7.6.0, but may get used in future versions
                valid = False
                msg = msg.format(
                    name  = name,
                    given = tuple(value.keys()),
                    known = (1, phys_sys.n_ions),
                )
            elif verbose:
                warn(
                    f"Shape checking for shape `{shape}` "
                    f"required by param `{name}` has not been implemented!"
                )

        elif isinstance(shape[1], tuple):
            # Case: Multidimensional arrays
            dim_msg = ""
            for dim, dim_max in enumerate(shape[1]):
                # Gets all i's together into one set
                dim_vals = {i[dim] for i in value}
                if isinstance(dim_max, int) and max(dim_vals) > dim:
                    valid = False
                    dim_msg += msg.format(
                        name  = name,
                        given = tuple(value.keys()),
                        known = dim_max,
                    ) + "\n"
                elif dim_max in {"ntyp", "ntype"} and max(dim_vals) > phys_sys.n_species:
                    valid = False
                    dim_msg += msg.format(
                        name  = name,
                        given = tuple(value.keys()),
                        known = (1, phys_sys.n_species),
                    ) + "\n"
                elif shape[1] == "natx" and max(dim_vals) > phys_sys.n_ions:
                    valid = False
                    dim_msg += msg.format(
                        name  = name,
                        given = tuple(value.keys()),
                        known = (1, phys_sys.n_ions),
                    ) + "\n"
                else:
                    wmsg = (
                        f"Shape checking for shape `{dim_max}` "
                        f"required by param `{name}` has not been implemented!"
                    )
                    dim_msg += wmsg + "\n"
                    warn(wmsg)

            msg = dim_msg

        if valid:
            return valid, ""
        else:
            return valid, msg

    @staticmethod
    def _check_param_version(
        *,
        name: str,
        current: str,
        added: str | None,
        removed: str | None,
    ) -> tuple[bool, str]:
        cur_ver = tuple(map(int, current.split(".")))
        add_ver = tuple(map(int, added.split("."))) if added is not None else None
        rem_ver = tuple(map(int, removed.split("."))) if removed is not None else None
        msg = name + " was {cond} in version {ver}, but you are using version " + current
        if rem_ver is not None and cur_ver > rem_ver:
            return False, msg.format(cond="removed", ver=removed)
        elif add_ver is not None and cur_ver < add_ver:
            return False, msg.format(cond="added", ver=removed)
        else:
            return True, ""

    def write(self) -> str:
        lines = []
        for key, value in self.params.items():
            if isinstance(value, Mapping):
                for idx, val in value.items():
                    if isinstance(val, str):
                        val = f'"{val}",'
                    else:
                        val = f"{val},"

                    if isinstance(idx, tuple):
                        idx = ",".join(map(str, idx))
                    else:
                        idx = str(idx)
                    k = f"{f"{key}({idx})"}"

                    lines.append((k, val))
                continue

            if isinstance(value, str):
                value = f'"{value}",'
            elif isinstance(value, bool):
                value = ".true.," if value else ".false.,"
            else:
                value = f"{value},"

            lines.append((key, value))
        pad = max(len(i[0]) for i in lines)

        text = f"&{self.name.upper()}\n"
        text += indent("".join(f"{i[0]:<{pad}} = {i[1]}\n" for i in lines), "    ")
        text += "/"
        return text

    def join_array_params(
        self, params: dict[str, PwscfInputType]
    ) -> dict[str, PwscfInputType]:
        """Join array parameters supplied as separate arguments.

        .. code-block:: python
            >>> params = {
            ...     'starting_ns_eigenvalue(1,2,1)' : 0.0,
            ...     'starting_ns_eigenvalue(2,2,1)' : 0.0476060,
            ...     'starting_ns_eigenvalue(3,2,1)' : 0.0476060,
            ...     'starting_ns_eigenvalue(4,2,1)' : 0.9654373,
            ...     'starting_ns_eigenvalue(5,2,1)' : 0.9954307,
            ... }
            >>> normalized = NamelistBase.normalize_params(params)
            >>> pprint.pp(normalized, indent=4, expand=True)
            {
                'starting_ns_eigenvalue': {
                    (1, 2, 1): 0.0,
                    (2, 2, 1): 0.047606,
                    (3, 2, 1): 0.047606,
                    (4, 2, 1): 0.9654373,
                    (5, 2, 1): 0.9954307,
                },
            }
        """
        normalized = {}
        for arr_name in self.array_params:
            current = {}
            matching = []
            for key in params:
                klow = key.lower()
                if (
                    key in normalized # Already handled "arr_name(1,1)", now hit "arr_name"
                    or ( # Already handled "arr_name", now hit "arr_name(1,1)"
                        klow.startswith(arr_name)
                        and arr_name in normalized
                    )
                ):
                    msg = (
                        f"Multiple formats of parameter '{arr_name}' found!\n"
                        f"Second Occurence: {key}\n"
                    )
                    raise ValueError(msg)

                if klow == arr_name: # Already in the correct form
                    continue
                elif klow.startswith(arr_name):
                    matching.append(key)

            if len(matching) == 0:
                continue

            for key in matching:
                value = params.pop(key)

                idx = key.split("(")[-1].rstrip(")").split(",")
                if len(idx) == 1:
                    idx = int(idx[0])
                else:
                    idx = tuple(map(int, idx))
                current[idx] = value

            normalized[arr_name] = current

        normalized.update(params)
        return normalized

    def incorporate_system(self, phys_sys: PhysicalSystem) -> None:
        pass

    @cached_property
    def array_params(self) -> frozenset[str]:
        return {i.name for i in self.namelist_def if i.shape is not None}

    def __repr__(self) -> str:
        text = f"{self.name.upper()}(\n"
        for name, value in self.params.items():
            if isinstance(value, str):
                value = f"'{value}'"
            text += f"    {name}={value}\n"
        text += ")"
        return text

    def __getattr__(self, name: str) -> PwscfInputType:
        if name in self.params:
            return self.params[name]

    def __setattr__(self, name: str, value: PwscfInputType) -> None:
        if (attr := name.lower()) in self.namelist_def.__members__:
            self.params[attr] = value
        else:
            super().__setattr__(name, value)

    def __contains__(self, item: str) -> bool:
        return item in vars(self) or item in self.params


class Control(NamelistBase):
    """Real instance of the ``&CONTROL`` namelist."""

    name = "control"
    namelist_def = ControlDefinitions
    required = True


class System(NamelistBase):
    """Real instance of the ``&SYSTEM`` namelist."""

    name = "system"
    namelist_def = SystemDefinitions
    required = True

    def incorporate_system(self, phys_sys: PhysicalSystem) -> None:
        if (name := "ibrav") not in self.params:
            self.params[name] = 0

        if (name := "nat") not in self.params:
            self.params[name] = phys_sys.n_ions

        if (name := "ntyp") not in self.params:
            self.params[name] = phys_sys.n_species

        if (name := "tot_charge") not in self.params:
            self.params[name] = phys_sys.net_charge


class Electrons(NamelistBase):
    """Real instance of the ``&ELECTRONS`` namelist."""

    name = "electrons"
    namelist_def = ElectronsDefinitions
    required = True


class Ions(NamelistBase):
    """Real instance of the ``&IONS`` namelist."""

    name = "ions"
    namelist_def = IonsDefinitions
    required = False


class Cell(NamelistBase):
    """Real instance of the ``&CELL`` namelist."""

    name = "cell"
    namelist_def = CellDefinitions
    required = False


class Fcp(NamelistBase):
    """Real instance of the ``&FCP`` namelist."""

    name = "fcp"
    namelist_def = FcpDefinitions
    required = False


class Rism(NamelistBase):
    """Real instance of the ``&RISM`` namelist."""

    name = "rism"
    namelist_def = RismDefinitions
    required = False


class CardBase(ABC):
    """Base class for :class:`PwscfInput` cards."""

    @classmethod
    @abstractmethod
    def read(cls, lines: list[str]) -> Self:
        """Read a card and return an instance of the card's class."""
        ...

    def post_process_read(self, pwscf_input: PwscfInput) -> None:  # noqa: B027
        pass

    @abstractmethod
    def write(self) -> str:
        ...

    @overload
    def change_specifier(
        self, new_specifier: str, pwi: PwscfInput, *, in_place: Literal[True]
    ) -> None: ...
    @overload
    def change_specifier(
        self, new_specifier: str, pwi: PwscfInput, *, in_place: Literal[False]
    ) -> Self: ...

    def change_specifier(
        self,
        new_specifier: str,
        pwi: PwscfInput,  # noqa: ARG002
        *,
        in_place: bool = True  # noqa: ARG002
    ) -> Self | None:
        msg = (
            f"Can not change to {new_specifier}, "
            f"no specifiers available for card {self.name.upper()}"
        )
        raise ValueError(msg)

    @nxs_deprecate(since="2.4.9", replacement="change_specifier")
    def change_option(self,*args,**kwargs):
        self.change_specifier(*args,**kwargs)

    @classmethod
    def from_phys_sys(self, phys_sys: PhysicalSystem | Structure, /, **kwargs: Any) -> Self:
        msg = f"Can not create {type(self).__name__} card from a PhysicalSystem/Structure!"
        raise NotImplementedError(msg)

    def __repr__(self) -> str:
        dct = vars(self)
        align = max(map(len, dct.keys()))
        text = f"{type(self).__name__}(\n"
        for param, val in dct.items():
            text += f"    {param:{align}} = {val!r},\n"
        text += ")"
        return text


class AtomicSpecies(CardBase):
    """Atomic species card in a :class:`PwscfInput` [1]_.

    References
    ----------
    .. [1] https://www.quantum-espresso.org/Doc/INPUT_PW.html#id265
    """

    name = "atomic_species"

    atoms: list[str]
    masses: dict[str, float]
    pseudos: dict[str, str]

    def __init__(
        self,
        atoms: list[str],
        masses: dict[str, float] | None,
        pseudos: dict[str, str | Path],
    ):
        if not all(Elements.is_element(e) for e in atoms):
            msg = "Not all atoms supplied are elements!"
            raise NotAnElementError(msg)

        if masses is None:
            masses = {}
            for atom in atoms:
                _, elem = Elements.is_element(atom, return_element=True)
                masses[atom] = elem.atomic_weight

        if set(atoms) != set(masses) != set(pseudos):
            msg = "Atoms, masses, and pseudos must all have the same atoms!"
            raise ValueError(msg)

        self.atoms = atoms
        self.masses = masses
        self.pseudos = {atom: Path(pseudo).name for atom, pseudo in pseudos.items()}

    @classmethod
    def read(cls, lines: list[str]) -> Self:
        atoms = []
        masses = {}
        pseudos = {}
        for line in lines[1:]:
            atom, mass, pseudo = line.split()
            atoms.append(atom)
            masses[atom] = float(mass.lower().replace("d", "e"))
            pseudos[atom] = pseudo

        return cls(atoms, masses, pseudos)

    def write(self) -> str:
        text = f"{self.name.upper()}\n"
        for at in self.atoms:
            text += f"    {at:2} {self.masses[at]} {self.pseudos[at]}\n"

        return text

    @classmethod
    def from_phys_sys(
        cls,
        phys_sys: PhysicalSystem,
        /,
        pseudos: PseudoSet | dict[str, str] | list[str],
        **kwargs,  # noqa: ARG003
    ) -> Self:
        if isinstance(pseudos, PseudoSet | list):
            psps = PseudoSet.get_pseudos(
                pseudos = pseudos,
                system = phys_sys,
                code = "pwscf",
            )
            psps = {
                pp_elem_label(fname, guard=True)[1]: fpath for fname, fpath in psps.items()
            }
            pseudos = {}
            for ion_label in phys_sys.ion_labels:
                _, elem = Elements.is_element(ion_label, return_element=True)
                pseudos[ion_label] = psps[elem.symbol]

        return cls(
            atoms=phys_sys.ion_labels,
            masses=None,
            pseudos=pseudos,
        )


class AtomicPositions(CardBase):
    """Atomic positions card in a :class:`PwscfInput` [1]_.

    References
    ----------
    .. [1] https://www.quantum-espresso.org/Doc/INPUT_PW.html#id269
    """

    name = "atomic_positions"

    valid_specifiers = frozenset({"alat", "bohr", "angstrom", "crystal"})

    specifier: AtomicPositionsSpecifier
    atoms: list[str]
    positions: npt.NDArray[np.float64]
    force_multipliers: npt.NDArray[np.int64] | None

    def __init__(
        self,
        atoms: list[str] | Structure | PhysicalSystem,
        positions: npt.ArrayLike[np.floating] | None = None,
        force_multipliers: npt.ArrayLike[np.floating] | None = None,
        specifier: AtomicPositionsSpecifier | None = None,
    ):
        self.atoms = atoms
        if not all(Elements.is_element(e) for e in self.atoms):
            msg = "Not all atoms supplied are elements!"
            raise NotAnElementError(msg)

        if positions is None:
            msg = "pos can not be None when `atoms` is a list of elements!"
            raise ValueError(msg)
        elif len(positions) != len(self.atoms):
            msg = "The pos array must be the same length as the number of atoms!"
            raise ValueError(msg)
        self.positions = np.asarray(positions, dtype=float)

        if (s_low := specifier.lower()) in self.valid_specifiers:
            self.specifier = s_low
        else:
            msg = f"Invalid or unsupported specifier '{specifier}'!"
            raise ValueError(msg)

        if force_multipliers is not None:
            if len(force_multipliers) != len(self.atoms):
                msg = "The force_multipliers array must be the same length as the number of atoms!"
                raise ValueError(msg)
            force_multipliers = np.asarray(force_multipliers)
            if force_multipliers.dtype != int or force_multipliers.dtype != bool:
                msg = "force_multipliers must be an integer or mask array!"
            force_multipliers = force_multipliers.astype(int)

        self.force_multipliers = force_multipliers

    @classmethod
    def read(cls, lines: list[str]) -> Self:
        specifier = lines[0].split()[1].strip("{}()").lower()
        npos = len(lines[1:])
        atoms = []
        positions = np.empty((npos, 3), dtype=float)
        force_multipliers = np.zeros((npos, 3), dtype=int)
        for idx, l in enumerate(lines[1:]):
            tokens = l.split()
            atoms.append(tokens[0])
            positions[idx,:] = np.array(tokens[1:4], dtype=float)
            if len(tokens) == 7:
                force_multipliers[idx,:] = np.array(tokens[4:7], dtype=int)

        if np.sum(force_multipliers) == 0:
            force_multipliers = None

        return cls(
            atoms=atoms,
            positions=positions,
            force_multipliers=force_multipliers,
            specifier=specifier,
        )

    def write(self) -> str:
        text = f"{self.name.upper()} ({self.specifier})\n"
        for idx, (atom, pos) in enumerate(zip(self.atoms, self.positions, strict=True)):
            fmt = "16.8f"
            text += f"    {atom:2} {pos[0]:{fmt}} {pos[1]:{fmt}} {pos[2]:{fmt}}"
            if self.force_multipliers is not None:
                mult = self.force_multipliers[idx]
                text += f" {mult[0]:d} {mult[1]:d} {mult[2]:d}"
            text += "\n"

        return text

    def change_specifier(
        self,
        new_specifier: AtomicPositionsSpecifier,
        pwi: PwscfInput,
        *,
        in_place: bool = True,
    ):
        if not in_place:
            cpy = deepcopy(self)
            cpy.change_specifier(new_specifier, pwi)
            return cpy

        new_specifier = new_specifier.lower()
        if new_specifier == self.specifier:
            return
        elif new_specifier not in AtomicPositions.valid_specifiers:
            msg = (
                f"Specifier provided was '{new_specifier}' but must be one of "
                f"{sorted(AtomicPositions.valid_specifiers)}"
            )
            raise ValueError(msg)

        scale = pwi.alat
        pos = self.positions
        match self.specifier: # Convert to bohr
            case "alat":
                pos *= scale
            case "bohr":
                pass
            case "angstrom":
                pos *= convert(1., "A", "B")
            case "crystal":
                pos = np.dot(pos, pwi.axes)

        match new_specifier:
            case "alat":
                pos /= scale
            case "bohr":
                pass
            case "angstrom":
                pos /= convert(1., "A", "B")
            case "crystal":
                pos  = np.dot(pos, inv(pwi.axes))

        self.positions = pos
        self.specifier = new_specifier

    @classmethod
    def from_phys_sys(cls, phys_sys: PhysicalSystem | Structure, /, **kwargs) -> Self:  # noqa: ARG003
        if isinstance(phys_sys, PhysicalSystem):
            structure = phys_sys.structure
        else:
            structure = phys_sys

        atoms = structure.elem.tolist()
        if not all(Elements.is_element(e) for e in atoms):
            msg = "Not all atoms supplied are elements!"
            raise NotAnElementError(msg)

        pos = structure.pos
        force_multipliers = structure.frozen
        specifier = "bohr" if "b" in structure.units.lower() else "angstrom"
        return cls(
            atoms=atoms,
            pos=pos,
            force_multipliers=force_multipliers,
            specifier=specifier,
        )


class AtomicForces(CardBase):
    """Atomic forces card in a :class:`PwscfInput` [1]_.

    References
    ----------
    .. [1] https://www.quantum-espresso.org/Doc/INPUT_PW.html#id316
    """

    name = "atomic_forces"

    def __init__(self, atoms: list[str], forces: npt.ArrayLike[np.floating]) -> None:
        if not all(Elements.is_element(e) for e in atoms):
            msg = "Not all atoms supplied are elements!"
            raise NotAnElementError(msg)

        self.atoms = atoms
        self.forces = np.asarray(forces, dtype=float)
        if (shape := self.forces.shape) != (len(self.atoms), 3):
            msg = f"'forces' must have shape {(len(self.atoms), 3)} but has shape {shape}!"
            raise ValueError(msg)

    @classmethod
    def read(cls, lines: list[str]) -> Self:
        npos = len(lines)
        dim = 3
        atoms = []
        forces = np.empty((npos,dim), dtype=float)
        for idx, l in enumerate(lines):
            tokens = l.split()
            atoms.append(tokens[0])
            forces[idx,:] = np.array(tokens[1:4], dtype=float)

        return cls(atoms=atoms,forces=forces)
    #end def read


    def write(self) -> str:
        text = f"{self.name.upper()} {self.specifier}\n"
        for atom, force in zip(self.atoms, self.forces, strict=True):
            text += "    {:2} {:{f}} {:{f}} {:{f}}\n".format(
                atom, *force, f="16.8f",
            )

        return text
    #end def write


class KPoints(CardBase):
    """K-points card in a :class:`PwscfInput` [1]_.

    References
    ----------
    .. [1] https://www.quantum-espresso.org/Doc/INPUT_PW.html#id277
    """

    name = "k_points"

    valid_specifiers = frozenset(
        {
            "tpiba",
            "crystal",
            "tpiba_b",
            "crystal_b",
            "tpiba_c",
            "crystal_c",
            "automatic",
            "gamma",
        }
    )

    specifier: KPointsSpecifier
    kpoints: npt.NDArray[np.float64] | None
    weights: npt.NDArray[np.float64] | None
    grid: npt.NDArray[np.int64] | None
    shift: npt.NDArray[np.int64] | None

    @overload
    def __init__(self, specifier: Literal["gamma"]) -> None: ...
    @overload
    def __init__(
        self,
        specifier: Literal[
            "tpiba", "crystal", "tpiba_b", "crystal_b", "tpiba_c", "crystal_c",
        ],
        nkpoints: int | None,
        kpoints: npt.ArrayLike[np.floating],
        weights: npt.ArrayLike[np.floating],
    ) -> None: ...
    @overload
    def __init__(
        self,
        specifier: Literal["automatic"],
        grid: npt.ArrayLike[np.integer],
        shift: npt.ArrayLike[np.integer],
    ) -> None: ...

    def __init__(
        self,
        specifier: KPointsSpecifier,
        nkpoints: int | None = None,
        kpoints: npt.ArrayLike[np.floating] | None = None,
        weights: npt.ArrayLike[np.floating] | None = None,
        grid: npt.ArrayLike[np.integer] | None = None,
        shift: npt.ArrayLike[np.integer] | None = None,
    ) -> None:
        s_low = specifier.lower()
        if s_low not in KPoints.valid_specifiers:
            msg = (
                f"Specifier '{specifier}' is not known to Nexus!\n"
                f"Valid specifiers are {sorted(KPoints.valid_specifiers)}"
            )
            raise ValueError(msg)

        inputs = {
            "specifier": specifier,
            "nkpoints": nkpoints,
            "kpoints": kpoints,
            "weights": weights,
            "grid": grid,
            "shift": shift,
        }

        self.specifier = s_low
        self.kpoints = None
        self.weights = None
        self.grid = None
        self.shift = None
        wmsg = "Ignoring inputs {ign} with specifier '{spec}'."
        emsg = "'{var}' must have shape {shp} but has shape {bad}"
        if self.specifier == "gamma":
            ignored = [
                i for i in ("nkpoints", "kpoints", "weights", "grid", "shift")
                if inputs[i] is not None
            ]
            if len(ignored) > 0:
                LOGGER.debug(wmsg.format(ign=ignored, spec=self.specifier))
        elif self.specifier == "automatic":
            ignored = [
                i for i in ("nkpoints", "kpoints", "weights") if inputs[i] is not None
            ]
            if len(ignored) > 0:
                LOGGER.debug(wmsg.format(ign=ignored, spec=self.specifier))

            self.grid = np.asarray(grid, dtype=int)
            self.shift = np.asarray(shift, dtype=int)

            if (bad := self.grid.shape) != (3,):
                msg = emsg.format(var="grid", shp=(3,), bad=bad)
                raise ValueError(msg)

            if (bad := self.shift.shape) != (3,):
                msg = emsg.format(var="shift", shp=(3,), bad=bad)
                raise ValueError(msg)
        else:
            ignored = [i for i in ("grid", "shift") if inputs[i] is not None]
            if len(ignored) > 0:
                LOGGER.debug(wmsg.format(ign=ignored, spec=self.specifier))

            self.kpoints = np.asarray(kpoints, dtype=float)
            self.weights = np.asarray(weights, dtype=float)
            nkpoints = int(nkpoints) if nkpoints is not None else len(self.kpoints)

            if (bad := self.kpoints.shape) != (nkpoints, 3):
                msg = emsg.format(var="kpoints", shp=(nkpoints, 3), bad=bad)
                raise ValueError(msg)

            if (bad := self.weights.shape) != (nkpoints,):
                msg = emsg.format(var="weights", shp=(nkpoints,), bad=bad)
                raise ValueError(msg)

    @property
    def nkpoints(self) -> int | None:
        if self.kpoints is not None:
            return self.kpoints.shape[0]
        return None

    @classmethod
    def read(cls, lines: list[str]) -> Self:
        specifier = lines[0].split()[1].strip("{}()").lower()
        specifier = specifier if specifier != "" else "tpiba"

        if specifier not in cls.valid_specifiers:
            msg = (
                f"Specifier '{specifier}' is not known to Nexus!\n"
                f"Valid specifiers are {sorted(cls.valid_specifiers)}"
            )
            raise ValueError(msg)

        nkpoints = None
        grid     = None
        kpoints  = None
        weights  = None
        shift    = None

        if specifier == "gamma":
            pass
        elif specifier == "automatic":
            line = lines[1].split()
            grid = np.array(line[0:3], dtype=int)
            shift = np.array(line[3:6], dtype=int)
        else:
            nkpoints = int(lines[1])
            kpoints = np.empty((nkpoints, 3), dtype=float)
            weights = np.empty((nkpoints,), dtype=float)
            for idx, line in enumerate(lines[2:]):
                kpt_info = line.split()
                kpoints[idx, :] = kpt_info[0:3]
                weights[idx] = kpt_info[3]

        return cls(
            specifier=specifier,
            nkpoints=nkpoints,
            kpoints=kpoints,
            weights=weights,
            grid=grid,
            shift=shift,
        )

    def write(self) -> str:
        text = f"{self.name.upper()} ({self.specifier})\n"
        if self.specifier == "gamma":
            pass
        elif self.specifier == "automatic":
            text += "    {:{f}} {:{f}} {:{f}}  {:{f}} {:{f}} {:{f}}\n".format(
                *self.grid, *self.shift, f="2d"
            )
        else:
            text += f"  {self.nkpoints}\n"
            fmt = "<16.8f"
            for kpt, weight in zip(self.kpoints, self.weights, strict=True):
                text += "    {:{f}} {:{f}} {:{f}}".format(
                    *kpt, weight, f=fmt
                )
                if weight != 1:
                    text += f" {weight:{fmt}}"
                text += "\n"

        return text

    def change_specifier(
        self,
        new_specifier: KPointsSpecifier,
        pwi: PwscfInput,
        *,
        in_place: bool = True,
    ):
        if not in_place:
            cpy = deepcopy(self)
            cpy.change_specifier(new_specifier, pwi)
            return cpy

        new_specifier = new_specifier.lower()
        if self.specifier == new_specifier:
            return
        elif new_specifier == "automatic":
            msg = (
                "Cannot map arbitrary kpoints into an automatic Monkhorst-Pack mesh!\n"
                f"(Existing specifier: {self.specifier})"
            )
            raise ValueError(msg)
        elif new_specifier in {"tpiba_b", "crystal_b", "tpiba_c", "crystal_c"}:
            msg = f"Conversion to specifier '{new_specifier}' has not been implemented yet!"
            raise ValueError(msg)
        elif self.specifier in {"tpiba_b", "crystal_b", "tpiba_c", "crystal_c"}:
            msg = f"Conversion from specifier '{self.specifier}' has not been implemented yet!"
            raise ValueError(msg)
        elif new_specifier not in KPoints.valid_specifiers:
            msg = f"Specifier '{new_specifier}' is not known to Nexus!"
            raise ValueError(msg)

        alat, kaxes = pwi.alat, pwi.kaxes
        match self.specifier:
            case "tpiba":
                kpoints = self.kpoints * (2 * pi) / alat
            case "gamma":
                kpoints = np.zeros((1, 3), dtype=int)
            case "crystal":
                kpoints = np.dot(self.kpoints, kaxes)
            case "automatic":
                grid  = self.grid
                shift = 0.5 * self.shift
                kpoints = kmesh(kaxes, grid, shift)

        match new_specifier:
            case 'tpiba':
                kpoints = kpoints/ ((2 * pi) / alat)
            case 'gamma':
                kpoints = np.zeros((1, 3), dtype=int)
            case 'crystal':
                kpoints = np.dot(kpoints, inv(kaxes))

        self.kpoints   = kpoints
        self.specifier = new_specifier

    @classmethod
    def from_phys_sys(cls, phys_sys: PhysicalSystem | Structure, /, **kwargs) -> Self:  # noqa: ARG003
        if isinstance(phys_sys, PhysicalSystem):
            structure = phys_sys.structure
        else:
            structure = phys_sys

        kpts = structure.kpoints
        kweights = structure.kweights
        return cls(
            specifier="tpiba",
            nkpoints=len(kpts),
            kpoints=kpts,
            weights=kweights,
        )


class CellParameters(CardBase):
    """Cell parameters card in a :class:`PwscfInput` [1]_.

    References
    ----------
    .. [1] https://www.quantum-espresso.org/Doc/INPUT_PW.html#id295
    """

    name = "cell_parameters"

    valid_specifiers = frozenset({"alat", "bohr", "angstrom"})

    axes: npt.NDArray[np.float64]
    specifier: CellParametersSpecifier

    def __init__(
        self,
        axes: npt.ArrayLike[np.floating],
        specifier: CellParametersSpecifier,
    ) -> None:
        s_low = specifier.lower()
        if s_low not in CellParameters.valid_specifiers:
            msg = (
                f"Specifier {specifier} is not known to Nexus!\n"
                f"Valid specifiers are {sorted(CellParameters.valid_specifiers)}"
            )
            raise ValueError(msg)

        self.axes = np.asarray(axes, dtype=float)
        if (shape := axes.shape) != (3, 3):
            msg = f"'axes' must have shape (3, 3) but has shape {shape}!"
            raise ValueError(msg)


    @classmethod
    def read(cls, lines: list[str]) -> Self:
        specifier = lines[0].split()[1].strip("{}()").lower()

        if specifier not in cls.valid_specifiers:
            msg = (
                f"Specifier '{specifier}' is not known to Nexus!\n"
                f"Valid specifiers are {sorted(cls.valid_specifiers)}"
            )
            raise ValueError(msg)

        axes = np.empty((3, 3), dtype=float)
        for idx, line in enumerate(lines[1:]):
            axes[idx, :] = line.split()

        return cls(axes=axes, specifier=specifier)
    #end def read

    def write(self) -> str:
        text = f"{self.name.upper()} {self.specifier}\n"
        for vec in self.axes:
            text += "    {:{f}} {:{f}} {:{f}}\n".format(*vec, f="16.8f")

        return text
    #end def write

    def change_specifier(
        self,
        new_specifier: CellParametersSpecifier,
        pwi: PwscfInput,
        *,
        in_place: bool = True,
    ) -> None:
        if not in_place:
            cpy = deepcopy(self)
            cpy.change_specifier(new_specifier, pwi)
            return cpy

        s_low = new_specifier.lower()
        if self.specifier == s_low:
            return
        elif s_low not in CellParameters.valid_specifiers:
            msg = (
                f"Specifier {new_specifier} is not known to Nexus!\n"
                f"Valid specifiers are {sorted(CellParameters.valid_specifiers)}"
            )
            raise ValueError(msg)

        alat = pwi.alat
        vec = self.axes
        match self.specifier:
            case "alat":
                vec *= alat
            case "bohr":
                pass
            case "angstrom":
                vec *= convert(1.0, "A", "B")

        match s_low:
            case "alat":
                vec /= alat
            case "bohr":
                pass
            case "angstrom":
                vec /= convert(1.0, "A", "B")

        self.vectors   = vec
        self.specifier = s_low
    #end def change_specifier

    @classmethod
    def from_phys_sys(cls, phys_sys: PhysicalSystem | Structure, /, **kwargs) -> Self:  # noqa: ARG003
        if isinstance(phys_sys, PhysicalSystem):
            structure = phys_sys.structure
        else:
            structure = phys_sys

        if not structure.has_axes():
            msg = "Structure does not have axes!"
            raise ValueError(msg)
        specifier = "bohr" if "b" in structure.units.lower() else "angstrom"
        axes = structure.axes
        return cls(axes=axes, specifier=specifier)


class Constraint(ABC):

    constraint_names: ClassVar = frozenset(
        {
            "type_coord",
            "atom_coord",
            "distance",
            "planar_angle",
            "torsional_angle",
            "bennett_proj",
            "potential_wall",
        }
    )

    name: ClassVar[ConstraintName]
    constr_target: float | None

    @classmethod
    def _verify_types(cls, **kwargs: int | float) -> str | Literal[""]:
        types = inspect.get_annotations(cls.__init__)
        types.pop("self", None)
        types.pop("return", None)

        msg = ""
        for key, value in kwargs.items():
            dtype = types[key]
            if dtype is int and value.is_integer():
                # Convert integer-value floats to ints
                value = int(value)

            dtype = int | np.integer if dtype is int else float | np.floating
            if not isinstance(value, dtype):
                msg += (
                    f"Input '{key}' should have type {dtype.__name__} "
                    f"but has type {type(value).__name__}!"
                )

        return msg

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        ...

    @property
    @abstractmethod
    def values(self) -> list[int | float]:
        ...


class TypeCoordConstraint(Constraint):
    name = "type_coord"

    atom_idx_1: int
    atom_idx_2: int
    cutoff_radius: float
    smoothing: float

    def __init__(
        self,
        atom_idx_1: int,
        atom_idx_2: int,
        cutoff_radius: float,
        smoothing: float,
        constr_target: float | None = None,
        *,
        check: bool = True,
    ) -> None:
        if check:
            msg = self.__class__._verify_types(
                atom_idx_1=atom_idx_1,
                atom_idx_2=atom_idx_2,
                cutoff_radius=cutoff_radius,
                smoothing=smoothing,
                constr_target=constr_target,
            )
            if len(msg) > 0:
                raise TypeError(msg)

        self.atom_idx_1 = atom_idx_1
        self.atom_idx_2 = atom_idx_2
        self.cutoff_radius = cutoff_radius
        self.smoothing = smoothing
        self.constr_target = constr_target

    @property
    def values(self) -> list[int, int, float, float]:
        vals = [
            self.atom_idx_1,
            self.atom_idx_2,
            self.cutoff_radius,
            self.smoothing,
        ]
        if self.constr_target is not None:
            vals.append(self.constr_target)

        return vals


class AtomCoordConstraint(TypeCoordConstraint):
    name = "atom_coord"


class DistanceConstraint(Constraint):
    name = "distance"

    atom_idx_1: int
    atom_idx_2: int

    def __init__(
        self,
        atom_idx_1: int,
        atom_idx_2: int,
        constr_target: float | None = None,
        *,
        check: bool = True,
    ) -> None:
        if check:
            msg = self.__class__._verify_types(
                atom_idx_1=atom_idx_1,
                atom_idx_2=atom_idx_2,
                constr_target=constr_target,
            )
            if len(msg) > 0:
                raise TypeError(msg)

        self.atom_idx_1 = atom_idx_1
        self.atom_idx_2 = atom_idx_2
        self.constr_target = constr_target

    @property
    def values(self) -> list[int, int]:
        vals = [
            self.atom_idx_1,
            self.atom_idx_2,
        ]
        if self.constr_target is not None:
            vals.append(self.constr_target)

        return vals


class PlanarAngleConstraint(Constraint):
    name = "planar_angle"

    atom_idx_1: int
    atom_idx_2: int
    atom_idx_3: int

    def __init__(
        self,
        atom_idx_1: int,
        atom_idx_2: int,
        atom_idx_3: int,
        constr_target: float | None = None,
        *,
        check: bool = True,
    ) -> None:
        if check:
            msg = self.__class__._verify_types(
                atom_idx_1=atom_idx_1,
                atom_idx_2=atom_idx_2,
                atom_idx_3=atom_idx_3,
                constr_target=constr_target,
            )
            if len(msg) > 0:
                raise TypeError(msg)

        self.atom_idx_1 = atom_idx_1
        self.atom_idx_2 = atom_idx_2
        self.atom_idx_3 = atom_idx_3
        self.constr_target = constr_target

    @property
    def values(self) -> list[int, int, int]:
        vals = [
            self.atom_idx_1,
            self.atom_idx_2,
            self.atom_idx_3,
        ]
        if self.constr_target is not None:
            vals.append(self.constr_target)

        return vals


class TorsionalAngleConstraint(Constraint):
    name = "torsional_angle"

    atom_idx_1: int
    atom_idx_2: int
    atom_idx_3: int
    atom_idx_4: int

    def __init__(
        self,
        atom_idx_1: int,
        atom_idx_2: int,
        atom_idx_3: int,
        atom_idx_4: int,
        constr_target: float | None = None,
        *,
        check: bool = True,
    ) -> None:
        if check:
            msg = self.__class__._verify_types(
                atom_idx_1=atom_idx_1,
                atom_idx_2=atom_idx_2,
                atom_idx_3=atom_idx_3,
                atom_idx_4=atom_idx_4,
                constr_target=constr_target,
            )
            if len(msg) > 0:
                raise TypeError(msg)

        self.atom_idx_1 = atom_idx_1
        self.atom_idx_2 = atom_idx_2
        self.atom_idx_3 = atom_idx_3
        self.atom_idx_4 = atom_idx_4
        self.constr_target = constr_target

    @property
    def values(self) -> list[int, int, int, int]:
        vals = [
            self.atom_idx_1,
            self.atom_idx_2,
            self.atom_idx_3,
            self.atom_idx_4,
        ]
        if self.constr_target is not None:
            vals.append(self.constr_target)

        return vals


class BennettProjConstraint(Constraint):
    name = "bennett_proj"

    atom_idx: int
    vector: npt.NDArray[np.float64]

    def __init__(
        self,
        atom_idx: int,
        vector: npt.ArrayLike[np.floating],
        constr_target: float | None = None,
        *,
        check: bool = True,
    ) -> None:
        if check:
            msg = self.__class__._verify_types(
                atom_idx=atom_idx,
                constr_target=constr_target,
            )
            if len(msg) > 0:
                raise TypeError(msg)

        vector = np.asarray(vector, dtype=float)
        if (shape := vector.shape) != (3,):
            msg = f"'vector' must have shape (3,) but has shape {shape}!"
            raise ValueError(msg)

        self.atom_idx = atom_idx
        self.vector = vector
        self.constr_target = constr_target

    @property
    def values(self) -> list[int, float, float, float]:
        vals = [
            self.atom_idx,
            *self.vector,
        ]
        if self.constr_target is not None:
            vals.append(self.constr_target)

        return vals


class PotentialWallConstraint(Constraint):
    name = "potential_wall"

    prefactor: float
    exponent: float
    cutoff: float

    def __init__(
        self,
        prefactor: float,
        exponent: float,
        cutoff: float,
        constr_target: float | None = None,
        *,
        check: bool = True,
    ) -> None:
        if check:
            msg = self.__class__._verify_types(
                prefactor=prefactor,
                exponent=exponent,
                cutoff=cutoff,
                constr_target=constr_target,
            )
            if len(msg) > 0:
                raise TypeError(msg)

        self.prefactor = prefactor
        self.exponent = exponent
        self.cutoff = cutoff
        self.constr_target = constr_target

    @property
    def values(self) -> list[float, float, float]:
        vals = [
            self.prefactor,
            self.exponent,
            self.cutoff,
        ]
        if self.constr_target is not None:
            vals.append(self.constr_target)

        return vals


class Constraints(CardBase):
    """Constraints card in a :class:`PwscfInput` [1]_.

    References
    ----------
    .. [1] https://www.quantum-espresso.org/Doc/INPUT_PW.html#id299
    """

    name = "constraints"

    constraint_map = MappingProxyType(
        {
            "type_coord":      TypeCoordConstraint,
            "atom_coord":      AtomCoordConstraint,
            "distance":        DistanceConstraint,
            "planar_angle":    PlanarAngleConstraint,
            "torsional_angle": TorsionalAngleConstraint,
            "bennett_proj":    BennettProjConstraint,
            "potential_wall":  PotentialWallConstraint,
        }
    )

    constraints: list[Constraint]
    tolerance: float | None

    def __init__(
        self,
        constraints: list[Constraint],
        tolerance: float | None = None,
    ):
        ...

    @classmethod
    def read(cls, lines: list[str]) -> Self:
        # First line has no useful information
        constr_info = lines[1].split()
        nconstraints = int(constr_info[0])
        if len(constr_info) > 1:
            tolerance = float(constr_info[1])

        constraints = []
        for line in lines[2:]:
            tokens = line.split()
            cons_name = tokens[0]
            if cons_name not in Constraints.constraint_map:
                msg = (
                    f"Constraint {cons_name} is not known to Nexus!\n"
                    f"Known constraints are {[*Constraints.constraint_map]}"
                )
                raise ValueError(msg)
            cons_type = Constraints.constraint_map[cons_name]
            cons_params = np.array(tokens[1:], dtype=np.float64)
            constraints.append(
                cons_type(*cons_params)
            )

        if (n_found := len(constraints)) != nconstraints:
            msg = f"Expected {nconstraints} constraints but only found {n_found}!"
            raise ValueError(msg)

        return cls(constraints=constraints, tolerance=tolerance)
    #end def read

    @property
    def nconstraints(self) -> int:
        return len(self.constraints)

    def write(self) -> str:
        text = f"{self.name.upper()}\n"
        text += f"  {self.nconstraints}"
        if self.tolerance is not None:
            text += f"  {self.tolerance:.8f}"

        for cons in self.constraints:
            text += f"    {cons.name}"
            for val in cons.values:
                fmt = "4d" if isinstance(val, int | np.integer) else "12.8f"
                text += f" {val:{fmt}}"
            text += "\n"
        return text
    #end def write


class Occupations(CardBase):
    """Occupations card in a :class:`PwscfInput` [1]_.

    References
    ----------
    .. [1] https://www.quantum-espresso.org/Doc/INPUT_PW.html#id308
    """

    name = "occupations"

    occupations = npt.NDArray[np.floating]
    minority_spin_occs = npt.NDArray[np.floating] | None

    def __init__(
        self,
        occupations: npt.ArrayLike[np.floating],
        minority_spin_occs: npt.ArrayLike[np.floating] | None = None,
    ) -> None:
        self.occupations = np.asarray(occupations, dtype=float)
        if len(shape := self.occupations.shape) != 1:
            msg = f"Spin occupations must be a 1D array, but has shape {shape}!"
            raise ValueError(msg)

        if minority_spin_occs is not None:
            if len(minority_spin_occs) != len(self.occupations):
                msg = (
                    "Majority and minority spin occupations must have the same length!\n"
                    f"Majority occupations length: {len(occupations)}\n"
                    f"Minority occupations length: {len(minority_spin_occs)}"
                )
                raise ValueError(msg)

            self.minority_spin_occs = np.asarray(minority_spin_occs, dtype=float)
        else:
            self.minority_spin_occs = minority_spin_occs

    @classmethod
    def read(cls, lines: list[str]) -> Self:
        spin_occs = []
        for line in lines[1:]: # First line is just card name
            spin_occs.extend(map(float, line.split()))

        return cls(spin_occs)
    #end def read

    def post_process_read(self, pwscf_input: PwscfInput) -> None:
        occupations = pwscf_input.system.params.get("occupations")
        if occupations != "from_input":
            msg = "Explicit occupations only available with 'occupations=from_input'!"
            raise ValueError(msg)

        nspin = pwscf_input.system.params.get("nspin")
        if nspin is None or nspin == 1:
            return

        nbnd = pwscf_input.system.params.get("nbnd")
        if nbnd is None:
            msg = "Can not set Occupations when `nbnd` is not specified!"
            raise ValueError(msg)
        elif (n_occs := len(self.occupations)) / 2 != nbnd:
            msg = (
                "The number of occupations must be '2*nbnd' for a spin-polarized calculation!\n"
                f"Number of Bands:       {nbnd}\n"
                f"Number of Occupations: {n_occs}\n"
            )
            raise ValueError(msg)

        occupations = self.occupations
        self.occupations = occupations[:nbnd]
        self.minority_spin_occs = occupations[nbnd:]
    #end def post_process_read

    def write(self) -> str:
        text = f"{self.name.upper()}\n"
        if self.minority_spin_occs is not None:
            occs = chain(self.occupations, self.minority_spin_occs)
        else:
            occs = self.occupations

        for occ_line in batched(occs, n=10):
            text += "   "
            for occ in occ_line:
                text += f" {occ:>.8f}"
            text += "\n"

        return text
    #end def write


class HubbardParam(ABC):
    """Base class for representations of a Hubbard parameter in the :class:`Hubbard` card."""

    sorted_param_names: ClassVar = ("U", "J0", "J", "V", "ALPHA")
    valid_param_names: ClassVar = frozenset(sorted_param_names)
    sorted_angular_terms: ClassVar = ("s", "p", "d", "f")
    valid_angular_terms: ClassVar = frozenset(sorted_angular_terms)

    hubbard_type: ClassVar[Literal["U", "J0", "J", "V", "ALPHA"]]

    @abstractmethod
    def __init__(self, label: str, manifold: str, **kwargs) -> None:
        ...

    @staticmethod
    def _check_label(label: str) -> str:
        """Make sure the label is a valid element."""
        if not Elements.is_element(label):
            msg = f"Can not determine element for label '{label}'"
            raise NotAnElementError(msg)

        return label

    @staticmethod
    def _check_manifold[T: str | None](manifold: T) -> T:
        """Check principle and angular terms, return lowercased manifold.

        If ``manifold`` is ``None``, return ``None``.
        """
        if manifold is None:
            return manifold

        manifold_l = manifold[1].lower()
        if manifold_l not in HubbardParam.valid_angular_terms:
            msg = (
                f"Invalid manifold '{manifold}', "
                f"angular term must be one of {HubbardParam.sorted_angular_terms}"
            )
            raise ValueError(msg)

        manifold_n = int(manifold[0])
        if (
            (manifold_l == "s" and manifold_n < (req_n := 1))
            or (manifold_l == "p" and manifold_n < (req_n := 2))
            or (manifold_l == "d" and manifold_n < (req_n := 3))
            or (manifold_l == "f" and manifold_n < (req_n := 4))
        ):
            msg = (
                f"Invalid manifold '{manifold}'!\n"
                f"{manifold_l} orbital must have n >= {req_n}"
            )
            raise ValueError(msg)

        return manifold.lower()

    @classmethod
    def _check_hub_type(cls, htype: str, expected: set[str] | None = None) -> str:
        """Check if the Hubbard type (U, J0, J, etc.) matches the expected.

        If the Hubbard type provided does not match the expected, this function
        will suggest another Hubbard class if the value of ``htype`` is known.
        """
        if expected is None:
            expected = {cls.hubbard_type}

        expected = {i.upper() for i in expected}
        msg = ""
        if (hub_type := htype.upper()) not in expected:
            msg += f"Line begins with '{hub_type}', not {expected}!\n"
            if (hub_class := Hubbard.hub_type_map.get(hub_type)) is not None:
                msg += f"Perhaps you should use `{hub_class.__name__}.from_str()` instead?"
                raise ValueError(msg)
            else:
                msg += (
                    "This value is not known by Nexus, please check your input!\n"
                )

        return msg

    @staticmethod
    def _join_lbl_mnfld(label: str, manifold: str, manifold_2: str | None = None) -> str:
        text = f"{label.title()}-{manifold.lower()}"
        if manifold_2 is not None:
            text += f"-{manifold_2.lower()}"
        return text

    @staticmethod
    def _split_lbl_mnfld(lbl_mnfld: str) -> tuple[str, str, str | None]:
        """Split a string of the form ``lbl-mnfld[-mnfld2]`` into a tuple of the parts.

        For consistency, this will always return three items, even if there is
        only one manifold in the string.

        For example, ``Mn-3d`` returns ``('Mn', '3d', None)``, and
        ``Mn-3p-3s`` into ``('Mn', '3p', '3s')``
        """
        parts = lbl_mnfld.split("-")
        if len(parts) == 2:
            return (parts[0], parts[1], None)
        elif len(parts) == 3:
            return tuple(parts)
        else:
            msg = f"Invalid label-manifold: '{lbl_mnfld}'!"
            raise ValueError(msg)

    @abstractmethod
    def write(self) -> str:
        ...

    @classmethod
    @abstractmethod
    def from_str(cls, text: str | list[str]) -> Self:
        ...

    @classmethod
    @abstractmethod
    def from_mapping(cls, mapping: Mapping, **kwargs: Any) -> list[Self]:
        ...

    def __repr__(self) -> str:
        dct = self.__dict__
        sig = inspect.signature(self.__init__).parameters
        align = max(map(len, sig.keys()))
        text = f"{type(self).__name__}(\n"
        for param in sig:
            text += f"    {param:{align}} = {dct.get(param)!r},\n"
        text += ")"
        return text


class HubbardU(HubbardParam):
    hubbard_type = "U"

    label: str
    manifold: str
    manifold_2: str | None
    u_val: float
    eigenstates: list[int]

    def __init__(
        self,
        label: str,
        manifold: str,
        u_val: float,
        *,
        manifold_2: str | None = None,
        eigenstates: Iterable[int] | None = None,
    ) -> None:
        self.label = super()._check_label(label)
        self.manifold = super()._check_manifold(manifold)
        self.manifold_2 = super()._check_manifold(manifold_2)
        self.u_val = float(u_val)
        self.eigenstates = eigenstates
        if self.eigenstates is not None:
            self.eigenstates = list(map(int, self.eigenstates))

    def write(self) -> str:
        text = "U"
        lbl_mnfld = super()._join_lbl_mnfld(self.label, self.manifold, self.manifold_2)
        text += f"  {lbl_mnfld:>5}  {self.u_val:<6}"
        if self.eigenstates is not None:
            text += " ".join(map(str, self.eigenstates))
        text += "\n"
        return text

    @classmethod
    def from_str(cls, text: str) -> Self:
        txt = text.strip()
        if len(txt.splitlines()) > 1:
            msg = "Got more than one line of text for a HubbardU value!"
            raise ValueError(msg)

        tokens = txt.split()
        if msg := super()._check_hub_type(tokens[0]):
            raise ValueError(msg)

        label, manifold, manifold_2 = cls._split_lbl_mnfld(tokens[1])
        u_val = float(tokens[2])
        eigenstates = [int(i) for i in tokens[3:]] if len(tokens) > 3 else None

        return cls(
            label=label,
            manifold=manifold,
            u_val=u_val,
            manifold_2=manifold_2,
            eigenstates=eigenstates,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float | Mapping], **_) -> list[Self]:
        items = []
        for lbl_mnfld, info in mapping.items():
            lbl, mnfld, mnfld_2 = cls._split_lbl_mnfld(lbl_mnfld)
            eigenstates = None
            if isinstance(info, Mapping):
                u_val = info["value"]
                eigenstates = info.get("eigenstates")
            else:
                u_val = info

            items.append(
                cls(
                    lbl,
                    mnfld,
                    u_val,
                    manifold_2=mnfld_2,
                    eigenstates=eigenstates,
                )
            )
        return items


class HubbardJ0(HubbardParam):
    hubbard_type = "J0"

    label: str
    manifold: str
    manifold_2: str | None
    j0_val: float

    def __init__(
        self,
        label: str,
        manifold: str,
        j0_val: float,
        *,
        manifold_2: str | None = None
    ) -> None:
        self.label = super()._check_label(label)
        self.manifold = super()._check_manifold(manifold)
        self.manifold_2 = super()._check_manifold(manifold_2)
        self.j0_val = float(j0_val)

    def write(self) -> str:
        text = "J0"
        lbl_mnfld = super()._join_lbl_mnfld(self.label, self.manifold, self.manifold_2)
        text += f" {lbl_mnfld}  {self.j0_val:<6}\n"
        return text

    @classmethod
    def from_str(cls, text: str) -> Self:
        txt = text.strip()
        if len(txt.splitlines()) > 1:
            msg = "Got more than one line of text for a HubbardJ0 value!"
            raise ValueError(msg)

        tokens = txt.split()
        if msg := super()._check_hub_type(tokens[0]):
            raise ValueError(msg)

        label, manifold, manifold_2 = cls._split_lbl_mnfld(tokens[1])
        j0_val = float(tokens[2])
        return cls(
            label=label,
            manifold=manifold,
            j0_val=j0_val,
            manifold_2=manifold_2,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float | Mapping], **_) -> list[Self]:
        items = []
        for lbl_mnfld, info in mapping.items():
            lbl, mnfld, mnfld_2 = HubbardParam._split_lbl_mnfld(lbl_mnfld)
            j0_val = info["value"] if isinstance(info, Mapping) else info
            items.append(
                HubbardJ0(
                    lbl,
                    mnfld,
                    j0_val,
                    manifold_2=mnfld_2,
                )
            )
        return items


class HubbardAlpha(HubbardParam):
    hubbard_type = "ALPHA"

    label: str
    manifold: str
    alpha_val: float
    eigenstates: list[int]

    def __init__(
        self,
        label: str,
        manifold: str,
        alpha_val: float,
        *,
        eigenstates: Iterable[int] | None = None,
    ) -> None:
        self.label = super()._check_label(label)
        self.manifold = super()._check_manifold(manifold)
        self.alpha_val = float(alpha_val)
        self.eigenstates = list(map(int, eigenstates))

    def write(self) -> str:
        text = "ALPHA"
        lbl_mnfld = super()._join_lbl_mnfld(self.label, self.manifold)
        text += f" {lbl_mnfld}  {self.alpha_val:<6}"
        if self.eigenstates is not None:
            text += " ".join(map(str, self.eigenstates))
        text += "\n"
        return text

    @classmethod
    def from_str(cls, text: str) -> Self:
        txt = text.strip()
        if len(txt.splitlines()) > 1:
            msg = "Got more than one line of text for a HubbardAlpha value!"
            raise ValueError(msg)

        tokens = txt.split()
        if msg := super()._check_hub_type(tokens[0]):
            raise ValueError(msg)

        label, manifold, _ = cls._split_lbl_mnfld(tokens[1])
        alpha_val = float(tokens[2])
        eigenstates = [int(i) for i in tokens[3:]] if len(tokens) > 3 else None

        return cls(
            label=label,
            manifold=manifold,
            alpha_val=alpha_val,
            eigenstates=eigenstates,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float | Mapping], **_) -> list[Self]:
        items = []
        for lbl_mnfld, info in mapping.items():
            lbl, mnfld, _ = HubbardParam._split_lbl_mnfld(lbl_mnfld)
            eigenstates = None
            if isinstance(info, Mapping):
                alpha_val = info["value"]
                eigenstates = info.get("eigenstates")
            else:
                alpha_val = info
            items.append(
                HubbardAlpha(
                    lbl,
                    mnfld,
                    alpha_val,
                    eigenstates=eigenstates,
                )
            )
        return items


class HubbardJ(HubbardParam):
    hubbard_type = "J"

    label: str
    manifold: str
    j_val: float | None = None
    b_val: float | None = None
    e2_val: float | None = None
    e3_val: float | None = None

    def __init__(
        self,
        label: str,
        manifold: str,
        j_val: float | None = None,
        b_val: float | None = None,
        e2_val: float | None = None,
        e3_val: float | None = None,
    ) -> None:
        self.label = super()._check_label(label)
        self.manifold = super()._check_manifold(manifold)

        if all(i is None for i in {j_val, b_val, e2_val, e3_val}):
            msg = "Must provide at least one non-None value for Hubbard J!"
            raise ValueError(msg)

        ang_momentum = self.manifold[1]
        if ang_momentum == "d" and not (e2_val is None and e3_val is None):
            msg = "Can not specify E2 and/or E3 value for d-orbital manifold!"
            raise ValueError(msg)
        elif ang_momentum == "f" and b_val is not None:
            msg = "Can not specify B value for f-orbital manifold!"
            raise ValueError(msg)

        self.j_val = float(j_val) if j_val is not None else j_val
        self.b_val = float(b_val) if b_val is not None else b_val
        self.e2_val = float(e2_val) if e2_val is not None else e2_val
        self.e3_val = float(e3_val) if e3_val is not None else e3_val

    def write(self) -> str:
        text = ""
        lbl_mnfld = super()._join_lbl_mnfld(self.label, self.manifold)
        if (j_val := self.j_val) is not None:
            text += f"J  {lbl_mnfld:>5}  {j_val:<6}\n"

        if (b_val := self.b_val) is not None:
            text += f"B  {lbl_mnfld:>5}  {b_val:<6}\n"
            return text # Must be for d-orbital, can't have others

        if (e2_val := self.e2_val) is not None:
            text += f"E2 {lbl_mnfld:>5}  {e2_val:<6}\n"

        if (e3_val := self.e3_val) is not None:
            text += f"E3 {lbl_mnfld:>5}  {e3_val:<6}\n"

        return text

    @classmethod
    def from_str(cls, text: str | list[str]) -> Self:
        if isinstance(text, str):
            text = text.strip().splitlines()

        if (nlines := len(text)) > 3:
            msg = f"Hubbard J values can have at most 3 lines, given {nlines}"
            raise ValueError(msg)

        values = {}
        for idx, line in enumerate(text):
            tokens = line.strip().split()
            hubj_part = tokens[0].upper()
            if msg := super()._check_hub_type(hubj_part, {"J", "B", "E2", "E3"}):
                raise ValueError(msg)

            if idx == 0:
                label, manifold, _ = cls._split_lbl_mnfld(tokens[1])
            else:
                tmp_lbl, tmp_mnfld, _ = cls._split_lbl_mnfld(tokens[1])
                if not (
                    tmp_lbl.lower() == label.lower()
                    and tmp_mnfld.lower() == manifold.lower()
                ):
                    msg = (
                        "Multi-line input has inconsistent labels/manifolds!\n"
                        f"Label/manifold #1: {label}-{manifold}\n"
                        f"Label/manifold #{idx}: {tmp_lbl}-{tmp_mnfld}"
                    )
                    raise ValueError(msg)

            values[hubj_part] = float(tokens[2])

        return cls(
            label=label,
            manifold=manifold,
            j_val=values.get("J"),
            b_val=values.get("B"),
            e2_val=values.get("E2"),
            e3_val=values.get("E3"),
        )

    @classmethod
    def join(cls, *hub_js: HubbardJ) -> list[Self]:
        """Join multiple :class:`HubbardJ` instances together."""
        if len(hub_js) == 0:
            return []

        if len(invalid := {i.__name__ for i in hub_js if not isinstance(i, cls)}) > 0:
            msg = f"Received non-{cls.__name__} instance(s) of type(s) {invalid}"
            raise TypeError(msg)

        lbl_mnflds: dict[tuple[str, str], list[HubbardJ]] = {}
        for hj in hub_js:
            lbl_mnfld = (hj.label, hj.manifold)
            if lbl_mnfld not in lbl_mnflds:
                lbl_mnflds[lbl_mnfld] = []
            lbl_mnflds[lbl_mnfld].append(hj)

        joined = []
        for (label, manifold), hjs in lbl_mnflds.items():
            if manifold[1] == "d" and (nhjs := len(hjs)) > 2:
                msg = (
                    f"Found {nhjs} HubbardJ objects for {label}-{manifold}, "
                    "but there can only be at most 2!"
                )
                raise ValueError(msg)
            elif manifold[1] == "f" and (nhjs := len(hjs)) > 3:
                msg = (
                    f"Found {nhjs} HubbardJ objects for {label}-{manifold}, "
                    "but there can only be at most 3!"
                )
                raise ValueError(msg)

            values = {}
            for hj in hjs:
                current = {}
                for k, v in (
                    ("j_val", hj.j_val),
                    ("b_val", hj.b_val),
                    ("e2_val", hj.e2_val),
                    ("e3_val", hj.e3_val),
                ):
                    if v is not None:
                        current[k] = v

                overlap = {
                    k: old_v for k in current if (old_v := values.get(k)) is not None
                }
                if len(overlap) > 0:
                    msg = "Found duplicate definitions of values for HubbardJ parameters!\n"
                    for k, old_v in overlap.items():
                        msg += f"Existing {k}: {old_v} - Duplicate {k}: {current[k]}\n"
                    raise ValueError(msg)

                values.update(current)

            joined.append(
                cls(
                    label=label,
                    manifold=manifold,
                    **values
                )
            )

        return joined

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping,
        itype: Literal["J", "B", "E2", "E3"],
        **_,
    ) -> list[Self]:
        items = []
        for lbl_mnfld, info in mapping.items():
            lbl, mnfld, _ = HubbardParam._split_lbl_mnfld(lbl_mnfld)
            value = {f"{itype.lower()}_val": info}
            items.append(
                HubbardJ(
                    lbl,
                    mnfld,
                    **value,
                )
            )
        return items


class HubbardV(HubbardParam):
    hubbard_type = "V"

    label_1: str
    label_1_manifold: str
    label_1_idx: int
    label_1_manifold_2: str | None
    label_2: str
    label_2_manifold: str
    label_2_idx: int
    label_2_manifold_2: str | None
    v_val: float

    def __init__(
        self,
        label_1: str,
        label_1_manifold: str,
        label_1_idx: int,
        label_2: str,
        label_2_manifold: str,
        label_2_idx: int,
        v_val: float,
        *,
        label_1_manifold_2: str | None = None,
        label_2_manifold_2: str | None = None,
    ) -> None:
        self.label_1 = super()._check_label(label_1)
        self.label_2 = super()._check_label(label_2)
        self.label_1_manifold = super()._check_manifold(label_1_manifold)
        self.label_2_manifold = super()._check_manifold(label_2_manifold)
        self.label_1_manifold_2 = super()._check_manifold(label_1_manifold_2)
        self.label_2_manifold_2 = super()._check_manifold(label_2_manifold_2)

        self.label_1_idx = int(label_1_idx)
        self.label_2_idx = int(label_2_idx)
        self.v_val = float(v_val)

    def write(self) -> str:
        text = "V   "
        lbl_1_mnfld = super()._join_lbl_mnfld(
            self.label_1, self.label_1_manifold, self.label_1_manifold_2,
        )

        lbl_2_mnfld = super()._join_lbl_mnfld(
            self.label_2, self.label_2_manifold, self.label_2_manifold_2,
        )

        text += (
            f"{lbl_1_mnfld:<8} "
            f"{lbl_2_mnfld:<8} "
            f"{self.label_1_idx:>2} {self.label_2_idx:>2}    "
            f"{self.v_val:<6}\n"
        )
        return text

    @classmethod
    def from_str(cls, text: str) -> Self:
        txt = text.strip()
        if len(txt.splitlines()) > 1:
            msg = "Got more than one line of text for a HubbardV value!"
            raise ValueError(msg)

        tokens = txt.split()
        if msg := super()._check_hub_type(tokens[0]):
            raise ValueError(msg)

        label_1, label_1_manifold, label_1_manifold_2 = cls._split_lbl_mnfld(tokens[1])
        label_1_idx = int(tokens[3])

        label_2, label_2_manifold, label_2_manifold_2 = cls._split_lbl_mnfld(tokens[2])
        label_2_idx = int(tokens[4])

        v_val = float(tokens[5])

        return cls(
            label_1=label_1,
            label_1_manifold=label_1_manifold,
            label_1_idx=label_1_idx,
            label_2=label_2,
            label_2_manifold=label_2_manifold,
            label_2_idx=label_2_idx,
            v_val=v_val,
            label_1_manifold_2=label_1_manifold_2,
            label_2_manifold_2=label_2_manifold_2,
        )

    def is_onsite(self) -> bool:
        return self.label_1_idx == self.label_2_idx

    def is_intersite(self) -> bool:
        return self.label_1_idx != self.label_2_idx

    @overload
    @staticmethod
    def sort_by_interaction_type(
        v_params: list[HubbardU | HubbardJ0 | HubbardV],
        standard: str | None = None,
        background: str | None = None,
    ) -> list[HubbardU | HubbardJ0 | HubbardV]:
        ...
    @overload
    @staticmethod
    def sort_by_interaction_type(
        v_params: list[HubbardV],
        standard: str,
        background: str,
    ) -> list[HubbardV]:
        ...

    @staticmethod
    def sort_by_interaction_type(
        v_params: list[HubbardU | HubbardJ0 | HubbardV],
        standard: str | None = None,
        background: str | None = None,
    ) -> list[HubbardU | HubbardJ0 | HubbardV]:
        """Sort a list of Hubbard V parameters by their interaction type.

        The QE docs state that the 4 types of interaction must be in the
        following order:

        1. Standard-Standard
        2. Standard-Background
        3. Background-Background
        4. Background-Standard

        Parameters
        ----------
        v_params : list of HubbardV or HubbardU
            List of parameters to sort. Can contain **AT MOST** one
            :class:`HubbardU` parameter and **AT MOST** one :class:`HubbardJ0`
            parameter. If there is one :class:`HubbardU` or :class:`HubbardJ0`
            instance, then *both* ``standard`` and ``background`` are optional.
        standard : str, optional
            The full name of the standard label+manifold, e.g. ``Fe-3d``. If
            this is not provided, you must provide either ``background`` or a
            :class:`HubbardU`/:class:`HubbardJ0` instance in ``v_params``.
        background : str, optional
            The full name of the background label+manifold, e.g. ``Fe-3d-4s``.
            If this is not provided, you must provide either ``standard`` or a
            :class:`HubbardU`/:class:`HubbardJ0` instance in ``v_params``.
        """
        if len([i for i in v_params if isinstance(i, HubbardU)]) > 1:
            msg = "Can only pass at most 1 Hubbard U parameter!"
            raise ValueError(msg)
        elif len([i for i in v_params if isinstance(i, HubbardJ0)]) > 1:
            msg = "Can only pass at most 1 Hubbard J0 parameter!"
            raise ValueError(msg)
        elif (
            (standard is None and background is None)
            and not any(isinstance(p, HubbardU | HubbardJ0) for p in v_params)
        ):
            msg = "Can only infer standard AND background when a HubbardU/J0 value is present!"
            raise ValueError(msg)

        if standard is None:
            hub_uj0 = next(i for i in v_params if isinstance(i, HubbardU | HubbardJ0))
            if hub_uj0.manifold_2 is not None:
                msg = "Hubbard U/J0 cannot have more than 1 manifold for a DFT+U+V calculation!"
                raise ValueError(msg)

            standard = HubbardParam._join_lbl_mnfld(hub_uj0.label, hub_uj0.manifold)

        if background is None:
            hvs = iter(i for i in v_params if isinstance(i, HubbardV) and i.is_onsite())
            while background is None:
                hv = next(hvs, None)
                if hv is None:
                    break
                tmp = HubbardParam._join_lbl_mnfld(
                    hv.label_1, hv.label_1_manifold, hv.label_1_manifold_2,
                )
                tmp2 = HubbardParam._join_lbl_mnfld(
                    hv.label_2, hv.label_2_manifold, hv.label_2_manifold_2,
                )
                if tmp != standard:
                    background = tmp
                elif tmp2 != standard:
                    background = tmp2

            if background is None:
                # Sorting key puts all non-HubbardV in the beginning
                return sorted(v_params, key=lambda x: isinstance(x, HubbardV))

        if standard == background:
            msg = (
                "The standard cannot be the same as the background!\n"
                f"Standard:   '{standard}'\n"
                f"Background: '{background}'"
                )
            raise ValueError(msg)

        # Populate list with defaults
        labeled_params = {i: None for i in range(6)}
        intersite_index = 0
        for param in v_params:
            if isinstance(param, HubbardU):
                labeled_params[0] = param
                continue
            if isinstance(param, HubbardJ):
                labeled_params[1] = param
                continue

            mnfld_1 = HubbardParam._join_lbl_mnfld(
                param.label_1, param.label_1_manifold, param.label_1_manifold_2,
            )
            mnfld_2 = HubbardParam._join_lbl_mnfld(
                param.label_2, param.label_2_manifold, param.label_2_manifold_2,
            )

            if param.is_intersite():
                if mnfld_1 == standard:
                    labeled_params[intersite_index + 1001] = param
                    intersite_index += 1
                else:
                    labeled_params[intersite_index + 10001] = param
                    intersite_index += 1
                continue

            if (mnfld_1, mnfld_2) == (standard, standard):
                if labeled_params.get(0) is not None:
                    msg = "Standard-Standard HubbardV found with existing HubbardU!"
                    raise ValueError(msg)

                labeled_params[2] = param
            elif (mnfld_1, mnfld_2) == (standard, background):
                labeled_params[3] = param
            elif (mnfld_1, mnfld_2) == (background, background):
                labeled_params[4] = param
            elif (mnfld_1, mnfld_2) == (background, standard):
                labeled_params[5] = param
            else:
                msg = (
                    "Unknown relationship detected!\n"
                    f"Parameter: {param.write()}"
                )
                raise ValueError(msg)

        sorted_params = [ # Sort by key, extract only non-None values
            i[1] for i in sorted(labeled_params.items(), key=lambda k: k[0])
            if i[1] is not None
        ]
        return sorted_params

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping,
        phys_sys: PhysicalSystem | None = None,
        **_,
    ) -> list[Self]:
        v_params = []
        keys = (
            "label_1",
            "label_1_manifold",
            "label_1_idx",
            "label_2",
            "label_2_manifold",
            "label_2_idx",
            "v_val",
            "label_1_manifold_2",
            "label_2_manifold_2",
        )
        for (lbl1_mnfld, lbl2_mnfld), info in mapping.items():
            l1, l1m1, l1m2 = HubbardParam._split_lbl_mnfld(lbl1_mnfld)
            l2, l2m1, l2m2 = HubbardParam._split_lbl_mnfld(lbl2_mnfld)
            if isinstance(info, int | float):
                if phys_sys is None:
                    msg = "Must pass `phys_sys` if indices are not specified!"
                    raise ValueError(msg)

                warn(
                    "Automatic determination of indices from a physical system "
                    "is deprecated and will result in an error in the future!",
                    warn_type="deprecate"
                )
                elem = phys_sys.structure.elem
                index1 = np.where(elem == l1)[0] + 1
                index2 = np.where(elem == l2)[0] + 1
                for l1idx, l2idx in product(index1, index2):
                    params = [l1, l1m1, l1idx, l2, l2m1, l2idx, info, l1m2, l2m2]
                    v_params.append(dict(zip(keys, params, strict=True)))
            elif isinstance(info, Sequence):
                for val in info:
                    if "value" not in val:
                        msg = "Missing key 'value' in HubbardV input mapping!"
                        raise ValueError(msg)
                    v_val = val["value"]
                    if (idcs := val.get("indices")) is not None:
                        params = [
                            l1, l1m1, idcs[0], l2, l2m1, idcs[1], v_val, l1m2, l2m2
                        ]
                        v_params.append(dict(zip(keys, params, strict=True)))
                    elif (radius := val.get("radius")) is not None:
                        elem = phys_sys.structure.elem
                        index1 = np.where(elem == l1)[0]
                        index2 = np.where(elem == l2)[0]
                        nn = phys_sys.structure.nearest_neighbors(rmax=radius)
                        pairs = filter(
                            lambda x: x[1] in nn[x[0]], product(index1, index2)
                        )
                        for in1, in2 in pairs:
                            params = [
                                l1, l1m1, in1+1, l2, l2m1, in2+1, v_val, l1m2, l2m2
                            ]
                            v_params.append(dict(zip(keys, params, strict=True)))
                    else:
                        msg = "Must supply 'radius' or 'indices' for HubbardV input mapping!"
                        raise ValueError(msg)
            else:
                msg = "Must provide a list of indices/radii for HubbardV input mapping!"
                raise TypeError(msg)
        items = []
        for kwargs in v_params:
            try:
                items.append(cls(**kwargs))
            except Exception as exc:
                exc.add_note(
                    "HubbardV.from_mapping() failed with args:\n"
                    f"{"\n".join(f"{k}: {v}" for k, v in kwargs.items())}"
                )
                raise
        return items


class Hubbard(CardBase):
    """Hubbard card in a :class:`PwscfInput` [1]_.

    References
    ----------
    .. [1] https://www.quantum-espresso.org/Doc/INPUT_PW.html#id329
    """

    name = "hubbard"

    hub_type_map = MappingProxyType(
        {
            "U":      HubbardU,
            "ALPHA":  HubbardAlpha,
            "J0":     HubbardJ0,
            "J":      HubbardJ,
            "B":      HubbardJ,
            "E2":     HubbardJ,
            "E3":     HubbardJ,
            "V":      HubbardV,
        }
    )

    valid_specifiers = frozenset(
        {"atomic", "ortho-atomic", "norm-atomic", "wf", "pseudo"}
    )

    specifier: HubbardSpecifier
    hubbard_params: list[HubbardParam]

    def __init__(
        self,
        specifier: HubbardSpecifier,
        hubbard_params: list[HubbardParam],
        *,
        disable_sort_and_group: bool = False,
    ) -> None:
        spec_low = specifier.lower()
        if spec_low not in Hubbard.valid_specifiers:
            msg = (
                f"Supplied specifier {specifier} is not known by Nexus!\n"
                f"Valid specifiers are {Hubbard.valid_specifiers}"
            )
            raise ValueError(msg)

        self.specifier = spec_low
        if not all(isinstance(i, HubbardParam) for i in hubbard_params):
            msg = (
                "Invalid type found in `hubbard_params`!\n"
                "Expected instance of one or more of "
                f"{[i.__name__ for i in Hubbard.hub_type_map.values()]}"
            )
            raise TypeError(msg)

        if not disable_sort_and_group:
            self.hubbard_params = Hubbard.group_and_sort(hubbard_params)
        else:
            self.hubbard_params = hubbard_params

    @classmethod
    def read(cls, lines: list[str]) -> Self:
        specifier = lines[0].split()[1].strip("{}()").lower()
        if specifier not in Hubbard.valid_specifiers:
            msg = (
                f"Invalid specifier '{specifier}' for Hubbard card!\n"
                f"Valid specifiers are {sorted(Hubbard.valid_specifiers)}"
            )
            raise ValueError(msg)

        dft_u = False
        dft_uj = False
        dft_uv = False

        hub_types = {i.split()[0].upper() for i in lines[1:]}

        if hub_types.issubset({"U", "J0", "ALPHA"}):
            dft_u = True
        elif hub_types.issubset({"U", "J", "B", "E2", "E3"}):
            dft_uj = True
        elif hub_types.issubset({"U", "J0", "V"}):
            dft_uv = True

        if not any((dft_u, dft_uj, dft_uv)):
            msg = (
                "Could not determine the calculation type!\n"
                f"Detected Hubbard parameter types: {hub_types}"
            )
            raise ValueError(msg)

        if dft_u + dft_uj + dft_uv > 1:
            msg = "Detected incompatible sets of Hubbard parameters!"
            raise ValueError(msg)

        hubbard_params = []
        for idx, line in enumerate(lines[1:]):
            line = line.strip()
            hub_type = line.split()[0].upper()
            try:
                hubbard_params.append(
                    Hubbard.hub_type_map[hub_type].from_str(line)
                )
            except Exception as exc:
                exc.add_note(f"Error occurred on line {idx} of the HUBBARD card")
                raise

        return cls(specifier=specifier, hubbard_params=hubbard_params)

    def write(self) -> str:
        text = f"HUBBARD ({self.specifier})\n"
        for param in self.hubbard_params:
            text += indent(f"{param.write()}", "    ")
        return text

    @staticmethod
    def group_and_sort(hubbard_params: list[HubbardParam]) -> list[HubbardParam]:
        """Group and sort a list of Hubbard parameters.

        Notes
        -----
        If they are present, :class:`HubbardJ` parameters are joined using
        :meth:`HubbardJ.join`, which will take ``J`` and ``B`` or ``E2``/``E3``
        values for the same label-manifold into a single instance of the class.

        After this, parameters are grouped by their label-manifold, even between
        different types of Hubbard parameters.

        For regular DFT+U, :class:`HubbardU`, :class:`HubbardJ0`, and
        :class:`HubbardAlpha` parameters are grouped.

        For DFT+U+J, the :class:`HubbardU` and :class:`HubbardJ` parameters for
        the same label-manifold are grouped.

        For DFT+U+V, :class:`HubbardU`, :class:`HubbardJ0`, and
        :class:`HubbardV` parameters for the same label-manifold are grouped.
        The parameters are sorted by :meth:`HubbardV.sort_by_interaction_type`.
        """
        import pprint
        j_params = [i for i in hubbard_params if isinstance(i, HubbardJ)]
        for p in j_params:
            hubbard_params.remove(p)
        joined_j_params = HubbardJ.join(*j_params)
        hubbard_params.extend(joined_j_params)

        # Track standard (std) and background (bg) manifolds.
        # Enables grouping the bg-bg HubbardV sites.
        std_bg_lbl_mnflds = set()
        bg_std_lbl_mnflds = {}
        lbl_mnfld_params: dict[str, list[HubbardParam]] = {}
        loop_count = 0
        while len(hubbard_params) > 0 and loop_count < 20:
            # This loop removes the first item of the input list, gets its label
            # and manifold (with optional second manifold), and cataloging it in
            # the lbl_mnfld_params dictionary.
            # In the case the parameter is a background-background HubbardV and
            # we don't know what the related standard is, it appends the
            # parameter to the list which guarantees that it will have a known
            # standard by the time it is hit again.
            # In case this logic fails for any reason, the loop count should be
            # incremented and eventually will cause the loop to stop.
            param = hubbard_params.pop(0)
            if isinstance(param, HubbardU | HubbardJ0 | HubbardAlpha | HubbardJ):
                if (
                    isinstance(param, HubbardU | HubbardJ0)
                    and param.manifold_2 is not None
                ):
                    lbl_mnfld = (param.label, param.manifold, param.manifold_2)
                else:
                    lbl_mnfld = (param.label, param.manifold)
                    std_bg_lbl_mnflds.add(lbl_mnfld)

                if lbl_mnfld not in lbl_mnfld_params:
                    lbl_mnfld_params[lbl_mnfld] = []

                lbl_mnfld_params[lbl_mnfld].append(param)
            elif isinstance(param, HubbardV):
                if param.label_1_manifold_2 is not None:
                    lbl_mnfld_1 = (
                        param.label_1, param.label_1_manifold, param.label_1_manifold_2
                    )
                else:
                    lbl_mnfld_1 = (param.label_1, param.label_1_manifold)

                if param.label_2_manifold_2 is not None:
                    lbl_mnfld_2 = (
                        param.label_2, param.label_2_manifold, param.label_2_manifold_2
                    )
                else:
                    lbl_mnfld_2 = (param.label_2, param.label_2_manifold)

                if lbl_mnfld_1 in std_bg_lbl_mnflds and lbl_mnfld_2 not in std_bg_lbl_mnflds:
                    bg_std_lbl_mnflds[lbl_mnfld_2] = lbl_mnfld_1

                if (
                    lbl_mnfld_1 == lbl_mnfld_2
                    and len(lbl_mnfld_1) == 2
                    and not (
                        lbl_mnfld_1 in bg_std_lbl_mnflds
                        or lbl_mnfld_1 in std_bg_lbl_mnflds
                    )
                ):
                    std_bg_lbl_mnflds.add(lbl_mnfld_1)

                if lbl_mnfld_1 in std_bg_lbl_mnflds:
                    standard = lbl_mnfld_1
                elif lbl_mnfld_1 in bg_std_lbl_mnflds:
                    standard = bg_std_lbl_mnflds[lbl_mnfld_1]
                elif lbl_mnfld_2 in std_bg_lbl_mnflds:
                    standard = lbl_mnfld_2
                elif lbl_mnfld_2 in bg_std_lbl_mnflds:
                    standard = bg_std_lbl_mnflds[lbl_mnfld_2]
                else:
                    # This is a bg-bg interaction with an unknown corresponding standard.
                    hubbard_params.append(param)
                    loop_count += 1
                    continue

                if standard not in lbl_mnfld_params:
                    lbl_mnfld_params[standard] = []
                lbl_mnfld_params[standard].append(param)

            loop_count = 0

        if loop_count >= 20 or len(hubbard_params) > 0:
            msg = (
                "Logic for sorting/grouping Hubbard parameters failed!\n"
                "This is a developer error!\n"
                "Current status of grouping:\n"
                f"lbl_mnfld_params = {pprint.pformat(lbl_mnfld_params, indent=4)}\n"
                f"std_bg_lbl_mnflds = {std_bg_lbl_mnflds}\n"
                f"bg_std_lbl_mnflds = {pprint.pformat(bg_std_lbl_mnflds, indent=4)}\n"
                "Remaining parameters:\n"
            )
            for leftover_param in hubbard_params:
                msg += leftover_param.write()
            raise NexusError(msg)

        param_order = list(Hubbard.hub_type_map)
        sorted_params = []
        for keys, values in lbl_mnfld_params.items():
            if any(isinstance(i, HubbardV) for i in values):
                std = HubbardParam._join_lbl_mnfld(*keys)
                sorted_values = HubbardV.sort_by_interaction_type(
                    values,
                    standard=std,
                    background=None,
                )
            else:
                sorted_values = sorted(
                    values,
                    key=lambda x: param_order.index(x.hubbard_type)
                )
            sorted_params.extend(sorted_values)

        return sorted_params

    @classmethod
    def from_dict(
        cls,
        inputs: Mapping[HubbardInteraction, Mapping],
        specifier: HubbardSpecifier,
        phys_sys: PhysicalSystem | None = None,
    ) -> Self:
        """Create an instance of the Hubbard card from a mapping of interaction types.

        Primarily used for backwards-compatibility with Nexus's previous method
        of specifying the Hubbard card.

        Parameters
        ----------
        inputs : Mapping
            A map with keys being interaction types and values being maps of
            manifold label(s) to values, along with any other parameters for
            that interaction type. See the Examples for more.
        specifier : HubbardSpecifier
            A valid :type:`HubbardSpecifier`.
        phys_sys : PhysicalSystem, optional
            An instance of a :class:`PhysicalSystem`. Only required if the
            interactions are not complete (e.g. missing species indices).

        Examples
        --------
        Simple DFT+U input.

        >>> inputs = {
        ...     "U": {"C-2p": 1.0}
        ... }
        >>> Hubbard.from_dict(inputs, specifier="atomic")
        Hubbard(
            specifier = 'atomic',
            hubbard_params = [
                HubbardU(
                    label       = 'C',
                    manifold    = '2p',
                    u_val       = 1.0,
                    manifold_2  = None,
                    eigenstates = None,
                ),
            ],
        )

        >>> inputs = {
        ...     "V" : {("C-2p", "C-2p"): 1e-8}
        ... }

        >>> inputs = {
        ...     "V": {
        ...         ("C-2p", "C-2p"): [
        ...             {"indices": (1, 2), "value": 1e-8},
        ...             {"indices": (2, 1), "value": 1e-8},
        ...         ]
        ...     }
        ... }

        >>> inputs = {
        ...     "V": {
        ...         ("C-2p", "C-2p"): [{"radius": 5.0, "value": 1e-8}]
        ...     }
        ... }

        >>> inputs = {
        ...     "U": {"C-2p": 1.0},
        ...     "V": {("C-2p", "C-2p"): 1e-8},
        ... }

        """
        inputs = {k.upper(): v for k, v in inputs.items()}
        hubbard_params: list[HubbardParam] = []
        for itype, values in inputs.items():
            if itype == "U":
                hubbard_params.extend(HubbardU.from_mapping(values))
            elif itype == "J0":
                hubbard_params.extend(HubbardJ0.from_mapping(values))
            elif itype == "ALPHA":
                hubbard_params.extend(HubbardAlpha.from_mapping(values))
            elif itype == {"J", "B", "E2", "E3"}:
                hubbard_params.extend(HubbardJ.from_mapping(values, itype=itype))
            elif itype == "V":
                hubbard_params.extend(HubbardV.from_mapping(values, phys_sys=phys_sys))
            else:
                msg = (
                    f"Unrecognized interaction type {itype}!\n"
                    f"Known interaction types: {", ".join(cls.hub_type_map)}"
                )
                raise ValueError(msg)
        return cls(specifier, hubbard_params)


    def __repr__(self) -> str:
        text = f"{type(self).__name__}(\n"
        text += f"    specifier = {self.specifier!r},\n"
        text += "    hubbard_params = [\n"
        for param in self.hubbard_params:
            text += f"{indent(repr(param), " " * 8)},\n"
        text += "    ],\n"
        text += ")"
        return text


class PwscfInputParser:

    num_patterns = MappingProxyType(
        {
            # e.g. 1.234e5
            "point": r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?",
            # e.g. 1,234e5
            "comma": r"[+-]?(?:\d+(?:,\d*)?|,\d+)(?:[eEdD][+-]?\d+)?",
        }
    )
    complex_patterns = MappingProxyType(
        {
            # Two floats in parentheses, separated by a comma
            "point": r"\(" + num_patterns["point"] + r" *, *" + num_patterns["point"] + r"\)",
            # Two floats in parentheses, separated by a semicolon
            "comma": r"\(" + num_patterns["comma"] + r" *; *" + num_patterns["comma"] + r"\)"
        }
    )
    string_patterns = MappingProxyType(
        {
            # delim='none' means the strings do not have surrounding single or double quote
            "none": {
                # decimal="point" means it can't contain commas
                "point": r"([^\s=$/&!,]+[\s=$/&!,])",
                # decimal="comma" means it can't contain semicolons
                "comma": r"([^\s=$/&!;]+[\s=$/&!;])",
            },
            # Match anything inside paired quotes (`?s:` means dot matches all, including newline)
            "quotes": r"'(?s:.*?)'|\"(?s:.*?)\"",
        }
    )

    decimal: Literal["point", "comma"]
    delim: Literal["quotes", "none"]
    patterns: dict[Literal["number", "bool", "complex", "string", "group"], Pattern]

    def __init__(
        self,
        decimal: Literal["point", "comma"] = "point",
        delim:   Literal["quotes", "none"] = "quotes",
    ):
        if decimal.lower() not in {"point", "comma"}:
            msg = f"decimal must be one of 'point' or 'comma' but is '{decimal}'!"
            raise ValueError(msg)
        else:
            decimal = decimal.lower()

        if delim.lower() not in {"quotes", "none"}:
            msg = f"delim must be one of 'quotes' or 'none' but is '{delim}'!"
            raise ValueError(msg)
        else:
            delim = delim.lower()

        if delim == "none":
            string_pat = PwscfInputParser.string_patterns[delim][decimal]
        else:
            string_pat = PwscfInputParser.string_patterns[delim]

        self.decimal = decimal
        self.delim = delim
        self.patterns = {
            "number": re.compile(PwscfInputParser.num_patterns[decimal]),
            # e.g. `.true.` or `.false.`, the `?i:` makes it case-insensitive
            "bool": re.compile(r"(?i:(?:\.true\.|\.false\.|\.?[tTfF]))"),
            "complex": re.compile(PwscfInputParser.complex_patterns[decimal]),
            "string": re.compile(string_pat),
            # Use `a-zA-Z` over `\w` beause it can't start with an underscore
            "namelist": re.compile(r"^&[a-zA-Z$][\w$]*"),
        }

    def read_text(self, text: str) -> dict:
        text = re.sub(r"!.*?\n", "\n", text) # Remove all comments
        # Move multiple variables on the same line separated by a comma
        # onto separate lines
        text = re.sub(r",([^0-9\n])", r",\n\1", text)
        nmlist_lines: list[str] = []
        card_lines:   list[str] = []
        at_cards = False
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith(PwscfInput.cards) and "=" not in line:
                at_cards = True

            if not at_cards:
                nmlist_lines.append(line)
            else:
                card_lines.append(line)
        current_group = None
        groups = {}
        for line in nmlist_lines:
            if line.strip() == "":
                continue
            if (nmlist_match := re.search(self.patterns["namelist"], line)) is not None:
                if current_group is not None:
                    msg = "Encountered new namelist before termination of previous namelist!\n"
                    raise FileFormatError(msg)
                current_group = nmlist_match.group()
                if (nmlist := current_group.removeprefix("&").lower()) in PwscfInput.namelists:
                    groups[nmlist] = {}
                    current_group = nmlist
                    continue
                else:
                    msg = f"Encountered unknown namelist '{current_group}'!"
                    raise FileFormatError(msg)
            elif line == "/":
                current_group = None
                continue
            elif current_group is None:
                continue

            key, value = line.split("=")
            key = key.strip().lower()

            try:
                value = self._parse_value(value)
            except Exception as exc:
                exc.add_note(f"Error occurred on line:\n{line}")
                raise

            if "(" in key:
                key, index = key.split("(")
                index = tuple(map(int, index.strip(")").split(",")))
                if len(index) == 1:
                    index = index[0]
                if key not in groups[current_group]:
                    groups[current_group][key] = {}

                groups[current_group][key][index] = value
            else:
                groups[current_group][key] = value

        cards = {}
        current_card = None
        for line in card_lines:
            line = line.strip()
            if line.lower().startswith(PwscfInput.cards):
                current_card = line.lower().split()[0].split("(")[0].split("{")[0]
                if current_card in cards:
                    msg = f"Encountered duplicate instance of '{current_card.title()}' card!"
                    raise FileFormatError(msg)
                cards[current_card] = []

            if current_card is not None and line != "":
                cards[current_card].append(line)

        for card, lines in cards.items():
            groups[card] = PwscfInput.card_types[card].read(lines)

        return groups

    def _parse_value(self, val_str: str) -> str | bool | int | float | complex:
        val_str = val_str.strip(", \n").lower()
        for dtype, pattern in self.patterns.items():
            value = re.match(pattern, val_str)
            if value is None:
                continue
            else:
                value = value.group()

            if dtype == "number":
                value = float(value.replace("d", "e"))
                if value.is_integer():
                    return int(value)
                else:
                    return value
            elif dtype == "bool":
                return value in {".true.", ".t", "t"}
            elif dtype == "complex":
                value = value.strip("()")
                sep = "," if self.decimal == "point" else ";"
                vals = value.split(sep, maxsplit=1)
                return complex(*map(float, vals))
            elif dtype == "string":
                if self.delim == "quotes":
                    return value.strip("'").strip('"')
                else:
                    return value
            elif dtype == "group":
                msg = "Invalid match found!"
                raise RuntimeError(msg)

        warn(f"Unknown type for value '{val_str}', assuming `str`!")
        return val_str


class PwscfInput(SimulationInput):

    required_elements: ClassVar = (
        'control',
        'system',
        'electrons',
        'atomic_species',
        'atomic_positions',
        'k_points',
    )

    parser: ClassVar = PwscfInputParser()
    phys_sys: PhysicalSystem
    control: Control
    system: System
    electrons: Electrons
    ions: Ions | None
    cell: Cell | None
    fcp: Fcp | None
    rism: Rism | None
    atomic_species: AtomicSpecies
    atomic_positions: AtomicPositions
    k_points: KPoints
    cell_parameters: CellParameters | None
    occupations: Occupations | None
    constraints: Constraints | None
    # atomic_velocities: AtomicVelocities | None
    atomic_forces: AtomicForces | None
    # solvents: Solvents | None
    hubbard: Hubbard | None

    namelist_types: ClassVar = MappingProxyType(
        {
            "control"   : Control,
            "system"    : System,
            "electrons" : Electrons,
            "ions"      : Ions,
            "cell"      : Cell,
            "fcp"       : Fcp,
            "rism"      : Rism,
        }
    )
    namelists: ClassVar[tuple[NamelistName]] = tuple(namelist_types.keys())

    card_types: ClassVar = MappingProxyType(
        {
            "atomic_species"   : AtomicSpecies,
            "atomic_positions" : AtomicPositions,
            "k_points"         : KPoints,
            "cell_parameters"  : CellParameters,
            "occupations"      : Occupations,
            "constraints"      : Constraints,
            "atomic_forces"    : AtomicForces,
            "hubbard"          : Hubbard,
        }
    )
    cards: ClassVar[tuple[CardName]] = tuple(card_types)
    ordered_elements: ClassVar = namelists + cards
    # allowed_attrs: ClassVar = frozenset({*ordered_elements, "phys_sys"})

    creatable_from_phys_sys = frozenset(
        {"atomic_species", "atomic_positions", "k_points", "cell_parameters"}
    )

    def __init__(
        self,
        *,
        filepath: PathLike | None = None,
        check: bool = False,
        version: str | None = None,
        phys_sys: PhysicalSystem | None = None,
        verbose: bool = False,
        strict: bool = False,
        pseudos: PseudoSet | None = None,
        **kwargs: PwscfInputType | CardBase | NamelistBase,
    ):
        if filepath is None and len(kwargs) == 0:
            return

        if filepath is not None:
            filepath = Path(filepath).resolve(strict=True)
            self.read_text(filepath.read_text())

        kwargs = {k.lower(): v for k, v in kwargs.items()}
        group_kwargs, leftover_kwargs = self._separate_group_kwargs(kwargs)
        for group in self.ordered_elements:
            if group not in group_kwargs:
                continue
            self._process_group(
                group,
                group_kwargs.pop(group),
                check=check,
                version=version,
                phys_sys=phys_sys,
                verbose=verbose,
                strict=strict,
            )

        if len(leftover_kwargs) != 0:
            msg = (
                f"{len(leftover_kwargs)} unprocessed keyword arguments remain!\n"
                "This is a developer error, please report!\n"
                f"Remaining keywords: {[*leftover_kwargs]}"
            )
            raise NexusError(msg)

        if not self.has_required_elements():
            missing_elems = set(PwscfInput.required_elements) - set(vars(self))
            if  len(missing := missing_elems - PwscfInput.creatable_from_phys_sys) > 0:
                current = indent("\n".join(vars(self)), "  ")
                missing = indent("\n".join(missing), "  ")
                msg = (
                    "Not all required groups have been supplied or can be created!\n"
                    "Current groups:\n"
                    f"{current}\n"
                    "Missing groups:\n"
                    f"{missing}"
                )
                raise ValueError(msg)

            for card_name in missing_elems:
                card_type = PwscfInput.card_types[card_name]
                card = card_type.from_phys_sys(phys_sys, pseudos=pseudos)
                self._process_group(
                    group=card_name,
                    value=card,
                    check=check,
                    version=version,
                    phys_sys=phys_sys,
                    verbose=verbose,
                    strict=strict,
                )

    def _separate_group_kwargs[T](
        self, kwargs: dict[str, T]
    ) -> tuple[dict[NamelistName | CardName, dict[str, T]], dict[str, T]]:
        """Separate out kwargs per namelist/card.

        Returns map from namelist/card name to their input args, along with any
        leftovers.
        """
        group_kwargs = {}
        for nmlist_name, nmlist_type in self.namelist_types.items():
            if nmlist_name in kwargs:
                group_kwargs[nmlist_name] = kwargs.pop(nmlist_name)
                continue

            current = {}
            for var in nmlist_type.namelist_def:
                if (var_name := var.input_name.lower()) in kwargs:
                    current[var_name] = kwargs.pop(var_name)

            if len(current) > 0:
                group_kwargs[nmlist_name] = current

        for card_name in self.card_types:
            if (card_val := kwargs.pop(card_name, unset)) is not unset:
                group_kwargs[card_name] = card_val

        return group_kwargs, kwargs # Leftovers

    def _process_group(
        self,
        group: CardName | NamelistName,
        value: CardBase | NamelistBase | Mapping,
        *,
        check: bool,
        version: str | None,
        phys_sys: PhysicalSystem | None,
        verbose: bool,
        strict: bool,
    ) -> None:
        if group in self.namelists:
            group_class = self.namelist_types[group]
        else:
            group_class = self.card_types[group]

        if isinstance(value, group_class):
            setattr(self, group, value)
        elif isinstance(value, Mapping):
            if group_class in self.namelist_types.values():
                namelist = group_class(
                    check=check,
                    version=version,
                    phys_sys=phys_sys,
                    verbose=verbose,
                    strict=strict,
                    **value,
                )
                if phys_sys is not None:
                    namelist.incorporate_system(phys_sys)
                setattr(self, group, namelist)
            else:
                card = group_class(**value)
                setattr(self, group, card)
        else:
            msg = (
                f"Invalid value type '{type(value).__name__}'!\n"
                "This is a developer error!"
            )
            raise NexusError(msg)

    def has_required_elements(self) -> bool:
        return all(hasattr(self, elem) for elem in PwscfInput.required_elements)

    def read_text(self, contents: str, filepath = None) -> None:  # noqa: ARG002
        self.__init__(**PwscfInput.parser.read_text(contents))

    def write_text(self, filepath = None) -> str:  # noqa: ARG002
        sections = {
            s: v for s, v in vars(self).items() if s in PwscfInput.ordered_elements
        }
        sections = [
            i[1] for i in sorted(
                sections.items(), key=lambda x: PwscfInput.ordered_elements.index(x[0])
            )
        ]
        text = ""
        for section in sections:
            text += section.write() + "\n"

        return text.strip() + "\n"

    @property
    def alat(self) -> float:
        """Either ``celldm(1)`` or 1.0"""
        if (celldm := self.system.params.get("celldm")) is not None:
            return celldm[1]
        else:
            return 1.0

    @property
    def axes(self) -> npt.NDArray[np.float64]:
        if self.system.ibrav == 0:
            return self.alat * np.asarray(self.cell_parameters.vectors, dtype=float)
        elif self.system.ibrav == 1:
            return self.alat * np.eye((3, 3), dtype=float)
        else:
            msg = f"Can not retrieve axes with `{self.system.ibrav=}`!"
            raise ValueError(msg)

    @property
    def kaxes(self) -> npt.NDArray[np.float64]:
        return 2 * pi * inv(self.axes).T

    def incorporate_hubbard(self, hubbard_result: str):
        hubbard_result = hubbard_result.splitlines()
        self.hubbard = Hubbard.read(hubbard_result)

    def incorporate_system(self,system,elem_order=None):
        system.check_folded_system()
        system.change_units('B')
        s  = system.structure
        nc = system.net_charge

        self.system.ibrav        = 0
#        self.system['celldm(1)'] = 1.0e0
        nions = system.n_ions
        nspecies = system.n_species
        self.system.nat          = nions
        self.system.ntyp         = nspecies
        self.system.tot_charge   = nc

        if 'cell_parameters' not in self:
            self.cell_parameters = self.element_types['cell_parameters']()
        #end if
        self.cell_parameters.specifier = 'bohr'
        self.cell_parameters.vectors   = s.axes.copy()

        self.k_points.clear()
        nkpoints = len(s.kpoints)
        if nkpoints>0:
            if s.at_Gpoint():
                self.k_points.specifier = 'gamma'
            elif s.at_Lpoint():
                self.k_points.update(
                    specifier = 'automatic',
                    grid  = (1,1,1),
                    shift = (1,1,1)
                    )
            else:
                kpoints = s.kpoints/(2*pi)
                self.k_points.specifier = 'tpiba'
                self.k_points.nkpoints  = nkpoints
                self.k_points.kpoints   = kpoints
                self.k_points.weights   = s.kweights.copy()
                self.k_points.change_specifier('crystal',self) #added to make debugging easier
            #end if
        #end if

        if 'masses' not in self.atomic_species:
            self.atomic_species.masses = obj()
        #end if
        for name in system.ion_labels:
            _, element = Elements.is_element(name, return_element=True)
            self.atomic_species.masses[name] = element.atomic_weight
        #end for
        if elem_order is None:
            self.atomic_species.atoms = sorted(system.ion_labels)
        else:
            if set(elem_order)!=set(system.ion_labels):
                msg = (
                    'elem_order is missing some atomic species\n'
                    f'atomic species present: {sorted(system.ion_labels)}\n'
                    f'elem_order: {elem_order}'
                    )
                raise ValueError(msg)
            elif len(elem_order)!=system.n_ions:
                msg = (
                    'elem_order has repeated elements\n'
                    f'elem_order: {elem_order}'
                    )
                raise ValueError(msg)
            #end if
            self.atomic_species.atoms = list(elem_order)
        #end if

        # set pseudopotentials for renamed atoms (e.g. Cu3 is same as Cu)
        pp = self.atomic_species.pseudos
        for atom in self.atomic_species.atoms:
            if atom not in pp:
                iselem,element = Elements.is_element(atom,return_element=True)
                symbol = element.symbol
                if iselem and symbol in pp:
                    pp[atom] = str(pp[symbol])
                #end if
            #end if
        #end for

        self.atomic_positions.specifier = 'bohr'
        self.atomic_positions.positions = s.pos.copy()
        self.atomic_positions.atoms     = list(s.elem)
        if s.frozen is not None:
            frozen = s.frozen
            if 'relax_directions' in self.atomic_positions:
                relax_directions = self.atomic_positions.relax_directions
            else:
                relax_directions = np.ones(s.pos.shape,dtype=int)
            #end if
            for i in range(len(s.pos)):
                relax_directions[i,0] = int(not frozen[i,0] and relax_directions[i,0])
                relax_directions[i,1] = int(not frozen[i,1] and relax_directions[i,1])
                relax_directions[i,2] = int(not frozen[i,2] and relax_directions[i,2])
            #end for
            self.atomic_positions.relax_directions = relax_directions
        #end if

    def incorporate_system_old(self,system,spin_polarized=None):
        system.check_folded_system()
        system.change_units('B')
        s  = system.structure
        nc = system.net_charge

        nup = system.n_up
        ndn = system.n_down

        self.system.ibrav        = 0
#        self.system['celldm(1)'] = 1.0e0
        nions = system.n_ions
        nspecies = system.n_species
        self.system.nat          = nions
        self.system.ntyp         = nspecies
        #self.system.nelec        = nup+ndn
        self.system.tot_charge   = nc
        mag = nup-ndn
        if (mag!=0 and spin_polarized!=False) or spin_polarized==True:
            self.system.nspin = 2
            self.system.tot_magnetization = mag
        #end if

        if 'cell_parameters' not in self:
            self.cell_parameters = self.element_types['cell_parameters']()
        #end if
        self.cell_parameters.specifier = 'bohr'
        self.cell_parameters.vectors   = s.axes.copy()

        self.k_points.clear()
        nkpoints = len(s.kpoints)
        if nkpoints>0:
            if s.at_Gpoint():
                self.k_points.specifier = 'gamma'
            elif s.at_Lpoint():
                self.k_points.update(
                    specifier = 'automatic',
                    grid  = (1,1,1),
                    shift = (1,1,1)
                    )
            else:
                kpoints = s.kpoints/(2*pi)
                self.k_points.specifier = 'tpiba'
                self.k_points.nkpoints  = nkpoints
                self.k_points.kpoints   = kpoints
                self.k_points.weights   = s.kweights.copy()
                self.k_points.change_specifier('crystal',self) #added to make debugging easier

            #end if
        #end if

        masses = obj()
        for name in system.ion_labels:
            _, element = Elements.is_element(name, return_element=True)
            masses[name] = element.atomic_weight
        #end for
        self.atomic_species.atoms  = sorted(system.ion_labels)
        self.atomic_species.masses = masses
        # set pseudopotentials for renamed atoms (e.g. Cu3 is same as Cu)
        pp = self.atomic_species.pseudos
        for atom in self.atomic_species.atoms:
            if atom not in pp:
                iselem,element = Elements.is_element(atom,return_element=True)
                symbol = element.symbol
                if iselem and symbol in pp:
                    pp[atom] = str(pp[symbol])
                #end if
            #end if
        #end for

        self.atomic_positions.specifier = 'bohr'
        self.atomic_positions.positions = s.pos.copy()
        self.atomic_positions.atoms     = list(s.elem)
        if s.frozen is not None:
            frozen = s.frozen
            if 'relax_directions' in self.atomic_positions:
                relax_directions = self.atomic_positions.relax_directions
            else:
                relax_directions = np.ones(s.pos.shape,dtype=int)
            #end if
            for i in range(len(s.pos)):
                relax_directions[i,0] = int(not frozen[i,0] and relax_directions[i,0])
                relax_directions[i,1] = int(not frozen[i,1] and relax_directions[i,1])
                relax_directions[i,2] = int(not frozen[i,2] and relax_directions[i,2])
            #end for
            self.atomic_positions.relax_directions = relax_directions
        #end if

    def return_system(
        self,
        *,
        structure_only: bool = False,
        **valency: int | float,
    ) -> Structure | PhysicalSystem:
        if (ibrav := self.system.ibrav) not in {0, 1}:
            msg = f"Can not handle `{ibrav=}`, only `ibrav=0` and `ibrav=1`!"
            raise NotImplementedError(msg)

        scale = self.alat
        axes = self.axes

        elem = list(self.atomic_positions.atoms)
        ap = self.atomic_positions.change_specifier("bohr", self, in_place=False)
        pos = ap.positions

        kp = self.k_points.change_specifier("tpiba", self, in_place=False)
        kpoints = kp.kpoints * (2 * pi) / scale

        center = axes.sum(0) / 2
        structure = Structure(
            axes    = axes,
            elem    = elem,
            scale   = scale,
            pos     = pos,
            center  = center,
            kpoints = kpoints,
            units   = 'B',
            rescale = False
            )
        structure.zero_corner()
        structure.recenter()

        if structure_only:
            return structure

        ion_charge = 0
        atoms = list(self.atomic_positions.atoms)
        for atom in self.atomic_species.atoms:
            if atom not in valency:
                msg = (
                    f"Valence charge for atom `{atom}` has not been defined!\n"
                    "Please provide the valence charge as an argument to return_system()"
                    )
                raise ValueError(msg)

            ion_charge += atoms.count(atom) * valency[atom]

        if "nelup" in self.system:
            nup = self.system.nelup
            ndn = self.system.neldw
            net_charge = ion_charge - nup - ndn
            net_spin   = nup - ndn
        elif "tot_magnetization" in self.system:
            net_spin = self.system.tot_magnetization
            if "nelec" in self.system:
                net_charge = ion_charge - self.system.nelec
            else:
                net_charge = 0
        else:
            net_spin = 0
            if "nelec" in self.system:
                net_charge = ion_charge - self.system.nelec
            else:
                net_charge = 0

        system = PhysicalSystem(
            structure,
            net_charge,
            net_spin,
            **valency,
        )

        return system


def generate_pwscf_input(
    selector: Literal["generic"] = "generic",
    **kwargs: PwscfInputType | CardBase | NamelistBase,
) -> PwscfInput:
    if selector.lower() != "generic":
        warn("Selectors other than generic are deprecated, defaulting to 'generic'")

    return generate_any_pwscf_input(**kwargs)


def generate_any_pwscf_input(
    *,
    system: PhysicalSystem | None = None,
    pseudos: PseudoSet | Sequence[str] = (),
    prefix: str = "pwscf",
    outdir: str | PathLike = "pwscf_output",
    pseudo_dir: str | PathLike = "./",
    kgrid: npt.ArrayLike[np.integer] | None = None,
    kshift: Sequence[Literal[0, 1]] = (0, 0, 0),
    nogamma: bool = False,
    use_folded: bool = True,
    ksymm_run: bool = False,
    bandfac: float | None = None,
    defaults: Mapping | None = None,
    version: str | None = None,
    check: bool = False,
    verbose: bool = False,
    strict: bool = False,
    control_namelist:   Control   | None = None,
    system_namelist:    System    | None = None,
    electrons_namelist: Electrons | None = None,
    ions_namelist:      Ions      | None = None,
    cell_namelist:      Cell      | None = None,
    fcp_namelist:       Fcp       | None = None,
    rism_namelist:      Rism      | None = None,
    atomic_species_card:   AtomicSpecies   | None = None,
    atomic_positions_card: AtomicPositions | None = None,
    atomic_forces_card:    AtomicForces    | None = None,
    kpoints_card:          KPoints         | None = None,
    cell_parameters_card:  CellParameters  | None = None,
    constraints_card:      Constraints     | None = None,
    occupations_card:      Occupations     | None = None,
    hubbard_card:          Hubbard         | None = None,
    at_pos_specifier: AtomicPositionsSpecifier | None = None,
    kpts_specifier: KPointsSpecifier | None = None,
    cell_params_specifier: CellParametersSpecifier | None = None,
    hubbard_specifier: HubbardSpecifier | None = None,
    **kwargs,
) -> PwscfInput:

    # Setup for k-point symmetry run
    #   ecutwfc is set to 1 so that pw.x will crash after initialization
    #   Symmetrized k-points will still be written to log output
    if ksymm_run:
        kwargs["ecutwfc"]    = 1
        kwargs["nosym"]      = False
        kwargs["use_folded"] = False

    # Assign default values
    if defaults is not None:
        if isinstance(defaults, str):
            warn(
                "Pre-set defaults are deprecated and will be removed in the future!",
                warn_type="deprecate",
            )
            if defaults == "standard":
                pass
            elif defaults not in LEGACY_GENERATE_DEFAULTS:
                msg = (
                    f"Invalid default set requested: {defaults}\n"
                    f"Valid options are {sorted(LEGACY_GENERATE_DEFAULTS.keys())}"
                    )
                raise ValueError(msg)
            else:
                defaults = LEGACY_GENERATE_DEFAULTS[defaults]
        elif not isinstance(defaults, Mapping):
            msg = "Defaults must be a dict mapping keywords to values!"
            raise TypeError(msg)
        if defaults != "standard":
            for name, default in defaults.items():
                if name in kwargs:
                    continue

                if inspect.isclass(default) or inspect.isfunction(default):
                    kwargs[name] = default()
                else:
                    kwargs[name] = default

    if ksymm_run and kwargs.get("calculation", "scf") != "scf":
        msg = "Input parameter 'calculation' must be set to 'scf' when ksymm_run is requested"
        raise ValueError(msg)

    # Pre 7.1 Hubbard tags
    hub_keys_pre71 = ("hubbard_u", "hubbard_j0", "hubbard_j", "u_projection_type")
    has_pre71_keys = any(_ in kwargs for _ in hub_keys_pre71)
    # QE >=7.1 Hubbard tags
    hub_keys_v71 = ("hubbard", "hubbard_proj")
    has_v71_keys = any(_ in kwargs for _ in hub_keys_v71)
    if has_pre71_keys and has_v71_keys:
        msg = (
            "Found incompatible mix of Hubbard keys!"
            f"Please use {hub_keys_pre71} for QE version <7.1 and {hub_keys_v71} for QE version >=7.1"
        )
        raise ValueError(msg)

    if at_pos_specifier is None:
        at_pos_specifier = _get_card_specifier(
            kwargs,
            ("atomic_positions", "pos", "positions", "at_pos"),
            preferred_name="at_pos_specifier",
        )
    if kpts_specifier is None:
        kpts_specifier = _get_card_specifier(
            kwargs,
            ("kpoints", "k_points", "kpts"),
            preferred_name="kpts_specifier",
        )
    if cell_params_specifier is None:
        cell_params_specifier = _get_card_specifier(
            kwargs,
            ("cell", "cell_parameters", "cell_params"),
            preferred_name="cell_params_specifier",
        )

    if system is None:
        msg = "You must provide `{}` if you do not provide a physical system!"
        atoms = kwargs.pop("atoms") if "atoms" in kwargs else kwargs.pop("elem", None)
        if atoms is None:
            msg = msg.format("atoms")
            raise ValueError(msg)

        if (positions := kwargs.pop("positions", None)) is not None:
            pos = positions
        else:
            pos = kwargs.pop("pos", None)

        force_multipliers = kwargs.pop("force_multipliers", None)

        if pos is None:
            msg = msg.format("positions")
            raise ValueError(msg)

        if at_pos_specifier is None:
            msg = msg.format("at_pos_specifier")
            raise ValueError(msg)

        species = set(atoms)
        if (elem_order := kwargs.pop("elem_order", None)) is not None:
            if len(missing := set(elem_order) - species) > 0:
                msg = (
                    "`elem_order` is missing some atomic species!\n"
                    f"Missing species: {",".join(missing)}"
                )
                raise ValueError(msg)
            elif len(elem_order) != len(species):
                duplicates = [(i[0], next(i[1])) for i in groupby(elem_order, len) if i[0] > 1]
                msg = (
                    "Repeated elements in `elem_order` not allowed!\n"
                    f"Duplicates (count, name): {", ".join(map(str, duplicates))}"
                    )
                raise ValueError(msg)
            ordered_species = list(elem_order)
        else:
            ordered_species = sorted(species, key=atoms.index)

        atomic_species_card = AtomicSpecies(
            atoms=ordered_species,
            masses=kwargs.pop("mass") if "mass" in kwargs else kwargs.pop("masses", None),
            pseudos=pseudos,
        )
        atomic_positions_card = AtomicPositions(
            atoms=atoms,
            positions=np.asarray(pos, dtype=float),
            force_multipliers=force_multipliers,
            specifier=at_pos_specifier,
        )
    else:
        system.check_folded_system()
        system.change_units("B")
        structure = system.structure
        # Setting the 'lattice' (cell axes) requires some delicate care
        #   QMCPACK will fail if this is even 1e-10 off of what is in
        #   the wavefunction hdf5 file from QE
        if structure.folded_structure is not None:
            fs = structure.folded_structure
            axes = np.asarray(fs.axes, dtype=float)
            axes = np.dot(structure.tmatrix, axes)
            if not np.allclose(axes, structure.axes, rtol=0, atol=1e-10):
                with np.printoptions(precision=12, floatmode="fixed"):
                    msg = (
                        "Supercell axes do not match tiled version of folded cell axes\n"
                        "You may have changed one set of axes (super/folded) and not the other\n"
                        "Folded cell axes:\n"
                        f"{fs.axes}\n"
                        "Supercell axes:\n"
                        f"{structure.axes}\n"
                        "Folded axes tiled:\n"
                        f"{axes}"
                    )
                    raise ValueError(msg)
        else:
            axes = np.asarray(structure.axes, dtype=float)

        structure.adjust_axes(axes)
        if use_folded:
            system = system.get_smallest()

    totmag_sys = kwargs.pop("totmag_sys", False)
    if totmag_sys:
        if "tot_magnetization" in kwargs:
            LOGGER.debug("`totmag_sys` ignored because `tot_magnetization` was provided.")
        kwargs["tot_magnetization"] = system.net_spin

    # Set the number of spins (tot_magnetization=0 still needs nspin=2)
    if kwargs.get("nspin") is not None:
        if kwargs.get("start_mag") is None and kwargs.get("tot_magnetization") is None:
            kwargs["nspin"] = 1
        else:
            kwargs["nspin"] = 2

    if bandfac is not None:
        if kwargs.get("nbnd") is not None:
            LOGGER.debug("Ignoring `bandfac` when `nbnd` was provided.")
        else:
            nocc = max(system.n_up, system.n_down)
            kwargs["nbnd"] = int(np.ceil(nocc * bandfac))

    if kwargs.get("hubbard_u") is not None and not kwargs.get("lda_plus_u", False):
        LOGGER.debug("Setting `lda_plus_u` to True because `hubbard_u` was provided")
        kwargs["lda_plus_u"] = True

    if (start_mag := kwargs.get("start_mag")) is not None:
        kwargs["starting_magnetization"] = start_mag

    if kpts_specifier == "gamma" and nogamma:
        msg = "Can not specify `kpts_specifier='gamma'` and set `nogamma=True`!"
        raise ValueError(msg)

    if kpoints_card is None and kpts_specifier is not None:
        # We will faithfully construct the card if the user provides the necessary info
        if kpts_specifier == "gamma":
            kpoints_card = KPoints(kpts_specifier)
        elif kpts_specifier == "automatic" and not (kgrid is None or kshift is None):
            kpoints_card = KPoints(kpts_specifier, grid=kgrid, shift=kshift)
        elif not (
            (kpoints := kwargs.get("kpoints")) is None
            or (kweights := kwargs.get("kweights")) is None
        ):
            kpoints_card = KPoints(kpts_specifier, kpoints=kpoints, kweights=kweights)

    if kpoints_card is None:
        # For some reason the legacy code likes to mangle user-supplied kpoint info
        # so that it is set to the gamma point if they fall into one of two cases:
        #   1) No PhysicalSystem, no kgrid, none/zero kshift
        #   2) kgrid produces 1 kpoint, none/zero kshift
        # Both are overridden if `nogamma=True`
        no_kpt_info = kgrid is None and system is None
        auto = kgrid is not None and kshift is not None

        zero_shift = kshift is None or tuple(kshift) == (0, 0, 0)
        single_kpoint = kgrid is not None and tuple(kgrid) == (1, 1, 1)

        at_gamma = zero_shift and (single_kpoint or no_kpt_info)

        if at_gamma and not nogamma:
            kpoints_card = KPoints(specifier="gamma")
        elif auto:
            kpoints_card = KPoints(
                specifier="automatic",
                grid=kgrid,
                shift=kshift,
            )
        elif at_gamma and nogamma:
            kpoints_card = KPoints(
                specifier="automatic",
                grid=(1, 1, 1),
                shift=(0, 0, 0),
            )

    # Occupations card
    occupations = kwargs.get("occupations")
    if occupations is None and occupations_card is not None:
        LOGGER.debug(
            "Setting `occupations='from_input'` because `occupations_card` was given"
        )
        kwargs["occupations"] = "from_input"
    elif (
        occupations_card is not None
        and occupations != "from_input"
    ):
        msg = (
            f"`{occupations=}` but must be 'from_input' if `occupations_card` is provided!"
        )
        raise ValueError(msg)
    elif isinstance(occupations, Sequence | np.ndarray):
        kwargs["occupations"] = "from_input"
        occupations_card = Occupations(
            occupations=occupations,
            minority_spin_occs=kwargs.pop("minority_spin_occs", None),
        )

    if (
        hubbard_card is None
        and (hubbard_inputs := kwargs.pop("hubbard", None)) is not None
    ):
        if hubbard_specifier is None:
            hubbard_specifier = _get_card_specifier(
                kwargs,
                ("hubbard", "hub"),
                preferred_name="hubbard_specifier",
                suffixes=("proj", "specifier"),
            )

        if hubbard_specifier is None:
            warn(
                "Hubbard specifier not specified, defaulting to 'atomic'. "
                "In the future, this will not be allowed!"
            )
            hubbard_specifier = "atomic"
        hubbard_card = Hubbard.from_dict(
            hubbard_inputs,
            specifier=hubbard_specifier,
            phys_sys=system,
        )

    pwscf_input = PwscfInput(
        check=check,
        version=version,
        phys_sys=system,
        verbose=verbose,
        strict=strict,
        pseudos=pseudos,

    )

    return pwscf_input
#end def generate_any_pwscf_input

@nxs_deprecate(since="2.4.9", replacement="generate_pwscf_input")
def generate_scf_input(**kwargs) -> PwscfInput:
    return generate_pwscf_input(**kwargs)

@nxs_deprecate(since="2.4.9", replacement="generate_pwscf_input")
def generate_nscf_input(**kwargs) -> PwscfInput:
    return generate_pwscf_input(**kwargs)

@nxs_deprecate(since="2.4.9", replacement="generate_pwscf_input")
def generate_relax_input(**kwargs):
    return generate_pwscf_input(**kwargs)

@nxs_deprecate(since="2.4.9", replacement="generate_pwscf_input")
def generate_vcrelax_input(**kwargs) -> PwscfInput:
    return generate_pwscf_input(**kwargs)

LEGACY_GENERATE_DEFAULTS = {
    "oldscf": {
        "calculation":      'scf',
        "prefix":           'pwscf',
        "outdir":           'pwscf_output',
        "pseudo_dir":       './',
        "ecutwfc":          200.,
        "occupations":      'smearing',
        "smearing":         'fermi-dirac',
        "degauss":          0.0001,
        "nosym":            False,
        "wf_collect":       True,
        "restart_mode":     'from_scratch',
        "tstress":          True,
        "tprnfor":          True,
        "disk_io":          'low',
        "verbosity":        'high',
        "ibrav":            0,
        "conv_thr":         1e-10,
        "electron_maxstep": 1000,
        "mixing_mode":      'plain',
        "mixing_beta":      .7,
        "diagonalization":  'david',
        "kgrid":            None,
        "kshift":           None,
        "use_folded":       True,
    },
    "oldrelax": {
        "calculation":       'relax',
        "prefix":            'pwscf',
        "outdir":            'pwscf_output',
        "pseudo_dir":        './',
        "ecutwfc":           50.,
        "occupations":       'smearing',
        "smearing":          'fermi-dirac',
        "degauss":           0.0001,
        "nosym":             True,
        "wf_collect":        False,
        "restart_mode":      'from_scratch',
        "disk_io":           'low',
        "verbosity":         'high',
        "ibrav":             0,
        "conv_thr":          1e-6,
        "electron_maxstep":  1000,
        "mixing_mode":       'plain',
        "mixing_beta":       .7,
        "diagonalization":   'david',
        "ion_dynamics":      'bfgs',
        "upscale":           100,
        "pot_extrapolation": 'second_order',
        "wfc_extrapolation": 'second_order',
        "kgrid":             None,
        "kshift":            None,
        "use_folded":        False,
    },
}

def _get_card_specifier(
    kwargs: dict[str, PwscfInputType],
    prefixes: tuple[str],
    preferred_name: str,
    *,
    suffixes: tuple[str] = ("specifier", "option")
) -> str | None:
    """Attempt to retrieve specifier via alternative prefixes/suffixes.

    This will use ``dict.pop()``, and thus modifies ``kwargs`` in-place!!
    """
    for prefix in prefixes:
        for suffix in suffixes:
            if (key := f"{prefix}_{suffix}") in kwargs:
                warn(
                    f"Passing specifier as '{key}' is deprecated, "
                    f"please use {preferred_name} instead!",
                    warn_type="deprecate"
                )
                return kwargs.pop(key)

    return None
