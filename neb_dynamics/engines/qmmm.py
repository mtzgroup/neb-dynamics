from neb_dynamics.engines.engine import Engine
from neb_dynamics.errors import GradientsNotComputedError, ElectronicStructureError
from dataclasses import dataclass
from pathlib import Path
from neb_dynamics.TreeNode import TreeNode
import neb_dynamics.chainhelpers as ch
from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs, RunInputs
from neb_dynamics.neb import NEB
from neb_dynamics.msmep import MSMEP
from neb_dynamics import StructureNode
from neb_dynamics.elementarystep import check_if_elem_step
from qccodec.parsers.terachem import parse_energy, parse_gradient
from qcio import FileInput, view
from chemcloud import compute as ccompute

from qcop import compute
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
import numpy as np
from neb_dynamics.elements import atomic_number_to_symbol
from neb_dynamics.geodesic_interpolation2.coord_utils import align_geom
import numpy as np
import os

from qcio import Structure, ProgramOutput
from neb_dynamics.helper_functions import parse_symbols_from_prmtop, rst7_to_coords_and_indices, parse_qmmm_gradients

from pathlib import Path

import matplotlib.pyplot as plt

@dataclass
class QMMMEngine(Engine):
    """"
    This is what initialization looks like:

    >>> eng = QMMMEngine(tcin_fp=Path("./tc.in"),
                qminds_fp=Path("./qmindices.dat"),
                prmtop_fp=Path("./ref.prmtop"),
                rst7_fp_prod=Path("./optim_product.rst7"),
                rst7_fp_react=Path("./ref.rst7"))
    """
    tcin_fp: Path = None
    qminds_fp: Path = None
    prmtop_fp: Path = None
    rst7_fp_prod: Path = None
    rst7_fp_react: Path = None
    compute_program: str = "qcop"

    def __post_init__(self):
        self.inp_file = self.tcin_fp.read_text()  # Or your own function to create tc.in
        self.qmindices = self.qminds_fp.read_text()
        self.prmtop = self.prmtop_fp.read_text()
        self.ref_rst7_react = self.rst7_fp_react.read_text()
        self.ref_rst7_prod = self.rst7_fp_prod.read_text()
        _, indices_coordinates = rst7_to_coords_and_indices(self.ref_rst7_react)
        self.indices_coordinates = indices_coordinates

    def _construct_input(self, rst7_string):
        # Create a FileInput object for TeraChem
        file_inp = FileInput(
            files={
                "tc.in": self.inp_file,
                "ref.rst7": rst7_string,
                "qmindices.dat": self.qmindices,
                "ref.prmtop": self.prmtop,
            },
            cmdline_args=["tc.in"],
        )
        return file_inp

    def _compute_enegrad(self, rst7_strings):

        inputs = [self._construct_input(string) for string in rst7_strings]
        qminds = [int(x) for x in self.qmindices.strip().split("\n")]


        if self.compute_program.lower() != 'chemcloud':
            outputs = []
            for string in inputs:
                outputs.append(compute("terachem", string, print_stdout=False))
        else:
            outputs = ccompute("terachem", inputs)

        gradients = []
        energies = []
        for output in outputs:
            qm_grad, mm_grad = parse_qmmm_gradients(output.stdout)
            if len(qm_grad) == 0:
                print("SOMETHING FAILED")
                print(output.stdout)
            lines = output.stdout.split("\n")
            nlink_atom = int([l for l in lines if "link" in l][0].split()[6])
            if nlink_atom > 0:
                qm_grad = qm_grad[:-nlink_atom] # Get rid of linked-atoms

            gradient = np.zeros((len(qm_grad)+len(mm_grad), 3))
            # print("len qmgrad:", len(qm_grad))
            # print("len mmgrad:", len(mm_grad))
            allinds = list(range(len(gradient)))
            mminds = np.delete(allinds, qminds)
            gradient[qminds] = qm_grad
            if len(mminds) > 0:
                gradient[mminds] = mm_grad

            energy = parse_energy(output.stdout) # Parses FINAL ENERGY: line

            energies.append(energy)
            gradients.append(gradient)

        return energies, gradients, outputs

    def compute_energies(self, chain: Chain):
        self.compute_gradients(chain)
        enes = np.array([node.energy for node in chain])

        return enes

    def compute_gradients(self, chain):
        try:
            grads = np.array([node.gradient for node in chain])

        except GradientsNotComputedError:
            qminds = [int(v) for v in self.qmindices.strip().split("\n")]

            rst7strings = [self.structure_to_rst7(qmstructure=node.structure) for node in chain]



            # print(rst7strings[0])
            energies, gradients, outputs = self._compute_enegrad(rst7strings)
            for node, energy, gradient in zip(chain, energies, gradients):
                node._cached_energy = energy
                node._cached_gradient = gradient
                # node._cached_result = enegrad_output[2]
                node._cached_result = None

            if not all([node._cached_gradient is not None for node in chain]):
                failed_results = []
                for node in chain:
                    if node._cached_result is not None and node._cached_gradient is None:
                        failed_results.append(node._cached_result)
                raise ElectronicStructureError(
                    msg="Gradient calculation failed.", obj=failed_results)
            grads = np.array([node.gradient for node in chain])

        return grads

    def structure_to_rst7(self, qmstructure):
        """
        takes a reference string of the rst7 file 'reference_string',
        indices of QM section of rst7 file, 'qmindices',
        indices of coordinates section of rst7 file, 'coordinate_indices',
        and a Structure object whose coordinates will be swapping in, 'qmstructure'
        """
        # reference_coords, _ = rst7_to_coords_and_indices(self.ref_rst7_react)

        # LMAO , removing alignment. Comment out next line if u want alignment
        # aligned_geom = (np.array(qmstructure.geometry)*(1/ANGSTROM_TO_BOHR))[qmindices]
        aligned_geom = (np.array(qmstructure.geometry)*(1/ANGSTROM_TO_BOHR))
        # reference_coords[qmindices] = aligned_geom
        reference_coords = aligned_geom

        arr = np.array([46.4274321, 82.5659739, 37.1465461, 9.8415857, 51.2254588, 44.5854156])

        # Determine the maximum width of the integer part
        max_int_width = 0
        for x in arr:
            int_part_len = len(str(int(abs(x))))  # Handle negative numbers
            if x < 0:
                int_part_len += 1 # Account for the negative sign
            if int_part_len > max_int_width:
                max_int_width = int_part_len

        # Define a custom formatter to align numbers by decimal point and specify precision
        def format_float_with_precision(x):
            return f"{x:{max_int_width + 8}.7f}"


        string = ""
        i=0
        for j, _ in enumerate(reference_coords):
            atom = reference_coords[j]
            if i==0:
                string+="  "+np.array2string(atom, separator='  ', prefix=' ', max_line_width=1e7, formatter={'float':format_float_with_precision})[1:-1]+" "
                i+=1
            elif i==1:
                string+=np.array2string(atom, separator='  ', prefix=' ', max_line_width=1e7, formatter={'float':format_float_with_precision})[1:-1]+"\n"
                i=0



        reference_string = self.ref_rst7_react

        headers = reference_string.split("\n")[:2]
        # headers+=velocities

        newfile = "\n".join(headers)
        newfile+='\n'
        newfile+=string
        # print(newfile)

        return newfile


def rst7prmtop_to_structure(rst7_str, prmtopdata_str):
    xyz_coords, coord_indices = rst7_to_coords_and_indices(rst7_str)
    symbols = np.array(parse_symbols_from_prmtop(prmtopdata_str))
    structure = Structure(geometry=xyz_coords*ANGSTROM_TO_BOHR, symbols=symbols)
    return structure
