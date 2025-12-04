from neb_dynamics.engines.engine import Engine
from neb_dynamics.errors import GradientsNotComputedError, ElectronicStructureError
from dataclasses import dataclass
from neb_dynamics.helper_functions import rst7_to_coords_and_indices
from neb_dynamics.geodesic_interpolation2.coord_utils import align_geom
from pathlib import Path
import numpy as np
from qcio import FileInput
from qccodec.parsers.terachem import parse_energy, parse_gradient
from qcop import compute
from neb_dynamics import Chain

@dataclass
class QMMEngine(Engine):

    tcin_fp: Path = Path("/home/nancyzmn/covalent_inhibitor/prepare_protein/3-QCOP-SP/tc.in")
    qminds_fp: Path = Path("/home/nancyzmn/covalent_inhibitor/prepare_protein/3-QCOP-SP/qmindices.dat")
    prmtop_fp: Path = Path("/home/nancyzmn/covalent_inhibitor/prepare_protein/3-QCOP-SP/5p9i_sphere_nobox.prmtop")
    rst7_fp_prod: Path = Path("/home/nancyzmn/covalent_inhibitor/prepare_protein/3-QCOP-SP/optim_product.rst7")
    rst7_fp_react: Path = Path("/home/nancyzmn/covalent_inhibitor/prepare_protein/3-QCOP-SP/optim_reactant.rst7")



    def __post_init__(self):
        self.inp_file = self.tcin_fp.read_text()  # Or your own function to create tc.in
        self.qmindices = self.qminds_fp.read_text()
        self.prmtop = self.prmtop_fp.read_text()
        self.ref_rst7_react = self.rst7_fp_react.read_text()
        self.ref_rst7_prod = self.rst7_fp_prod.read_text()
        _, indices_coordinates = rst7_to_coords_and_indices(self.ref_rst7_react)
        self.indices_coordinates = indices_coordinates


    def _compute_enegrad(self, rst7_string):
        # f.close()
        # for i, (line1, line2) in enumerate(zip(rst7_string.split("\n"), self.ref_rst7.split("\n"))):
        #     same = line1.replace(" ", "")==line2.replace(" ", "")
        #     if not same:
        #         print(i, '\n',line1,'\n',line2)
        ## Input files for QC Program

        # Create a FileInput object for TeraChem
        file_inp = FileInput(
            files={
                "tc.in": self.inp_file,
                "mycoolrst7.rst7": rst7_string,
                "qmindices.dat": self.qmindices,
                "5p9i_sphere_nobox.prmtop": self.prmtop,
            },
            cmdline_args=["tc.in"],
        )

        # This will write the files to disk in a temporary directory and then run
        # "terachem tc.in" in that directory.
        # return file_inp
        output = compute("terachem", file_inp, print_stdout=False)

        # Save Output
        # output.save("output.json")  # Only if you want to save the output

        gradient = parse_gradient(output.stdout) # Parses QM part of the gradient
        gradient = gradient[:-3] #Get rid of linked-atoms
        energy = parse_energy(output.stdout) # Parses FINAL ENERGY: line
        return energy, gradient, output

    def compute_energies(self, chain: Chain):
        self.compute_gradients(chain)
        enes = np.array([node.energy for node in chain])

        return enes

    def compute_gradients(self, chain):
        try:
            grads = np.array([node.gradient for node in chain])

        except GradientsNotComputedError:
            qminds = [int(v) for v in self.qmindices.strip().split("\n")]

            rst7strings = [self.structure_to_rst7(qmstructure=node.structure,
                                             qmindices=qminds,
                                             coordinate_indices=self.indices_coordinates) for node in chain]




            node_list_enegrads = [self._compute_enegrad(string) for string in rst7strings]
            for node, enegrad_output in zip(chain, node_list_enegrads):
                node._cached_energy = enegrad_output[0]
                node._cached_gradient = enegrad_output[1]
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

    def structure_to_rst7(self, qmstructure, qmindices, coordinate_indices):
        """
        takes a reference string of the rst7 file 'reference_string',
        indices of QM section of rst7 file, 'qmindices',
        indices of coordinates section of rst7 file, 'coordinate_indices',
        and a Structure object whose coordinates will be swapping in, 'qmstructure'
        """
        reference_coords_react, _ = rst7_to_coords_and_indices(self.ref_rst7_react)
        reference_coords_prod, _ = rst7_to_coords_and_indices(self.ref_rst7_prod)

        reference_qminds_react = reference_coords_react[qmindices]
        reference_qminds_prod = reference_coords_prod[qmindices]

        rmsd_react, aligned_geom_react = align_geom(refgeom=reference_qminds_react, geom=qmstructure.geometry_angstrom)
        rmsd_prod, aligned_geom_prod = align_geom(refgeom=reference_qminds_prod, geom=qmstructure.geometry_angstrom)

        if rmsd_react < rmsd_prod:
            aligned_geom = aligned_geom_react
            reference_coords = reference_coords_react
            reference_string = self.ref_rst7_react

        else:
            aligned_geom = aligned_geom_prod
            reference_coords = reference_coords_prod
            reference_string = self.ref_rst7_prod

        # LMAO , removing alignment. Comment out next line if u want alignment
        aligned_geom = qmstructure.geometry_angstrom
        reference_coords[qmindices] = aligned_geom
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

        reshaped_coords = reference_coords.reshape(-1, 6)
        # reshaped_coords_str = ["  "+np.array2string(l, separator='  ', prefix=' ', max_line_width=1e7, formatter={'float_kind':lambda x: "%.7f" % x})[1:-1] for l in reshaped_coords]
        reshaped_coords_str = ["  "+np.array2string(l, separator='  ', prefix=' ', max_line_width=1e7, formatter={'float':format_float_with_precision})[1:-1] for l in reshaped_coords]
        # print(len(reshaped_coords_str))

        # velocities = ['  0.0000000   0.0000000   0.0000000   0.0000000   0.0000000   0.0000000']*len(reshaped_coords)
        # velocities = "\n".join(velocities)

        headers = reference_string.split("\n")[:2]
        headers+=reshaped_coords_str
        # headers+=velocities

        newfile = "\n".join(headers)
        return newfile