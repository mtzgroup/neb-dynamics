import os
import subprocess
from dataclasses import dataclass
from typing import List

import numpy as np
from qcio import Structure
from neb_dynamics.nodes.nodehelpers import split_structure_into_frags


@dataclass
class Solvator:
    sphere_rad: float = 1.0
    tol: float = 2.0
    n_solvent: int = 10
    solvent: Structure = Structure.from_smiles("O")

    def solvate_td(self, structure: Structure):
        td_list = split_structure_into_frags(structure=structure)
        solv_td_list = [self.solvate_single_structure(td) for td in td_list]
        adj_solv_td_list = self.adjust_td_list_coords(solv_td_list, offset=3)
        solvated_td = self.join_td_list(adj_solv_td_list)

        return solvated_td

    def solvate_single_structure(self, structure: Structure):

        inp_text = f"""
        tolerance {self.tol}
        filetype xyz
        output /tmp/solvated.xyz
        discale 1.5

        structure /tmp/solute.xyz
          number 1
          center
          fixed 0. 0. 0. 0. 0. 0.
        end structure

        structure /tmp/solvent.xyz
          number    {self.n_solvent}
          inside sphere 0.0 0.0 0.0 {self.sphere_rad}
        end structure"""

        f = open("/tmp/inp_file.inp", "w+")
        f.write(inp_text)
        f.close()

        structure.save("/tmp/solute.xyz")

        self.solvent.save("/tmp/solvent.xyz")

        subprocess.run(
            "/home/henryw7/test/nanoreactor/packmol/packmol-16.343/packmol < /tmp/inp_file.inp",
            shell=True, stdout=subprocess.DEVNULL
        )

        solv = Structure.open(
            "/tmp/solvated.xyz", charge=structure.charge,
            multiplicity=structure.multiplicity)

        os.remove("/tmp/solute.xyz")
        os.remove("/tmp/solvated.xyz")
        os.remove("/tmp/inp_file.inp")

        return solv

    def adjust_coords_td2_ref_td1(self, first_td: Structure, second_td: Structure, offset=5):

        max_x = first_td.geometry.argmax(axis=0)[0]
        min_x = second_td.geometry.argmin(axis=0)[0]

        new_coords2 = (
            second_td.geometry - second_td.geometry[min_x]
        )  # make the leftmost (x-direction) the zero of the molecule
        new_coords2 += first_td.geometry[
            max_x
        ]  # there should now be a direct overlap between the leftmost of td2 and the rightmost of td1
        new_coords2 += [offset, 0, 0]  # add an offset

        new_td2 = Structure(
            geometry=new_coords2,
            symbols=second_td.symbols,
            charge=second_td.charge,
            multiplicipy=second_td.multiplicity,
        )

        return new_td2

    def adjust_td_list_coords(self, td_list, offset=5):
        new_list = []
        for i, td in enumerate(td_list):
            if i == 0:
                new_list.append(td)
            else:
                new_td = self.adjust_coords_td2_ref_td1(
                    td_list[i - 1], td_list[i], offset=offset
                )
                new_list.append(new_td)

        return new_list

    def join_td_list(self, list_of_td: List[Structure]):
        all_symbs = []
        all_coords_list = []
        tot_charge = 0
        for td in list_of_td:
            all_symbs.extend(td.symbols)
            all_coords_list.append(td.coords)
            tot_charge += td.charge

        all_coords = np.concatenate(all_coords_list, axis=0)
        out = Structure(
            geometry=all_coords,
            symbols=all_symbs,
            charge=tot_charge,
            multiplicity=list_of_td[0].multiplicity,
        )

        return out
