import os
import subprocess
from dataclasses import dataclass

import numpy as np

from neb_dynamics.tdstructure import TDStructure


@dataclass
class Solvator:
    sphere_rad: float = 1.0
    tol: float = 2.0
    n_solvent: int = 10
    solvent: TDStructure = TDStructure.from_smiles("O")

    def solvate_td(self, td):
        td_list = td.split_td_into_frags()
        solv_td_list = [self.solvate_single_td(td) for td in td_list]
        adj_solv_td_list = self.adjust_td_list_coords(solv_td_list, offset=3)
        solvated_td = self.join_td_list(adj_solv_td_list)
        solvated_td.update_tc_parameters(td)

        return solvated_td

    def solvate_single_td(self, td):

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

        td.to_xyz("/tmp/solute.xyz")

        self.solvent.to_xyz("/tmp/solvent.xyz")

        subprocess.run(
            "/home/henryw7/test/nanoreactor/packmol/packmol-16.343/packmol < /tmp/inp_file.inp", shell=True,
            stdout=subprocess.DEVNULL
        )

        solv = TDStructure.from_xyz(
            "/tmp/solvated.xyz", tot_charge=td.charge, tot_spinmult=td.spinmult
        )

        solv.update_tc_parameters(td_ref=td)

        solv.mm_optimization("mmff94")

        os.remove("/tmp/solute.xyz")
        os.remove("/tmp/solvated.xyz")
        os.remove("/tmp/inp_file.inp")

        return solv

    def adjust_coords_td2_ref_td1(self, first_td, second_td, offset=5):

        max_x = first_td.coords.argmax(axis=0)[0]
        min_x = second_td.coords.argmin(axis=0)[0]

        new_coords2 = (
            second_td.coords - second_td.coords[min_x]
        )  # make the leftmost (x-direction) the zero of the molecule
        new_coords2 += first_td.coords[
            max_x
        ]  # there should now be a direct overlap between the leftmost of td2 and the rightmost of td1
        new_coords2 += [offset, 0, 0]  # add an offset

        new_td2 = TDStructure.from_coords_symbols(
            coords=new_coords2,
            symbols=second_td.symbols,
            tot_charge=second_td.charge,
            tot_spinmult=second_td.spinmult,
        )
        new_td2.update_tc_parameters(first_td)

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

    def join_td_list(self, list_of_td):
        all_symbs = []
        all_coords_list = []
        tot_charge = 0
        for td in list_of_td:
            all_symbs.extend(td.symbols)
            all_coords_list.append(td.coords)
            tot_charge += td.charge

        all_coords = np.concatenate(all_coords_list, axis=0)
        out = TDStructure.from_coords_symbols(
            coords=all_coords,
            symbols=all_symbs,
            tot_charge=tot_charge,
            tot_spinmult=list_of_td[0].spinmult,
        )
        out.update_tc_parameters(list_of_td[0])

        return out
