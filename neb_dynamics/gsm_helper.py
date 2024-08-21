from pyGSM.coordinate_systems.delocalized_coordinates import DelocalizedInternalCoordinates
from pyGSM.coordinate_systems.primitive_internals import PrimitiveInternalCoordinates
from pyGSM.coordinate_systems.topology import Topology
from pyGSM.growing_string_methods import DE_GSM
from pyGSM.level_of_theories.ase import ASELoT
from pyGSM.optimizers.eigenvector_follow import eigenvector_follow
from pyGSM.optimizers.lbfgs import lbfgs
from pyGSM.potential_energy_surfaces import PES
from pyGSM.utilities import nifty
from pyGSM.utilities.elements import ElementData
from pyGSM.molecule.molecule import Molecule
from ase import Atoms
import os
from pathlib import Path
import shutil


def minimal_wrapper_de_gsm(
        atoms_reactant: Atoms,
        atoms_product: Atoms,
        calculator,
        fixed_reactant=True,
        fixed_product=True,

        num_nodes=11,  # 20 for SE-GSM
        add_node_tol=0.1,  # convergence for adding new nodes
        conv_tol=0.0005,  # Convergence tolerance for optimizing nodes
        conv_Ediff=100.,  # Energy difference convergence of optimization.
        conv_gmax=100.,  # Max grad rms threshold
        max_gsm_iterations=10,
        max_opt_steps=3,  # 20 for SE-GSM
):
    # optimizer
    optimizer_method = "eigenvector_follow"  # OR "lbfgs"
    line_search = 'NoLineSearch'  # OR: 'backtrack'
    only_climb = True
    # 'opt_print_level': args.opt_print_level,
    step_size_cap = 0.1  # DMAX in the other wrapper

    # molecule
    coordinate_type = "TRIC"

    lot = ASELoT.from_options(calculator,
                              geom=[[x.symbol, *x.position] for x in atoms_reactant])

    # PES
    pes_obj = PES.from_options(lot=lot, ad_idx=0, multiplicity=1)

    # Build the topology
    element_table = ElementData()
    elements = [element_table.from_symbol(sym) for sym in atoms_reactant.get_chemical_symbols()]

    topology_reactant = Topology.build_topology(
        xyz=atoms_reactant.get_positions(),
        atoms=elements
    )
    topology_product = Topology.build_topology(
        xyz=atoms_product.get_positions(),
        atoms=elements
    )

    # Union of bonds
    # debated if needed here or not
    for bond in topology_product.edges():
        if bond in topology_reactant.edges() or (bond[1], bond[0]) in topology_reactant.edges():
            continue
        print(" Adding bond {} to reactant topology".format(bond))
        if bond[0] > bond[1]:
            topology_reactant.add_edge(bond[0], bond[1])
        else:
            topology_reactant.add_edge(bond[1], bond[0])

    # primitive internal coordinates
    prim_reactant = PrimitiveInternalCoordinates.from_options(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
        topology=topology_reactant,
        connect=coordinate_type == "DLC",
        addtr=coordinate_type == "TRIC",
        addcart=coordinate_type == "HDLC",
    )

    prim_product = PrimitiveInternalCoordinates.from_options(
        xyz=atoms_product.get_positions(),
        atoms=elements,
        topology=topology_product,
        connect=coordinate_type == "DLC",
        addtr=coordinate_type == "TRIC",
        addcart=coordinate_type == "HDLC",
    )

    # add product coords to reactant coords
    prim_reactant.add_union_primitives(prim_product)

    # Delocalised internal coordinates
    deloc_coords_reactant = DelocalizedInternalCoordinates.from_options(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
        connect=coordinate_type == "DLC",
        addtr=coordinate_type == "TRIC",
        addcart=coordinate_type == "HDLC",
        primitives=prim_reactant
    )

    # Molecules
    from_hessian = optimizer_method == "eigenvector_follow"

    molecule_reactant = Molecule.from_options(
        geom=[[x.symbol, *x.position] for x in atoms_reactant],
        PES=pes_obj,
        coord_obj=deloc_coords_reactant,
        Form_Hessian=from_hessian
    )

    molecule_product = Molecule.copy_from_options(
        molecule_reactant,
        xyz=atoms_product.get_positions(),
        new_node_id=num_nodes - 1,
        copy_wavefunction=False
    )

    # optimizer
    opt_options = dict(print_level=0,
                       Linesearch=line_search,
                       update_hess_in_bg=not (only_climb or optimizer_method == "lbfgs"),
                       conv_Ediff=conv_Ediff,
                       conv_gmax=conv_gmax,
                       DMAX=step_size_cap,
                       opt_climb=only_climb)
    if optimizer_method == "eigenvector_follow":
        optimizer_object = eigenvector_follow.from_options(**opt_options)
    elif optimizer_method == "lbfgs":
        optimizer_object = lbfgs.from_options(**opt_options)
    else:
        raise NotImplementedError

    # GSM
    ID = 0
    gsm = DE_GSM.from_options(
        reactant=molecule_reactant,
        product=molecule_product,
        nnodes=num_nodes,
        CONV_TOL=conv_tol,
        CONV_gmax=conv_gmax,
        CONV_Ediff=conv_Ediff,
        ADD_NODE_TOL=add_node_tol,
        growth_direction=0,  # I am not sure how this works
        optimizer=optimizer_object,
        ID=ID,
        print_level=0,
        mp_cores=1,  # parallelism not tested yet with the ASE calculators
        interp_method="DLC",
    )

    # optimize reactant and product if needed
    if not fixed_reactant:
        nifty.printcool("REACTANT GEOMETRY NOT FIXED!!! OPTIMIZING")
        path = os.path.join(os.getcwd(), 'scratch', f"{ID:03}", "0")
        optimizer_object.optimize(
            molecule=molecule_reactant,
            refE=molecule_reactant.energy,
            opt_steps=10,
            path=path
        )
    if not fixed_product:
        nifty.printcool("PRODUCT GEOMETRY NOT FIXED!!! OPTIMIZING")
        path = os.path.join(os.getcwd(), 'scratch', f"{ID:03}", str(num_nodes - 1))
        optimizer_object.optimize(
            molecule=molecule_product,
            refE=molecule_product.energy,
            opt_steps=10,
            path=path
        )

    # set 'rtype' as in main one (???)
    if only_climb:
        rtype = 1
    else:
        rtype = 2

    # do GSM
    gsm.go_gsm(max_gsm_iterations, max_opt_steps, rtype=rtype)

    scr_path = Path(os.getcwd(), 'scratch')
    shutil.rmtree(scr_path)

    return gsm


def gsm_to_ase_atoms(gsm: DE_GSM):
    # string
    frames = []
    for energy, geom in zip(gsm.energies, gsm.geometries):
        at = Atoms(symbols=[x[0] for x in geom], positions=[x[1:4] for x in geom])
        at.info["energy"] = energy
        frames.append(at)

    # TS
    ts_geom = gsm.nodes[gsm.TSnode].geometry
    ts_atoms = Atoms(symbols=[x[0] for x in ts_geom], positions=[x[1:4] for x in ts_geom])

    return frames, ts_atoms
