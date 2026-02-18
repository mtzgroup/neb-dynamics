import numpy as np
from neb_dynamics.nodes.node import Node, XYNode
from neb_dynamics.qcio_structure_helpers import split_structure_into_frags
from neb_dynamics.geodesic_interpolation2.coord_utils import align_geom
# from neb_dynamics.molecule import Molecule
# from neb_dynamics.qcio_structure_helpers import molecule_to_structure
import traceback
from qcinf import structure_to_smiles
# from rxnmapper import RXNMapper

# Rich imports for flashy CLI output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _console = Console()
    _rich_available = True
except ImportError:
    _console = None
    _rich_available = False

# RDKit for ASCII molecule rendering
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    _rdkit_available = True
except ImportError:
    _rdkit_available = False


# Collector for comparison results to print once at the end
_comparison_results = []


def _reset_comparison_results():
    """Reset the comparison results collector."""
    global _comparison_results
    _comparison_results = []


def _add_comparison_result(result: dict):
    """Add a comparison result to the collector."""
    global _comparison_results
    _comparison_results.append(result)


def _print_all_comparisons():
    """Print all collected comparison results as a single consolidated report."""
    if not _comparison_results:
        return

    if not _rich_available:
        # Fall back to simple print
        for result in _comparison_results:
            print(f"\t{result.get('smi1', '')} || {result.get('smi2', '')}")
        return

    # Create a consolidated report table
    table = Table(title="Elementary Step Comparison Results", show_header=True)
    table.add_column("Comparison", style="cyan", width=15)
    table.add_column("Graph Isomorphic", justify="center", width=12)
    table.add_column("SMILES Match", justify="center", width=12)
    table.add_column("Structure 1 (ASCII)", width=30)
    table.add_column("Structure 2 (ASCII)", width=30)

    for i, result in enumerate(_comparison_results):
        graph_iso = "[green]✓[/green]" if result.get('graph_isomorphic', False) else "[red]✗[/red]"
        smiles_match = "[green]✓[/green]" if result.get('smiles_identical', False) else "[red]✗[/red]"

        # Get ASCII representations
        smi1 = result.get('smi1', '')
        smi2 = result.get('smi2', '')
        ascii1 = _render_molecule_ascii(smi1, width=28, height=6) if _rdkit_available else smi1[:25]
        ascii2 = _render_molecule_ascii(smi2, width=28, height=6) if _rdkit_available else smi2[:25]

        # Truncate for table cell
        ascii1_lines = ascii1.split('\n')[:6] if '\n' in ascii1 else [ascii1[:28]]
        ascii2_lines = ascii2.split('\n')[:6] if '\n' in ascii2 else [ascii2[:28]]

        # Pad to same height
        max_lines = max(len(ascii1_lines), len(ascii2_lines))
        ascii1_lines.extend([''] * (max_lines - len(ascii1_lines)))
        ascii2_lines.extend([''] * (max_lines - len(ascii2_lines)))

        # Combine lines for the cell
        ascii1_cell = '\n'.join(ascii1_lines)
        ascii2_cell = '\n'.join(ascii2_lines)

        table.add_row(f"#{i+1}", graph_iso, smiles_match, ascii1_cell, ascii2_cell)

    _console.print(table)


def _render_molecule_ascii(smiles: str, width: int = 60, height: int = 12) -> str:
    """
    Render a SMILES string as ASCII art using RDKit.
    Generates a 2D depiction and converts to ASCII.
    """
    if not _rdkit_available:
        return smiles

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)

        # Draw to PNG bytes and convert to ASCII approximation
        # RDKit doesn't have native ASCII, so we create a simple text representation
        # by parsing the bond information
        return _ascii_from_mol(mol, width, height)
    except Exception:
        return smiles


def _ascii_from_mol(mol, width: int = 50, height: int = 12) -> str:
    """
    Create an ASCII representation of a molecule using 2D coordinates.
    Projects atoms onto a 2D grid for a more visual representation.
    """
    try:
        from rdkit.Chem import AllChem

        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)

        # Get conformer
        conf = mol.GetConformer()
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

        # Get 2D coordinates
        coords_2d = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords_2d.append((pos.x, pos.y))

        if not coords_2d:
            return Chem.MolToSmiles(mol)

        # Normalize coordinates to fit in grid
        xs = [c[0] for c in coords_2d]
        ys = [c[1] for c in coords_2d]
        x_range = max(xs) - min(xs) if max(xs) != min(xs) else 1
        y_range = max(ys) - min(ys) if max(ys) != min(ys) else 1

        # Create grid
        grid_width = width - 4
        grid_height = height - 2

        grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]

        # Map atoms to grid positions
        atom_positions = []
        for i, (x, y) in enumerate(coords_2d):
            gx = int((x - min(xs)) / x_range * (grid_width - 1))
            gy = int((y - min(ys)) / y_range * (grid_height - 1))
            gy = grid_height - 1 - gy  # Flip y axis
            atom_positions.append((gx, gy, atoms[i]))

        # Draw bonds using Bresenham's line algorithm
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            x1, y1 = atom_positions[i][0], atom_positions[i][1]
            x2, y2 = atom_positions[j][0], atom_positions[j][1]

            # Different characters for different bond types
            bond_type = bond.GetBondType()
            bond_char = '-'
            if bond_type == 2:  # Double bond
                bond_char = '='
            elif bond_type == 3:  # Triple bond
                bond_char = '#'

            # Bresenham-like line drawing
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy

            x, y = x1, y1
            while True:
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    if grid[y][x] == ' ':
                        grid[y][x] = bond_char
                if x == x2 and y == y2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

        # Place atom symbols on top
        for gx, gy, symbol in atom_positions:
            if 0 <= gx < grid_width and 0 <= gy < grid_height:
                grid[gy][gx] = symbol

        # Build output
        lines = []
        lines.append("┌" + "─" * (grid_width) + "┐")
        for row in grid:
            lines.append("│" + "".join(row) + "│")
        lines.append("└" + "─" * (grid_width) + "┘")

        return "\n".join(lines)
    except Exception:
        return Chem.MolToSmiles(mol)


def _render_smiles_ascii(smiles: str, width: int = 40) -> str:
    """
    Render a SMILES string as simple ASCII art representation.
    Shows the structure in a compact form.
    """
    # Simple representation - just wrap the SMILES nicely
    if len(smiles) <= width:
        return smiles

    # Try to break at special characters for readability
    for sep in ['>>', '.', '(', ')', '[', ']', '=', '#', ':', '-', '+']:
        if sep in smiles:
            parts = smiles.split(sep)
            lines = []
            current = ""
            for part in parts:
                if len(current) + len(sep) + len(part) <= width:
                    current += sep + part if current else part
                else:
                    if current:
                        lines.append(current)
                    current = part
            if current:
                lines.append(current)
            if lines:
                return '\n'.join(lines)

    # Fallback: just wrap at width
    return '\n'.join([smiles[i:i+width] for i in range(0, len(smiles), width)])

# Rich imports for flashy CLI output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _console = Console()
    _rich_available = True
except ImportError:
    _console = None
    _rich_available = False


def _smiles_to_ascii(smiles: str) -> str:
    """Render SMILES as ASCII art using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        import io
        import base64

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        # Generate 2D coordinates
        Chem.Compute2DCoords(mol)

        # Draw to PNG bytes
        img = Draw.MolToImage(mol, size=(300, 150))
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()

        # Convert to base64 for display
        b64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"[image]data:image/png;base64,{b64}[/image]"
    except Exception:
        return smiles


def _render_smiles_comparison(smi1: str, smi2: str) -> str:
    """Render a side-by-side comparison of two SMILES as ASCII art."""
    if not _rich_available:
        return f"{smi1} || {smi2}"

    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        import io
        import base64

        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)

        if mol1 is None or mol2 is None:
            return f"[bold cyan]{smi1}[/bold cyan] [dim]||[/dim] [bold yellow]{smi2}[/bold yellow]"

        Chem.Compute2DCoords(mol1)
        Chem.Compute2DCoords(mol2)

        # Draw both molecules side by side
        img = Draw.MolsToGridImage([mol1, mol2], molsPerRow=2, subImgSize=(250, 200))

        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        b64 = base64.b64encode(img_bytes).decode('utf-8')

        return f"[image]data:image/png;base64,{b64}[/image]"
    except Exception:
        return f"[bold cyan]{smi1}[/bold cyan] [dim]||[/dim] [bold yellow]{smi2}[/bold yellow]"


def is_identical(
    self: Node,
    other: Node,
    *,
    global_rmsd_cutoff: float = 20.0,
    fragment_rmsd_cutoff: float = 1.0,
    kcal_mol_cutoff: float = 1.0,
    verbose: bool = True,
) -> bool:
    """
    computes whether two nodes are identical.
    if nodes have computable molecular graphs, will check that
    they are isomorphic to each other + that their distances are
    within a threshold (default: 1.0 Bohr) and their enegies
    are within a windos (default: 1.0 kcal/mol).
    otherwise, will only check distances.
    """
    if self.has_molecular_graph and other.has_molecular_graph:
        if verbose and _rich_available:
            _console.print(Panel.fit(
                "[bold cyan]Using graph and distance based comparison[/bold cyan]",
                border_style="cyan"
            ))
        elif verbose:
            print("elemsteps: Using graph and distance based comparison")
        conditions = [
            _is_connectivity_identical(self, other, verbose=verbose),
            _is_conformer_identical(
                self,
                other,
                global_rmsd_cutoff=global_rmsd_cutoff,
                fragment_rmsd_cutoff=fragment_rmsd_cutoff,
                kcal_mol_cutoff=kcal_mol_cutoff,
                verbose=verbose
            ),
        ]
    else:
        if verbose and _rich_available:
            _console.print(Panel.fit(
                "[bold yellow]Using distance based comparison, no graphs[/bold yellow]",
                border_style="yellow"
            ))
        elif verbose:
            print("elemsteps: Using distance based comparison, no graphs")
        conditions = [
            _is_conformer_identical(
                self,
                other,
                global_rmsd_cutoff=global_rmsd_cutoff,
                fragment_rmsd_cutoff=fragment_rmsd_cutoff,
                kcal_mol_cutoff=kcal_mol_cutoff,
                verbose=verbose
            )
        ]

    return all(conditions)


def _is_conformer_identical(
    self: Node,
    other: Node,
    *,
    global_rmsd_cutoff: float = 20.0,
    fragment_rmsd_cutoff: float = 0.5,
    kcal_mol_cutoff: float = 1.0,
    verbose: bool = True,
) -> bool:

    if verbose:
        if _rich_available:
            _console.print(Panel.fit(
                "[bold]Verifying if two geometries are identical[/bold]",
                border_style="blue"
            ))
        else:
            print("\n\tVerifying if two geometries are identical.")
    if isinstance(self, XYNode):
        global_dist = np.linalg.norm(other.coords - self.coords)
    else:
        global_dist, aligned_geometry = align_geom(
            refgeom=other.coords, geom=self.coords
        )

    per_frag_dists = []
    if self.has_molecular_graph:
        if not _is_connectivity_identical(self, other, verbose=verbose):
            if verbose:
                if _rich_available:
                    _console.print(Panel.fit(
                        "[bold red]Graphs differed. Not identical.[/bold red]",
                        border_style="red"
                    ))
                else:
                    print("\t\tGraphs differed. Not identical.")
            return False
        if verbose:
            if _rich_available:
                _console.print(Panel.fit(
                    "[bold green]Graphs identical. Checking distances...[/bold green]",
                    border_style="green"
                ))
            else:
                print("\t\tGraphs identical. Checking distances...")
        self_frags = split_structure_into_frags(self.structure)
        other_frags = split_structure_into_frags(other.structure)
        if len(self_frags) != len(other_frags):
            if verbose:
                if _rich_available:
                    _console.print(Panel.fit(
                        "[bold red]Fragments differed in number. Not identical.[/bold red]",
                        border_style="red"
                    ))
                else:
                    print("\t\tFragments differed in number. Not identical.")
            return False

        info_self = [(i, len(structure.geometry), structure.symbols)
                     for i, structure in enumerate(self_frags)]  # index, sys_size, symbols

        info_other = [(i, len(structure.geometry), structure.symbols)
                      for i, structure in enumerate(other_frags)]

        sorted_info_self = sorted(info_self, key=lambda x: (x[1], x[2]))
        sorted_info_other = sorted(info_other, key=lambda x: (x[1], x[2]))

        inds_self = [val[0] for val in sorted_info_self]
        inds_other = [val[0] for val in sorted_info_other]

        for i_self, i_other in zip(inds_self, inds_other):
            frag_self = self_frags[i_self]
            frag_other = other_frags[i_other]

            frag_dist, _ = align_geom(
                refgeom=frag_self.geometry, geom=frag_other.geometry
            )
            per_frag_dists.append(frag_dist)
            if verbose:
                if _rich_available:
                    _console.print(f"[dim]Fragment distance: {frag_dist:.4f}[/dim]")
                else:
                    print(f"\t\t\tfrag dist: {frag_dist}")
    else:
        per_frag_dists.append(global_dist)

    en_delta = np.abs((self.energy - other.energy) * 627.5)

    global_rmsd_identical = global_dist <= global_rmsd_cutoff
    fragment_rmsd_identical = max(per_frag_dists) <= fragment_rmsd_cutoff
    rmsd_identical = global_rmsd_identical and fragment_rmsd_identical
    energies_identical = en_delta < kcal_mol_cutoff

    if verbose:
        if _rich_available:
            # Show convergence results in a table
            results_table = Table(title="Geometry Comparison Results", show_header=False)
            results_table.add_column("Criterion", style="cyan")
            results_table.add_column("Value", style="white")
            results_table.add_column("Status", justify="center")

            rmsd_val = f"RMSD: {global_dist:.4f} (cutoff: {global_rmsd_cutoff})"
            rmsd_status = "[green]✓[/green]" if global_rmsd_identical else "[red]✗[/red]"
            results_table.add_row("RMSD", rmsd_val, rmsd_status)

            frag_rmsd_val = f"Max fragment RMSD: {max(per_frag_dists):.4f} (cutoff: {fragment_rmsd_cutoff})"
            frag_rmsd_status = "[green]✓[/green]" if fragment_rmsd_identical else "[red]✗[/red]"
            results_table.add_row("Fragment RMSD", frag_rmsd_val, frag_rmsd_status)

            en_val = f"ΔE: {en_delta:.4f} kcal/mol (cutoff: {kcal_mol_cutoff})"
            en_status = "[green]✓[/green]" if energies_identical else "[red]✗[/red]"
            results_table.add_row("Energy", en_val, en_status)

            _console.print(results_table)
        else:
            print(f"\t\t{rmsd_identical=} {energies_identical=} {en_delta=}")
    if rmsd_identical and energies_identical:
        return True
    else:
        return False


def _is_connectivity_identical(self: Node, other: Node, verbose: bool = True) -> bool:
    """
    checks graphs of both nodes and returns whether they are isomorphic
    to each other.
    """
    # print("different graphs")
    connectivity_identical = self.graph.remove_Hs().is_bond_isomorphic_to(
        other.graph.remove_Hs()
    )
    natom = len(self.coords)
    smi1 = ""
    smi2 = ""
    stereochem_identical = True

    if natom < 100:  # arbitrary number, else this takes too long
        try:

            smi1 = structure_to_smiles(self.structure)
            smi2 = structure_to_smiles(other.structure)
            stereochem_identical = smi1 == smi2

        except Exception:
            if verbose:
                print("Constructing smiles failed. Pretending this check succeeded.")
            stereochem_identical = True
    else:
        print("System too large. Not checking stereochemistry.")
        stereochem_identical = True

    # Always collect results for final consolidated report
    if smi1 and smi2:
        _add_comparison_result({
            'smi1': smi1,
            'smi2': smi2,
            'graph_isomorphic': connectivity_identical,
            'smiles_identical': stereochem_identical,
        })

    if verbose:
        # Create a table for the comparison results
        if _rich_available:
            table = Table(title="Molecular Graph Comparison")
            table.add_column("Property", style="cyan", justify="left")
            table.add_column("Status", justify="left")

            # Graphs isomorphic
            graph_status = "[green]✓ Yes[/green]" if connectivity_identical else "[red]✗ No[/red]"
            table.add_row("Graphs isomorphic", graph_status)

            # Stereochemical smiles identical
            smi_status = "[green]✓ Yes[/green]" if stereochem_identical else "[red]✗ No[/red]"
            table.add_row("Stereochemical SMILES identical", smi_status)

            _console.print(table)

            # If not identical, show the SMILES side by side
            if not stereochem_identical:
                smiles_table = Table(title="SMILES Comparison", show_header=True)
                smiles_table.add_column("Structure 1", style="yellow", width=50)
                smiles_table.add_column("Structure 2", style="yellow", width=50)

                # Render SMILES in ASCII art style (wrapped for readability)
                smi1_rendered = _render_smiles_ascii(smi1, 45)
                smi2_rendered = _render_smiles_ascii(smi2, 45)

                smiles_table.add_row(smi1_rendered, smi2_rendered)
                _console.print(smiles_table)
        else:
            print(f"\t\tGraphs isomorphic to each other: {connectivity_identical}")
            print(f"\t\tStereochemical smiles identical: {stereochem_identical}")
            if not stereochem_identical:
                print(f"\t{smi1} || {smi2}")

    return connectivity_identical and stereochem_identical


def update_node_cache(node_list, results):
    """
    inplace update of cached results
    """
    for node, result in zip(node_list, results):
        node._cached_result = result
        if result is not None:
            if result.success:
                node._cached_energy = result.results.energy
                node._cached_gradient = result.results.gradient
        else:
            node._cached_energy = None
            node._cached_gradient = None


def create_pairs_from_smiles(smi1: str, smi2: str, spinmult=1):
    raise NotImplementedError(
        "Latest RXNMapper update has made this feature incompatible. Need to fix compatibility.")
    # rxnsmi = f"{smi1}>>{smi2}"
    # rxn_mapper = RXNMapper()
    # rxn = [rxnsmi]
    # result = rxn_mapper.get_attention_guided_atom_maps(rxn)[0]
    # mapped_smi = result["mapped_rxn"]
    # r_smi, p_smi = mapped_smi.split(">>")
    # print(r_smi, p_smi)
    # r = Molecule.from_mapped_smiles(r_smi)
    # p = Molecule.from_mapped_smiles(p_smi)

    # td_r, td_p = (
    #     molecule_to_structure(r, charge=r.charge, spinmult=spinmult),
    #     molecule_to_structure(p, charge=p.charge, spinmult=spinmult),
    # )
    # return td_r, td_p


def displace_by_dr(node: Node, displacement: np.array, dr: float = 0.1) -> Node:
    """returns a new node object that has been displaced along the input 'displacement' vector by 'dr'.

    Args:
        node (Node): Node to displace
        displacement (np.array): vector along which to displace
        dr (float, optional): Magnitude of displacement vector. Defaults to 0.1.
    """
    displacement = displacement / np.linalg.norm(displacement)
    new_coords = node.coords + dr*displacement
    return node.update_coords(new_coords)
