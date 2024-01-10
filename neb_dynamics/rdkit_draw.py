from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


def moldrawsvg(smiles, mapping, molSize=(300, 300), kekulize=True, fixed_bond_length=None, fixedScale=None, fontSize=None, lineWidth=None):
    """
    returns a chemdraw style svg image of mol (an rdkit molecule object)
    """
    mol = Chem.MolFromSmiles(smiles)
    try:
        mc = Chem.Mol(mol.ToBinary())
    except AttributeError:
        return f"I cannot draw this smiles: {smiles}"

    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)

    for i, atom in enumerate(mc.GetAtoms()):
        # For each atom, set the property "atomLabel" to a custom value, let's say a1, a2, a3,...
        sym = atom.GetSymbol()
        if sym in mapping:
            atom.SetProp("atomLabel", mapping[sym])

    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    if fixed_bond_length:
        drawer.drawOptions().fixedBondLength = fixed_bond_length
    if fixedScale:
        drawer.drawOptions().fixedScale = fixedScale
    if fontSize:
        # drawer.SetFontSize(fontSize)
        # drawer.drawOptions().fixedFontSize = fontSize
        drawer.drawOptions().minFontSize = -1
    if lineWidth:
        drawer.drawOptions().bondLineWidth = lineWidth
    drawer.clearBackground = False
    drawer.scaleBondWidth = True
    drawer.centreMoleculesBeforeDrawing = True
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return svg.replace("svg:", "").replace("", "")
