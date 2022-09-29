from retropaths.abinitio.tdstructure import TDStructure
from retropaths.molecules.smiles_tools import graph_to_smiles
from retropaths.molecules.smiles_tools import __ATOM_LIST__ as AtomList
from neb_dynamics.NEB import Node3D

smi = "C1CC2CCC1=C2Cl"
# smi = 'CC(CCC(=O)N)CN'
# smi = "C=1CCCCC1"
td = TDStructure.from_smiles(smi)
mol = td.molecule_rp
mol.draw()

no = Node3D(td)
no.energy
# no.opt_func()
no.gradient

remapped_smi = mol.smiles
remapped_smi

# +
smi_ind_bridgeheads = []

count = 0
while count < len(remapped_smi):

    double_char = remapped_smi[count:count+2]
    char = remapped_smi[count]
    print(count, double_char, char)
    
    if double_char.lower() in AtomList:
        
        count+=2

    else:
        try: 
            int(char)
            smi_ind_bridgeheads.append(i-1)
            count+=1
        except:
            count += 1
    
smi_ind_bridgeheads
# -

for node in mol.nodes:
    if mol.nodes[node]['element'] != "H":
        print(f"{node}, {mol.nodes[node]}")

mol.smiles

graph_to_smiles(mol)


for e in mol.edges:

    s = mol.edges[e]["bond_order"]
    print(s)
    if s == "single":
        i = 1
    elif s == "double":
        i = 2
    elif s == "aromatic":
        i = 1.5
    elif s == "triple":
        i = 3
    mol.edges[e]["order"] = i

    smi = write_smiles(mol)
    oemol = oechem.OEGraphMol()
    oechem.OESmilesToMol(oemol, smi)
    smi2 = oechem.OEMolToSmiles(oemol)



