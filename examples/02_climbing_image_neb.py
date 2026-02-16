"""
Tutorial 2: Climbing Image NEB
This builds on Tutorial 1 - uses the same setup but with climbing image enabled.
Climbing image NEB finds the exact saddle point.
"""
from qcio import Structure
from neb_dynamics import StructureNode, NEBInputs, ChainInputs
from neb_dynamics.engines.qcop import QCOPEngine
from neb_dynamics.neb import NEB
import neb_dynamics.chainhelpers as ch
from neb_dynamics import Chain
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer

# Define structures using embedded XYZ data
start_xyz = """17
Frame 0
 C          0.885440409184        2.102076768875        0.627821743488
 C          0.841612815857        1.115651130676       -0.247040778399
 C         -0.420807182789        0.556873917580       -0.838685691357
 O         -0.430870711803        0.798001468182       -2.239025592804
 C         -0.522329509258       -0.964021146297       -0.644988238811
 C         -0.715307652950       -1.349323630333        0.785972416401
 C          0.097422197461       -2.135773420334        1.466876506805
 H         -0.267592728138        1.735007524490       -2.391090631485
 H         -1.386946558952       -1.296320080757       -1.227010726929
 H          0.993258416653       -2.552916049957        1.031124353409
 H         -0.008914750069        2.554719924927        1.029638171196
 H          1.820721507072        2.502734899521        0.982666194439
 H          1.748645901680        0.692936420441       -0.660514891148
 H         -1.296261906624        1.040292620659       -0.372635990381
 H          0.376222431660       -1.434682250023       -1.046772599220
 H         -1.611668467522       -0.956238925457        1.253052473068
 H         -0.102624341846       -2.409019231796        2.490613460541
"""

end_xyz = """17
Frame 29
 C          0.576232671738        0.921668708324        0.825146675110
 C          0.811691701412        1.315650582314       -0.631796598434
 C         -0.290672153234        0.930290937424       -1.571661710739
 O         -0.139186233282        0.584553778172       -2.709024667740
 C         -1.166546344757       -1.861497998238       -0.220005810261
 C         -0.774998486042       -1.183089017868        0.841198801994
 C          0.589584112167       -0.601971745491        1.032947182655
 H          0.880913138390        2.407624959946       -0.707833826542
 H         -2.171884059906       -2.235154390335       -0.311172336340
 H          1.294490098953       -1.047748923302        0.330671131611
 H         -0.374946832657        1.333065629005        1.164748311043
 H          1.358177542686        1.381788730621        1.429385662079
 H          1.752718806267        0.898995816708       -0.990877032280
 H         -1.302031993866        1.039528012276       -1.137167930603
 H         -0.504405736923       -2.070925474167       -1.043461441994
 H         -1.469262003899       -0.993297398090        1.650624632835
 H          0.930125653744       -0.819482147694        2.048279047012
"""

# Load structures from XYZ strings
start = Structure.from_xyz(start_xyz)
end = Structure.from_xyz(end_xyz)

# Create nodes
start_node = StructureNode(structure=start)
end_node = StructureNode(structure=end)

# Set up engine using ChemCloud
eng = QCOPEngine(compute_program="chemcloud")

# Optimize geometries
start_opt = eng.compute_geometry_optimization(start_node)
end_opt = eng.compute_geometry_optimization(end_node)

start_node = start_opt[-1]
end_node = end_opt[-1]

# Set up chain inputs
cni = ChainInputs(k=0.1, delta_k=0.09)

# Create a Chain object
chain = Chain.model_validate({
    'nodes': [start_node, end_node],
    'parameters': cni
})

# Generate initial path using geodesic interpolation
initial_chain = ch.run_geodesic(chain, nimages=15)

# Enable climbing image NEB - this is Tutorial 2
nbi = NEBInputs(climb=True, v=True)

# Use Velocity Projected Optimizer (recommended for climbing image)
opt = VelocityProjectedOptimizer(timestep=0.5)

# Create NEB object
n = NEB(
    initial_chain=initial_chain,
    parameters=nbi,
    optimizer=opt,
    engine=eng
)

# Run optimization
results = n.optimize_chain()

print("Climbing Image NEB completed successfully!")
print(f"Number of images in final chain: {len(n.optimized.nodes)}")
