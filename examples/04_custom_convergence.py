"""
Tutorial 4: Custom Convergence Criteria
This demonstrates different convergence settings for NEB calculations.
"""
from qcio import Structure
from neb_dynamics import StructureNode, NEBInputs, ChainInputs
from neb_dynamics.engines.qcop import QCOPEngine
from neb_dynamics.neb import NEB
import neb_dynamics.chainhelpers as ch
from neb_dynamics import Chain
from neb_dynamics.optimizers.cg import ConjugateGradient

# Tight convergence for high-accuracy calculations
nbi_tight = NEBInputs(
    en_thre=1e-5,           # Tighter energy threshold
    rms_grad_thre=0.002,     # Tighter gradient threshold
    max_steps=1000,          # More steps allowed
    v=True
)

# Loose convergence for quick screening
nbi_loose = NEBInputs(
    en_thre=1e-3,
    rms_grad_thre=0.1,
    max_steps=100,
    v=True
)

print("Tutorial 4: Custom Convergence Criteria")
print(f"Tight convergence settings: en_thre={nbi_tight.en_thre}, rms_grad_thre={nbi_tight.rms_grad_thre}")
print(f"Loose convergence settings: en_thre={nbi_loose.en_thre}, rms_grad_thre={nbi_loose.rms_grad_thre}")
print("\nUse these NEBInputs with your NEB calculation:")
print("  n = NEB(initial_chain=initial_chain, parameters=nbi_tight, optimizer=opt, engine=eng)")
