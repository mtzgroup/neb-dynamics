#!/usr/bin/env python3
"""
Optimize NEB convergence parameters for fast convergence while ensuring
the TS guess can be successfully optimized to a true TS (1 negative frequency).

Usage:
    python -m neb_dynamics.scripts.optimize_convergence --reactants reactants.xyz --products products.xyz

This script will:
1. Run NEB with various convergence parameter combinations
2. For each, attempt TS optimization on the TS guess node
3. Verify the result has exactly 1 negative frequency
4. Log all results and recommend optimal parameters
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from neb_dynamics.chain import Chain
from neb_dynamics import ChainInputs, GIInputs, NEBInputs, StructureNode
from neb_dynamics.engines.qcop import QCOPEngine
from neb_dynamics.errors import NoneConvergedException
import neb_dynamics.chainhelpers as ch
from neb_dynamics.neb import NEB
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer
from neb_dynamics.qcio_structure_helpers import read_multiple_structure_from_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of a single parameter set test."""
    params: dict
    neb_steps: int = 0
    ts_converged: bool = False
    n_neg_freqs: int = -1
    success: bool = False
    error_msg: str = ""
    runtime_seconds: float = 0.0

    def __str__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"[{status}] NEB steps: {self.neb_steps} | "
            f"TS converged: {self.ts_converged} | "
            f"Neg freqs: {self.n_neg_freqs} | "
            f"Time: {self.runtime_seconds:.1f}s"
        )


def count_negative_frequencies(hessian: np.ndarray, tol: float = -0.01) -> int:
    """Count negative eigenvalues in Hessian matrix.

    Args:
        hessian: Hessian matrix (n_atoms*3 x n_atoms*3)
        tol: Threshold for considering an eigenvalue negative

    Returns:
        Number of negative eigenvalues
    """
    eigenvalues = np.linalg.eigvalsh(hessian)
    n_neg = np.sum(eigenvalues < tol)
    return n_neg


def load_structures(reactants_file: Path, products_file: Path) -> tuple:
    """Load reactants and products from XYZ files."""
    from neb_dynamics.qcio_structure_helpers import read_multiple_structure_from_file

    reactants_list = read_multiple_structure_from_file(reactants_file)
    products_list = read_multiple_structure_from_file(products_file)

    if len(reactants_list) == 0:
        raise ValueError(f"No structures found in {reactants_file}")
    if len(products_list) == 0:
        raise ValueError(f"No structures found in {products_file}")

    reactants = reactants_list[0]
    products = products_list[0]

    logger.info(f"Loaded reactants: {len(reactants.atoms)} atoms")
    logger.info(f"Loaded products: {len(products.atoms)} atoms")

    return reactants, products


def create_initial_chain(reactants, products, nimages: int = 15) -> Chain:
    """Create initial chain using geodesic interpolation."""
    # Create nodes from reactant and product
    node0 = StructureNode(structure=reactants)
    node1 = StructureNode(structure=products)

    chain_inputs = ChainInputs(
        k=0.1,
        delta_k=0.09,
        use_geodesic_interpolation=True,
        node_freezing=True,
    )

    # Create chain with just endpoints
    chain = Chain.model_validate({
        'nodes': [node0, node1],
        'parameters': chain_inputs,
    })

    # Generate initial path using geodesic interpolation
    initial_chain = ch.run_geodesic(chain, nimages=nimages)

    return initial_chain


def run_neb_and_verify(
    reactants_file: Path,
    products_file: Path,
    engine: QCOPEngine,
    neb_params: NEBInputs,
    max_chain_images: int = 15,
    v: bool = False,
) -> tuple[Optional[Chain], Optional[dict], str]:
    """Run NEB and verify the TS guess can be optimized to a true TS.

    Returns:
        tuple: (optimized_chain, ts_result_dict, error_message)
               ts_result_dict contains: {converged, n_neg_freqs, ts_node}
    """
    # Load structures
    reactants, products = load_structures(reactants_file, products_file)

    logger.info(f"Running NEB with parameters:")
    logger.info(f"  rms_grad_thre: {neb_params.rms_grad_thre}")
    logger.info(f"  max_rms_grad_thre: {neb_params.max_rms_grad_thre}")
    logger.info(f"  ts_grad_thre: {neb_params.ts_grad_thre}")
    logger.info(f"  ts_spring_thre: {neb_params.ts_spring_thre}")
    logger.info(f"  max_steps: {neb_params.max_steps}")
    logger.info(f"  barrier_thre: {neb_params.barrier_thre}")

    # Interpolate initial chain
    chain = create_initial_chain(reactants, products, nimages=max_chain_images)

    # Set up optimizer and NEB
    optimizer = VelocityProjectedOptimizer(timestep=0.5)
    neb = NEB(
        initial_chain=chain,
        optimizer=optimizer,
        parameters=neb_params,
        engine=engine,
    )

    try:
        elem_results = neb.optimize_chain()
        chain = neb.optimized
        logger.info(f"NEB completed in {len(neb.chain_trajectory)} steps")
    except NoneConvergedException as e:
        logger.warning(f"NEB did not converge: {e}")
        return None, None, f"NEB did not converge: {e}"
    except Exception as e:
        logger.error(f"NEB failed with error: {e}")
        return None, None, f"NEB error: {e}"

    if chain is None:
        return None, None, "Chain is None after NEB"

    # Get TS guess (highest energy node)
    ts_idx = np.argmax(chain.energies)
    ts_node = chain[ts_idx]
    logger.info(f"TS guess at node {ts_idx}, energy: {chain.energies[ts_idx]:.6f} Ha")

    if ts_idx == 0 or ts_idx == len(chain) - 1:
        logger.warning("TS guess is at endpoint - not a valid TS guess")
        return chain, {"converged": False, "n_neg_freqs": -1, "ts_node": ts_node}, "TS at endpoint"

    # Try to optimize to TS
    logger.info("Attempting TS optimization on TS guess...")
    try:
        ts_result = engine.compute_transition_state(ts_node)

        if hasattr(ts_result, 'success') and not ts_result.success:
            msg = getattr(ts_result, 'message', 'Unknown error')
            logger.warning(f"TS optimization failed: {msg}")
            return chain, {"converged": False, "n_neg_freqs": -1, "ts_node": ts_node}, f"TS opt failed: {msg}"

        # Get the optimized node
        if hasattr(ts_result, 'structure'):
            ts_node_optimized = ts_result
        else:
            ts_node_optimized = ts_result

        # Compute Hessian on the optimized TS
        logger.info("Computing Hessian...")
        hessian = engine.compute_hessian(ts_node_optimized)
        n_neg = count_negative_frequencies(hessian)

        logger.info(f"Hessian computed. Negative frequencies: {n_neg}")

        is_valid_ts = (n_neg == 1)
        if is_valid_ts:
            logger.info("SUCCESS: Found valid TS with exactly 1 negative frequency!")
        else:
            logger.warning(f"FAILED: Got {n_neg} negative frequencies (expected 1)")

        return chain, {
            "converged": is_valid_ts,
            "n_neg_freqs": n_neg,
            "ts_node": ts_node_optimized,
        }, ""

    except Exception as e:
        logger.error(f"TS verification failed: {e}")
        return chain, None, f"TS verification error: {e}"


def generate_parameter_grid() -> list[dict]:
    """Generate parameter combinations to test.

    Strategy:
    - Loosen overall gradient thresholds for faster NEB convergence
    - Keep TS-specific thresholds tight to ensure good TS guess
    """
    configs = [
        # Very fast
        {"rms_grad_thre": 0.04, "max_rms_grad_thre": 0.08, "ts_grad_thre": 0.07,
         "ts_spring_thre": 0.03, "max_steps": 300, "barrier_thre": 0.2},
        # Balanced fast
        {"rms_grad_thre": 0.03, "max_rms_grad_thre": 0.06, "ts_grad_thre": 0.05,
         "ts_spring_thre": 0.02, "max_steps": 400, "barrier_thre": 0.15},
        # Default-ish
        {"rms_grad_thre": 0.02, "max_rms_grad_thre": 0.05, "ts_grad_thre": 0.05,
         "ts_spring_thre": 0.02, "max_steps": 500, "barrier_thre": 0.1},
        # Tight TS (should work better)
        {"rms_grad_thre": 0.03, "max_rms_grad_thre": 0.06, "ts_grad_thre": 0.03,
         "ts_spring_thre": 0.015, "max_steps": 400, "barrier_thre": 0.1},
        # Very tight TS
        {"rms_grad_thre": 0.02, "max_rms_grad_thre": 0.04, "ts_grad_thre": 0.02,
         "ts_spring_thre": 0.01, "max_steps": 500, "barrier_thre": 0.1},
        # Relaxed barrier
        {"rms_grad_thre": 0.04, "max_rms_grad_thre": 0.08, "ts_grad_thre": 0.05,
         "ts_spring_thre": 0.025, "max_steps": 300, "barrier_thre": 0.3},
    ]

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Optimize NEB convergence parameters"
    )
    parser.add_argument("--reactants", type=Path, required=True,
                        help="Path to reactants XYZ file")
    parser.add_argument("--products", type=Path, required=True,
                        help="Path to products XYZ file")
    parser.add_argument("--program", type=str, default="xtb",
                        help="Quantum program for engine (default: xtb)")
    parser.add_argument("--output", type=Path, default=Path("convergence_results.csv"),
                        help="Output CSV file for results")
    parser.add_argument("--nimages", type=int, default=15,
                        help="Number of NEB images (default: 15)")
    parser.add_argument("--v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Validate inputs
    if not args.reactants.exists():
        raise FileNotFoundError(f"Reactants file not found: {args.reactants}")
    if not args.products.exists():
        raise FileNotFoundError(f"Products file not found: {args.products}")

    logger.info("=" * 60)
    logger.info("NEB Convergence Parameter Optimizer")
    logger.info("=" * 60)
    logger.info(f"Reactants: {args.reactants}")
    logger.info(f"Products: {args.products}")
    logger.info(f"Program: {args.program}")
    logger.info(f"NImages: {args.nimages}")

    # Set up engine
    engine = QCOPEngine(program=args.program)

    # Generate parameter grid
    param_configs = generate_parameter_grid()
    logger.info(f"Testing {len(param_configs)} parameter configurations")

    results: list[OptimizationResult] = []

    # Test each configuration
    for i, config in enumerate(param_configs):
        logger.info("-" * 40)
        logger.info(f"Config {i+1}/{len(param_configs)}: {config}")

        # Build NEB inputs
        neb_params = NEBInputs(
            climb=False,
            en_thre=0.0001,
            rms_grad_thre=config["rms_grad_thre"],
            max_rms_grad_thre=config["max_rms_grad_thre"],
            ts_grad_thre=config["ts_grad_thre"],
            ts_spring_thre=config["ts_spring_thre"],
            barrier_thre=config["barrier_thre"],
            early_stop_force_thre=0.03,
            negative_steps_thre=5,
            positive_steps_thre=10,
            max_steps=config["max_steps"],
            do_elem_step_checks=False,
            v=args.v,
        )

        start_time = time.time()

        try:
            chain, ts_result, error_msg = run_neb_and_verify(
                reactants_file=args.reactants,
                products_file=args.products,
                engine=engine,
                neb_params=neb_params,
                max_chain_images=args.nimages,
                v=args.v,
            )

            runtime = time.time() - start_time

            # Determine success
            success = False
            if ts_result is not None:
                success = ts_result.get("converged", False)

            result = OptimizationResult(
                params=config,
                neb_steps=len(chain.chain_trajectory) if chain else 0,
                ts_converged=ts_result.get("converged", False) if ts_result else False,
                n_neg_freqs=ts_result.get("n_neg_freqs", -1) if ts_result else -1,
                success=success,
                error_msg=error_msg,
                runtime_seconds=runtime,
            )

        except Exception as e:
            import traceback
            runtime = time.time() - start_time
            logger.error(f"Exception during test: {e}")
            logger.error(traceback.format_exc())
            result = OptimizationResult(
                params=config,
                success=False,
                error_msg=str(e),
                runtime_seconds=runtime,
            )

        results.append(result)
        logger.info(f"Result: {result}")

    # Summary
    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)

    # Find best result
    successful_results = [r for r in results if r.success]

    if successful_results:
        # Sort by runtime (fastest first)
        successful_results.sort(key=lambda r: r.runtime_seconds)
        best = successful_results[0]

        logger.info(f"\nBest configuration (fastest successful):")
        logger.info(f"  Parameters: {best.params}")
        logger.info(f"  Runtime: {best.runtime_seconds:.1f}s")
        logger.info(f"  NEB Steps: {best.neb_steps}")
        logger.info(f"  Negative frequencies: {best.n_neg_freqs}")
    else:
        logger.warning("\nNo configuration produced a valid TS!")

    # Log all results
    logger.info("\nAll results:")
    for r in results:
        logger.info(f"  {r}")

    # Save to CSV
    import csv
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rms_grad_thre', 'max_rms_grad_thre', 'ts_grad_thre',
            'ts_spring_thre', 'max_steps', 'barrier_thre',
            'neb_steps', 'ts_converged', 'n_neg_freqs',
            'success', 'runtime_seconds', 'error_msg'
        ])
        writer.writeheader()
        for r in results:
            row = r.params.copy()
            row.update({
                'neb_steps': r.neb_steps,
                'ts_converged': r.ts_converged,
                'n_neg_freqs': r.n_neg_freqs,
                'success': r.success,
                'runtime_seconds': r.runtime_seconds,
                'error_msg': r.error_msg,
            })
            writer.writerow(row)

    logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
