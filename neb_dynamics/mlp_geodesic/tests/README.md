# Test Suite for MLP Geodesic Optimization

This directory contains three test cases to validate the MLP Geodesic path optimization code. These tests demonstrate the process of taking a Morse-geodesic interpolated path and refining it on a machine learned (ML) potential energy surface (PES).

## Directory Contents

* **Initial Paths (`morse_geodesic_path_*.xyz`)**: These are the input files for the MLP Geodesic optimization, which are paths obtained from Morse-geodesic. 
* **Optimized Paths (`esen_sm_mlp_geodesic_path_*.xyz`)**: These are the final output files which represent the reaction paths after being optimized on the eSEN-sm-cons ML PES.
* **Log Files (`*.log`)**: These files capture the standard output from the `mlp_geodesic` optimization step.

## How to Run the Tests

The tests are run from the command line. Each execution takes an initial path from the `morse_geodesic` tool and uses it as input for the MLP Geodesic optimization.

The general command structure is as follows:
```bash
python ../cli.py [initial_path.xyz] [output_path.xyz] &> [output.log]
```

### Example Command

To run the test for the HCN to HNC isomerization:
```bash
python ../cli.py morse_geodesic_path_HCN_to_HNC.xyz esen_sm_mlp_geodesic_path_HCN_to_HNC.xyz &> esen_sm_mlp_geodesic_path_HCN_to_HNC.log
```
*Note: This command assumes you have the required MLP model file available and specify it with the `--model-path` argument, as shown in the main README.*

## Test Cases Included

This directory contains tests for the following chemical systems:

* **`ethene_dehydrogenation`**: The dehydrogenation of ethene.
* **`HCN_to_HNC`**: The isomerization of hydrogen cyanide to hydrogen isocyanide.
* **`vinyl_alcohol_to_acetaldehyde`**: The tautomerization of vinyl alcohol to acetaldehyde.

