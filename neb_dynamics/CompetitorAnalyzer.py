import contextlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompetitorAnalyzer:
    comparisons_dir: Path
    method: str

    DLFIND_NAMES = ["dlfind", "dl-find"]
    PYGSM_NAMES = ["pygsm", "py-gsm"]
    NEB_DYNAMICS_NAMES = [
        "nebd",
        "neb-d",
        "nebdynamics",
        "neb-dynamics",
        "neb_dynamics",
    ]

    ASNEB_NAMES = [
        "as-neb",
        'asneb',
        'msmep',
        'autosplitneb',
        'autosplittingneb'
    ]
    ASNEB_DFT_NAMES = [
        "asneb_dft",
        "asneb-dft"
    ]

    def __post_init__(self):
        self.structures_dir = self.comparisons_dir / "structures"
        self.inputs_dir = self.comparisons_dir / "input_files"

        if not self.structures_dir.exists():
            raise ValueError("No structures directory exists. Please make it")

        if not self.inputs_dir.exists():
            raise ValueError("No inputs directory exists. Please make it")

        if self.method.lower() in self.DLFIND_NAMES:
            self.out_folder = self.comparisons_dir / "dlfind"
            self.input_file = self.inputs_dir / "dlfind_input.txt"
            self.command = """
source activate rp
ml TeraChem/2020.11-intel-2017.0.098-MPICH2-1.4.1p1-CUDA-9.0.176
terachem input.in
"""

        elif self.method.lower() in self.PYGSM_NAMES:
            self.out_folder = self.comparisons_dir / "pygsm"
            self.input_file = self.inputs_dir / "pygsm_input.txt"
            self.command = """
ml TeraChem/2020.11-intel-2017.0.098-MPICH2-1.4.1p1-CUDA-9.0.176
eval "$(conda shell.bash hook)"
conda activate gsm_env
/home/jdep/.conda/envs/gsm_env/bin/gsm  -xyzfile initial_guess.xyz -mode DE_GSM -num_nodes 15 -package TeraChem -lot_inp_file input.in -interp_method Geodesic -coordinate_type DLC -CONV_TOL 0.001 -reactant_geom_fixed -product_geom_fixed > log 2>&1
            """

        elif self.method.lower() in self.NEB_DYNAMICS_NAMES:
            self.out_folder = self.comparisons_dir / "nebd"
            self.input_file = self.inputs_dir / "nebd_input.txt"
            self.command = """
eval "$(conda shell.bash hook)"
export OMP_NUM_THREADS=1
conda activate rp
create_neb_from_geodesic.py  -f initial_guess.xyz -c 0 -s 1 &> out.txt   
            """

        elif self.method.lower() in self.ASNEB_NAMES:
            self.out_folder = self.comparisons_dir / "asneb"
            self.input_file = self.inputs_dir / "nebd_input.txt"
            self.command = """
eval "$(conda shell.bash hook)"
export OMP_NUM_THREADS=1
conda activate rp
create_msmep_from_geodesic.py -f ./initial_guess.xyz  -c 0 -s 1 &> out.txt
            """
            
        elif self.method.lower() in self.ASNEB_DFT_NAMES:
            self.out_folder = self.comparisons_dir / "asneb_dft"
            self.input_file = self.inputs_dir / "nebd_input.txt"
            self.command = """
eval "$(conda shell.bash hook)"
export OMP_NUM_THREADS=1
conda activate rp
create_msmep_from_geodesic.py -f ./initial_guess.xyz  -c 0 -s 1 -nc node3d_tc &> out.txt
            """
            
        else:
            raise ValueError(f"Invalid input method: {self.method}")

    def create_all_files_and_folders(self):
        for rn_path in self.reaction_initial_guesses_dirs:
            self.create_directory_and_input_file(rn_path)

    def edit_submit_file(self, path):
        submit_path = path / "submit.sh"

        f = open(submit_path, "a+")
        f.write(self.command)
        f.close()

    def create_directory_and_input_file(self, path):
        rn = path.stem
        guess_path = path / "initial_guess.xyz"
        if not guess_path.exists():
            print(f">>>{rn} does not have an interpolation. Check this out.")
            return

        out_path = self.out_folder / rn
        out_path.mkdir(parents=True, exist_ok=True)

        shutil.copy(self.input_file, out_path / "input.in")
        shutil.copy(guess_path, out_path / "initial_guess.xyz")
        # shutil.copy(self.comparisons_dir / "submit_template.sh", out_path / "submit.sh")
        shutil.copy(self.comparisons_dir / "submit_template_cpu.sh", out_path / "submit.sh")

        self.edit_submit_file(out_path)

    @property
    def reaction_initial_guesses_dirs(self):
        return Path.glob(self.structures_dir, "*")

    @property
    def available_reaction_names(self):
        return [path.stem for path in Path.glob(self.structures_dir, "*")]

    @contextlib.contextmanager
    def working_directory(self, path):
        """Changes working directory and returns to previous on exit."""
        prev_cwd = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev_cwd)
            
    def submit_a_job_by_name(self, name):
        fp = self.out_folder / name
        submit_file = fp / "submit.sh"
        with self.working_directory(fp):
            subprocess.Popen(["sbatch", "submit.sh"])
            time.sleep(0.1)

    def submit_all_jobs(self):
        for fp in self.out_folder.iterdir():
            submit_file = fp / "submit.sh"
            with self.working_directory(fp):
                subprocess.Popen(["sbatch", "submit.sh"])
                time.sleep(0.1)
