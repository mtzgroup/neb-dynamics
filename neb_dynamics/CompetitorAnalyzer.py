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

    def __post_init__(self):
        self.structures_dir = self.comparisons_dir / "structures"
        self.inputs_dir = self.comparisons_dir / "input_files"

        if not self.structures_dir.exists():
            raise ValueError("No structures directory exists. Please make it")

        if not self.inputs_dir.exists():
            raise ValueError("No inputs directory exists. Please make it")

        if self.method.lower() in self.DLFIND_NAMES:
            self.out_folder = self.comparisons_dir / "dlfind"
            self.command = "source activate rp \nml TeraChem/2020.11-intel-2017.0.098-MPICH2-1.4.1p1-CUDA-9.0.176\nterachem input.in"
            self.input_file = self.inputs_dir / "dlfind_input.txt"

        elif self.method.lower() in self.PYGSM_NAMES:
            self.out_folder = self.comparisons_dir / "pygsm"
            self.input_file = self.inputs_dir / "pygsm_input.txt"
            self.command = "source activate gsm_env \ngsm  -xyzfile initial_guess.xyz -mode DE_GSM -num_nodes 15 -package TeraChem -lot_inp_file input.in -interp_method Geodesic -coordinate_type DLC -CONV_TOL 0.001 -reactant_geom_fixed -product_geom_fixed > log 2>&1"
        else:
            raise ValueError(f"Invalid input method: {self.method}")

    def edit_submit_file(self, path):
        submit_path = path / "submit.sh"
        f = open(submit_path)
        data = f.read()
        lines = data.splitlines()

        if lines[-1] == "#endfile":
            f2 = open(submit_path, "a+")
            f2.write(self.command)
            f2.close()

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
        shutil.copy(self.comparisons_dir / "submit_template.sh", out_path / "submit.sh")

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

    def submit_all_jobs(self):
        for fp in self.out_folder.iterdir():
            submit_file = fp / "submit.sh"
            with self.working_directory(fp):
                subprocess.Popen(["sbatch", str(submit_file.resolve())])
                time.sleep(0.1)
