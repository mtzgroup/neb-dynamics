from pathlib import Path
import shutil

data_dir = Path("/home/jdep/T3D_data/nebts_repro/configurations/")

output_dir = Path("/home/jdep/T3D_data/msmep_draft/comparisons_benchmark")

output_dir.mkdir()

for xyz in list(data_dir.glob("*.xyz")):

    sys_dir_name = xyz.stem.split("-")[0]
    filename = xyz.stem.split("-")[1]+'.xyz'
    
    sys_dir = output_dir / sys_dir_name
    if not sys_dir.exists():
        sys_dir.mkdir()
    shutil.copy(xyz, sys_dir / filename)
