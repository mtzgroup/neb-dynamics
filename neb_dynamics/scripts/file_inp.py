from qcio import FileInput, Structure
from qcop import compute


inp = FileInput(cmdline_args=[
                'run', '--start', 'start.xyz', "--end", "end.xyz", "--inputs", "inputs.toml"])

# Option One
start_stuct = Structure.open("start.xyz")
end_struct = Structure.open("end.xyz")
inp.files['start.xyz'] = start_stuct.to_xyz()
inp.files['start.xyz'] = start_stuct.to_xyz()

# Option Two
inp.add_file("start.xyz")
inp.add_file("end.xyz")
inp.add_file("inputs.toml")

# Option Three (if you have a directory with everything you need)
inp.add_files("myinputs")

prog_out = compute('mepd', inp)

# To save all output files
prog_out.save_files('myoutputs')
