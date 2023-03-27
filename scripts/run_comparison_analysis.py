from pathlib import Path
from neb_dynamics.CompetitorAnalyzer import CompetitorAnalyzer

comparisons_dir = Path("/home/jdep/T3D_data/msmep_draft/comparisons") 
ca = CompetitorAnalyzer(comparisons_dir=comparisons_dir,method='dlfind')
ca2 = CompetitorAnalyzer(comparisons_dir=comparisons_dir,method='pygsm')
ca3 = CompetitorAnalyzer(comparisons_dir=comparisons_dir,method='nebd')

ca3.submit_all_jobs()


