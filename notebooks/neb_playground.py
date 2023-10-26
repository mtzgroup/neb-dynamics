# -*- coding: utf-8 -*-
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.Chain import Chain
from neb_dynamics.Node3D import Node3D
from neb_dynamics.Inputs import ChainInputs, NEBInputs
from neb_dynamics.NEB import NEB, NoneConvergedException

from retropaths.abinitio.trajectory import Trajectory

# t = Trajectory.from_xyz('/home/jdep/T3D_data/template_rxns/Chan-Rearrangement/traj_0-79.xyz')
t = Trajectory.from_xyz('/home/jdep/T3D_data/geometry_spawning/claisen_results/claisen_ts_profile.xyz')

r,p = t[0], t[-1]

gi = Trajectory([r,p]).run_geodesic(nimages=10)

import numpy as np

flush_steps = [5, 10, 20, 50, 100]
flush_thres = [.50, .80, .90, .99]

from itertools import product

len(results_nebs)

len(list(product(flush_steps, flush_thres)))

results_nebs = []
for fs, ft in list(product(flush_steps, flush_thres)):
    print(f"doing: force steps:{fs} force thre: {ft}")
    cni = ChainInputs(step_size=0.33*gi[0].atomn, min_step_size=.01*gi[0].atomn)
    chain = Chain.from_traj(gi, cni)

    nbi = NEBInputs(v=True,tol=0.001, bfgs_flush_steps=fs, bfgs_flush_thre=ft, max_steps=500)

    n = NEB(initial_chain=chain, parameters=nbi)
    try:
        n.optimize_chain()
        results_nebs.append(n)
    except NoneConvergedException as e:
        results_nebs.append(e.obj)

# +
print(f"doing reference")
cni = ChainInputs(step_size=0.33*gi[0].atomn, min_step_size=.01*gi[0].atomn)
chain = Chain.from_traj(gi, cni)

nbi = NEBInputs(v=True,tol=0.001, bfgs_flush_steps=fs, bfgs_flush_thre=ft, max_steps=500, do_bfgs=False)

n = NEB(initial_chain=chain, parameters=nbi)
n.optimize_chain()
# -



for neb, params in zip(results_nebs, list(product(flush_steps, flush_thres))):
    fs, ft = params
    neb.write_to_disk(Path(f"/home/jdep/T3D_data/bfgs_results/claisen_fs_{fs}_ft_{ft}.xyz"))

import pandas as pd

conditions = list(product(flush_steps, flush_thres))
results = [] # f_steps, f_thre, n_steps
for n_result, (f_steps, f_thre)  in zip(results_nebs, conditions):
    results.append([f_steps, f_thre, len(n_result.chain_trajectory)])

df = pd.DataFrame(results, columns=['f_steps','f_thre','n_steps'])

df.sort_values(by='n_steps')

# # Foobar

from neb_dynamics.TreeNode import TreeNode

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig2/react_msmep/")

h.output_chain.to_trajectory()

# from neb_dynamics.MSMEPAnalyzer import MSMEPAnalyzer
from neb_dynamics.NEB import NEB
from retropaths.abinitio.tdstructure import TDStructure
from pathlib import Path

p = Path('/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/')
msma = MSMEPAnalyzer(parent_dir=p, msmep_root_name='react')
msma_dft = MSMEPAnalyzer(parent_dir=p, msmep_root_name='dft_early_stop')



name = 'system23'
msma_obj = msma
# n = NEB.read_from_disk(p / name / 'dft_early_stop_msmep' / 'node_0.xyz')
n = NEB.read_from_disk(msma_obj.parent_dir / name /  (str(msma_obj.msmep_root_name)+"_msmep") / 'node_0.xyz')
h = msma_obj.get_relevant_tree(name)
out = msma_obj.get_relevant_chain(name)
sp = msma_obj.get_relevant_saddle_point(name)

n.optimized.get_ts_guess()

out.plot_chain()

distances = [msma_obj._distance_to_sp(c, sp) for c in n.chain_trajectory]

out.to_trajectory()

sp

out.to_trajectory()

plt.plot(distances,'o-')

np.argmin(distances)

sp

n.chain_trajectory[7].get_ts_guess()

sp

out.get_ts_guess()

out.plot_chain()

sp

out.get_ts_guess()



sp40 = TDStructure.from_xyz(p/'system40'/'sp.xyz')

tsg = out.to_trajectory()[26]

sp.tc_model_method = 'B3LYP-D3BJ'
sp.tc_model_basis = 'def2-svp'
sp.tc_kwds = {'reference':'uks'}

sp_opt = sp.tc_geom_optimization('ts')

sp_opt

tsg

tsg.update_tc_parameters(sp)

ts = tsg.tc_geom_optimization('ts')

ts.tc_freq_calculation()

sp

ts.to_xyz(p / name / 'sp_discovered.xyz')

# # TCPB Shit

from retropaths.abinitio.tdstructure import TDStructure

import tcpb 

import os
import sys
import atexit
import time
import signal
import socket
import logging
import subprocess
import tempfile
from contextlib import closing

# +
"""A parallel engine using a farm of TCPB servers as backend."""
import threading
import socket
import numpy as np
from collections import namedtuple, OrderedDict, defaultdict
from multiprocessing.dummy import Pool
import os


logger = logging.getLogger(__name__)


TCPBAddress = namedtuple("TCPBWorker", ('host', 'port'))
"""A TCPB server address, in the form of host/port tuple"""

# -

DEBYE_TO_AU = 0.393430307
class TCPBEngine:
    """TCPB engine using a pool of TeraChem TCPB workers as backend

    Example:  If you have 1 server on node fire-20-05 on ports 30001,
    and two on fire-13-01 on ports 40001, 40002, then
    >>> servers = [('fire-20-05', 30001),
                   ('fire-13-01', 40001), ('fire-13-01', 40002)]
    >>> engine = TCPBEngine(servers)
    >>> engine.status()
    TCPBEngine has 3 workers connected.
      [Ready] fire-20-06:30001
      [Ready] fire-13-01:40001
      [Ready] fire-13-01:40002
    Total number of available workers: 3

    Alternatively, one can let the engine to discover servers running on a node
    with the `scan_node` method.
    >>> engine = TCPBEngine()
    >>> engine.status()
    TCPBEngine has 0 workers connected.
    Total number of available workers: 0
    >>> engine.scan_node('fire-13-01')
    >>> engine.status()
    TCPBEngine has 2 workers connected.
      [Ready] fire-13-01:40001
      [Ready] fire-13-01:40002
    Total number of available workers: 2

    Then one can setup() and run calculations like on a normal engine.
    To add or remove servers from the worker pool, use `add_worker` and `remove_worker`.

    """


    @property
    def max_workers(self):
        """Redefine max_workers here to basically reflect the length of worker list,
        and at the same time make it read-only"""
        return len(self.workers)

    def __init__(self, workers=(), cwd="", max_cache_size=300, resubmit=True, propagate_orb=True,
                 *tcpb_args, **tcpb_kwargs):
        """Initialize TCPBWorkers and the engine

        Args:
            workers:    List/tuple of TCPBAddress [in forms of (host, port)]
                        which specifies the initial workers
            cwd:        Current working directory for terachem instance
            max_cache_size: Maximum number of cached results.
            resubmit:   Whether to resubmit a job if a worker becomes unavailable
            propagate_orb:  Whether to use the orbitals of the previous calculation as
                        starting guess for the next.  Currently, one orbital set is saved
                        for each host, and data transfer between host is not supported.
            """
        self.max_cache_size = max_cache_size
        self.workers = OrderedDict()
        self.tcpb_args = tcpb_args
        self.tcpb_kwargs = tcpb_kwargs
        self.error_count = defaultdict(int)
        self.options = dict()
        self.orbfiles = {}
        self.lock = threading.Lock()
        self.resubmit = resubmit
        self.propagate_orb = propagate_orb
        self.cwd = cwd
        for worker in workers:
            self.add_worker(worker)

    def _try_port(self, address):
        """Used by scan_node to scan a port on a host.
        Returns a connected TCPB client if the port hosts a TCPB server.
        """
        host, port = address
        try:
            # Check if the port is open.  Would get a RuntimeError if not
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)  # A short timeout of 1 sec
            sock.connect(address)
            client = tcpb.TCProtobufClient(host, port, *self.tcpb_args, **self.tcpb_kwargs)
            client.tcsock = sock
            client.tcsock.settimeout(30.0)
            if client.is_available():   # Check if accepting job
                return True, port, client
            else:
                return False, port, client
        except Exception as e:
            return None, port, e

    def scan_node(self, host, port_range=(1024, 65536), threads=None,
                  busy_servers=False, max_servers=None):
        """Scan node for all ports that have TCPB server running.
        A worker will be added to each available server.

        * Remember, with great power, comes great responsibility.

        Args:
            host:   Hostname on which to search for servers.
            port_range: Range to search.  Default is all valid ports.
            threads:    Number of threads.  Default is number of CPUs.
            busy_servers:   Whether or not to add servers that are
                    currently busy computing jobs.  Default is False.
            max_servers:  Maximum number of servers to connect. Default is no limit.
        """
        pool = Pool(threads)
        logger.info('Scanning host %s port %d to %d for TCPB servers.',
                    host, *port_range)
        n_added = 0
        addrs = ((host, port) for port in xrange(*port_range)
                 if (host, port) not in self.workers)
        for success, port, client in pool.imap_unordered(self._try_port, addrs, 100):
            if success is not None and success or busy_servers:
                address = TCPBAddress(host, port)
                logger.info('Found TCPB server at port %d', port)
                if n_added == max_servers:
                    logger.info("Number of servers added reached maximum %d.  Not adding.", max_servers)
                else:
                    self.workers[address] = client
        pool.close()
        pool.join()
        logger.info("Found %d new workers.  The engine now has %d workers.",
                    n_added, self.max_workers)

    @staticmethod
    def validate_address(address):

        if not isinstance(address, (tuple, list)) or len(address) != 2:
            raise TypeError("Worker ID must be a (host, port) tuple")
        if not isinstance(address[0], str):
            raise TypeError("Hostname must be a string")
        if not isinstance(address[1], int) or address[1] < 0:
            raise TypeError("port number must be a positive integer")
        return TCPBAddress(*address)

    def add_worker(self, address):
        """Add a tcpb worker to the worker pool"""
        logger.debug('Trying to add worker at address %s', str(address))
        address = self.validate_address(address)
        with self.lock:
            if address in self.workers:
                logger.warn('Address %s cannot be added: already in worker pool.',
                            str(address))
            client = tcpb.TCProtobufClient(address.host, address.port,
                                           *self.tcpb_args, **self.tcpb_kwargs)
            client.connect()
            self.workers[address] = client

    def remove_worker(self, address):
        """Remove a worker from worker pool"""
        address = self.validate_address(address)
        with self.lock:
            if address not in self.workers:
                raise KeyError("Cannot remove worker.  Not in worker pool.")
            try:
                self.workers[address].disconnect()
            except:
                pass
            del self.workers[address]

    def setup(self, mol, restricted=None, closed_shell=None, propagate_orb=None, **kwargs):
        """Setup the client based on information provided in a Molecule object
        Args:
            mol:    A Molecule object, the info in which will be used to
                    configure the client
            restricted: Whether to use restricted WF. Default is only to use
                    restricted when multiplicity is 1.
            closed_shell: Whether the system is closed shell.  Default is
                    to use closed shell wave function when multiplicity is 1
                    and restricted has not been set to false.
            propagate_orb:   Whether to use the previous orbital set as starting guess
                    for the next.  Current, cannot cross hosts.
            Other keyword arguments will be passed to job_spec() directly
        """
        with self.lock:
            if propagate_orb is not None:
                self.propagate_orb = propagate_orb
            self.orbfiles = {}
            if mol is None:
                atoms = charge = spinmult = None
            else:
                if not isinstance(mol, Molecule):
                    raise TypeError("mol must be a Molecule object.")
                atoms = list(mol.atoms)
                charge, spinmult = mol.charge, mol.multiplicity
                if restricted is not None:
                    self.options["restricted"] = restricted
                elif self.options.setdefault('restricted') is None:
                    self.options['restricted'] = mol.multiplicity == 1
                    logger.info('Infering `restricted` from multiplicity. (%s)', str(self.options['restricted']))
                if closed_shell is not None:
                    self.options['closed_shell'] = closed_shell
                elif self.options.setdefault('closed_shell') is None:
                    self.options['closed_shell'] = self.options['restricted'] and mol.multiplicity == 1
                    logger.info('Infering `closed_shell` from multiplicity. (%s)', str(self.options['closed_shell']))
            self.options.update(dict(atoms=atoms, charge=charge, spinmult=spinmult, **kwargs))

    def parse_options(self, args, separator='='):
        """Parse TeraChem arguments in a list of key/value strings"""
        if args is None:
            return
        tc_options = {'method': 'b3lyp', 'basis': '3-21g'}
        if isinstance(args, dict):  # No need to parse
            tc_options.update(args)
        else: 
            logger.info('Parsing TeraChem options')
            if isinstance(args, basestring):     # If args is just a string, assume it is the filename
                args = [args]
            for entry in args:
                raw = entry.split(separator, 1)
                if len(raw) == 1: 
                    filename = raw[0]
                    logging.info('Loading TeraChem input from file %s', filename)
                    with open(filename) as f:
                        for line in f:
                            splitted = line.strip().split(None, 1)
                            if len(splitted) != 2:
                                logger.info("Input file line ignored: %s", line.strip())
                            else:
                                key, value = splitted
                                key = key.lower()
                                value = value.strip()
                                if key in ['charge', 'spinmult']:
                                    value = int(value)
                                tc_options[key] = value
                else:
                    key, value = raw
                    key = key.lower()
                    value = value.strip()
                    if value[0] == '"' and value[-1] == '"' or value[0] == "'" and value[-1] == "'":
                        value = value[1:-1]
                    if key in ['charge', 'spinmult']:
                        value = int(value)
                    tc_options[key] = value
            for key, value in tc_options.iteritems():
                logger.info('%12s %12s', key, value)

        with self.lock:
            if tc_options['method'].lower().startswith('r'):
                tc_options['restricted'] = True
                tc_options['method'] = tc_options['method'][1:]
            elif tc_options['method'].lower().startswith('u'):
                tc_options['restricted'] = False
                tc_options['closed_shell'] = False
                tc_options['method'] = tc_options['method'][1:]
                self.closed_shell = False
            carry_over = ["atoms", "charge", "spinmult", "restricted", "closed_shell"]
            self.options = {key: self.options.get(key, None) for key in carry_over} 
            self.options.update(tc_options)

    def worker_by_index(self, index):
        """Get the client associated with a worker's index
        Returns a (address, client) tuple
        """
        with self.lock:
            return list(self.workers.items())[index]

    def compute__(self, geom, job_type, workerid, *args, **kwargs):
        address, worker = self.worker_by_index(workerid)
        with self.lock:
            options = self.options.copy()
        options.update(kwargs)
        if self.propagate_orb and 'guess' not in kwargs:
            if address.host in self.orbfiles:
                options['guess'] = self.orbfiles[address.host]
                logger.debug('Using previous orbitals as starting guess: %s', options['guess'])
            else:
                logger.debug('No previous orbital guess available')
        elif 'guess' in kwargs:
            logger.debug('Using %s as initial orbital guess', options['guess'])
        logger.debug("Perform calculation on worker %s:%d", *address)
        result = worker.compute_job_sync(jobType=job_type, geom=geom, unitType="bohr",
                                         bond_order=True, *args, **options)
        if self.propagate_orb:
            if 'orbfile' in result:
                self.orbfiles[address.host] = result['orbfile']
            elif 'orbfile_a' in result and 'orbfile_b' in result:
                self.orbfiles[address.host] = result['orbfile_a'] + " " + result['orbfile_b']
            else:
                logger.debug('Result does not contain orbital file info.')
        # Try to extract S^2
        if options.get('closed_shell', True):
            s2 = 0.0
        else:
            s2 = np.nan
            job_dir = result['job_dir']
            job_id = result['server_job_id'] 
            
            #If using pre-October 2020 TeraChem, use commented output_file line instead
            #output_file = result['output_file'] = "{}/{}.log".format(job_dir, job_id)
            output_file = result['output_file'] = "{}/{}/tc.out".format(self.cwd,job_dir)
            try:
                with open(output_file) as f:
                    for line in f:
                        if line.startswith('SPIN S-SQUARED:'):
                            s2 = float(line.replace('SPIN S-SQUARED:', '').split()[0])
                            break
            except IOError or OSError:
                logger.warn('Cannot access TeraChem output to extract <S^2>')
        result['S2'] = s2
        result['S2_exact'] = (options['spinmult'] ** 2 - 1) * 0.25
        # Convert dipole from Debye to a.u. (TCPB returns debye)
        result['dipole_vector'] = np.array(result['dipole_vector']) * DEBYE_TO_AU
        result['dipole_moment'] = np.array(result['dipole_moment']) * DEBYE_TO_AU
        return result

    def available__(self, workerid):
        """Check if a TCPB client is available for submission
        This subroutine itself should be exception proof

        workerid:   Which worker to check"""
        if workerid < 0 or workerid >= len(self.workers):
            logger.error("Worker index %d is out of range!", workerid)
            return False
        address, worker = self.worker_by_index(workerid)
        try:
            result = worker.is_available()
            if not result:
                logger.warn("Worker %d at address %s reported as unavailable.",
                            workerid, str(address))
            return result
        except (RuntimeError, tcpb.exceptions.ServerError, tcpb.exceptions.TCPBError):
            logger.debug("Worker %s:%d reported ERROR.", *address)
            count = self.error_count[address] + 1
            self.error_count[address] = count
            if count > 5:
                self.remove_worker(address)
                logger.info('Worker %s:%d removed', *address)
            return False
        except Exception:
            logger.error('Unknown exception thrown in TCPBEngine.available__', exc_info=True)
            return False

    def status(self):
        """Report on worker status"""
        n_ready = 0
        with self.lock:
            print ("TCPBEngine has %d workers connected." % self.max_workers)
            for key, worker in self.workers.iteritems():
                try:
                    if worker.is_available():
                        status = "Ready"
                        n_ready += 1
                    else:
                        status = " Busy"
                except RuntimeError:
                    status = "Error"
                print ("  [{}] {}:{}".format(status, *key))
            print ("Total number of available workers: %d" % n_ready)

    def disconnect(self):
        """Disconnect all servers"""
        with self.lock:
            logger.info('Trying to disconnect the clients and shutdown any housed servers.')
            try:
                for worker in self.workers.itervalues():
                    worker.disconnect()
            except:
                pass
            self.workers = {}
            for server in getattr(self, 'servers', ()):
                logger.debug('Trying to shut down hosted server %s', repr(server))
                server.shutdown()


class TCPBServer(object):
    """Maintains a TCPB server instance"""
    def __init__(self, terachem_exe='terachem', port=None, gpus='0',
                 output='server.out', wait=3, cwd=None):
        """Start a TeraChem protobuf server.

        Args:
            terachem_exc:   ['terachem'] TeraChem executable
            port:           [None] If specified, use this port between server and client
                            (instead of choosing a random unused port)
            gpus:           ['0'] List of GPUs to be used by the server
            output:         Filename (or file object) for the terachem output file
            cwd:            Working directory for the server.  The directory must exist!
            wait:           [3] Number of seconds to wait before the client is connected.

        All other arguments are passed to TCPB (methods, basis, etc ...)
        """
        self.gpus = gpus
        self.wait = wait
        self.terachem = terachem_exe
        if isinstance(output, str):
            self.output = open(output, 'w+')
        else:
            self.output = output
        if cwd is None:
            self.cwd = tempfile.mkdtemp()
            # raise AttributeError("No working directory for server specified in yaml file. Please use /u/username as directory (cwd).")
        else:
            self.cwd = cwd
        self.proc = None
        self.wait = wait
        self.start_server(port)
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def find_free_port(self):
        """Identify a free port number to be used.
        This sets the `port` field of the object."""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            self.port = s.getsockname()[1]
            print('Selecting port %d', self.port)

    def start_server(self, port=None):
        """Start terachem server"""
        self.shutdown()
        self.port = port
        print('Starting server for TeraChem engine in %s.', self.cwd)
        if self.port is None:
            self.find_free_port()
        cmd = [self.terachem, '-g', self.gpus, '-s', str(self.port)]
        self.proc = subprocess.Popen(cmd, stdout=self.output, cwd=self.cwd)
        self.active_time = self.wait + time.time()

    def shutdown(self, signum=None, stack=None):
        """Stop the server instance. Not sure if we'll ever need this run manually, but
        it will be executed at least once when the program exits."""
        if signum is not None:
            print('TCPB Servers received shutdown signal : ' + str(signum))
        if self.proc is not None and self.proc.poll() is None:
                print("Shutting down TeraChem server at port %d", self.port)
                self.proc.kill()
        self.proc = None

    @property
    def address(self):
        """Returns the current server address (host, port).  If there is no active server,
        None is returned."""
        if time.time() < self.active_time:
            time.sleep(self.active_time - time.time())
        if self.proc is not None and self.proc.poll() is None:
            return socket.gethostname(), self.port
        else:
            return None


def prepare_engine(gpus, cwd, terachem_exe, wait):
    
    servers, server_addr = [], []
    for gpu_list in gpus.split(','):
        server = TCPBServer(gpus=gpu_list, cwd=cwd, terachem_exe=terachem_exe)
        servers.append(server)
        server_addr.append(server.address)
        
        
        
    pcwd = os.getcwd()
    if pcwd not in sys.path:
        sys.path.append(pcwd)
        
        
    time.sleep(wait)


    engine = TCPBEngine(server_addr, cwd = cwd)
    engine.servers = servers
    
    return engine


def prepare_tcpb_options(td):
    d = {'atoms':td.symbols, 'charge':td.charge, 'spinmult':td.spinmult,
          'closed_shell':False, 'restricted':False
          }
    return d


engine = prepare_engine(gpus, cwd, terachem_exe, wait)

engine.servers[0].active_time

td = TDStructure.from_smiles("O.O")

mol = None
gpus = '0'
wait = 3
cwd = None
terachem_exe = 'terachem'
basestring = 'terachem'

# +
options = prepare_tcpb_options(td)

# restricted = bool(mol.multiplicity == 1)
restricted = True
engine.setup(mol, restricted=restricted, closed_shell=restricted)
engine.parse_options(options)

# +
geom = td.coords

result = engine.compute__(geom, 'gradient', 0)
# -

result["energy"]

result['gradient']

engine.disconnect()

# +

td.gradient_tc_tcpb()
# -

from neb_dynamics.Node3D_TC_TCPB import Node3D_TC_TCPB
from neb_dynamics.Node3D_TC import Node3D_TC

from neb_dynamics.Chain import Chain

from neb_dynamics.Inputs import ChainInputs

c = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Claisen-Rearrangement/initial_guess_msmep.xyz", ChainInputs(node_class=Node3D_TC_TCPB, do_parallel=False))

tr = c.to_trajectory()

td_ref = tr[0]
td_ref.tc_model_method = 'wb97xd3'
td_ref.tc_model_basis = 'def2-svp'
td_ref.tc_kwds = {'restricted': False}

tr.update_tc_parameters(td_ref)

gi = tr.run_geodesic(nimages=8)

# %%time
gi.energies_tc()

# %%time
[td.energy_tc_tcpb() for td in gi]



# # BFGS



import numpy as np

# +
import numpy as np
from numpy.linalg import inv
from scipy import optimize as opt
import math

class BFGS:
    """
    Constructor BFGS (gracefully stolen from https://github.com/Paulnkk/Nonlinear-Optimization-Algorithms/blob/main/bfgs.py)
    """
    def __init__ (self, f, fd, H, xk, eps):
        self.fd = fd
        self.H = H
        self.xk = xk
        self.eps = eps
        self.f = f
        return
    """
    BFGS-Method 
    """

    def work (self):
        f = self.f
        fd = self.fd
        H = self.H
        xk = self.xk
        eps = self.eps
        """
        Initial Matrix for BFGS (Identitiy Matrix)
        """
        E =  np.array([   [1.,         0.],
                            [0.,         1.] ])
        xprev = xk
        it = 0
        maxit = 10000

        while (np.linalg.norm(fd(xk)) > eps) and (it < maxit):
            Hfd = inv(E)@fd(xprev)
            xk = xprev - Hfd
            sk = np.subtract(xk, xprev)
            yk = np.subtract(fd(xk), fd(xprev))

            b1 = (1 / np.dot(yk, sk))*(np.outer(yk, yk))
            sub1b2 = np.outer(sk, sk)
            Esk = E @ (sk)
            sub2b2 = (1 / np.dot(sk, Esk))
            sub3b2 = np.matmul(E, sub1b2)
            sub4b2 = np.matmul(sub3b2, E)
            b2 = sub2b2 * sub4b2
            E1 = np.add(E, b1)
            E = np.subtract(E1, b2)

            xprev = xk
            print(f'\t{xk}')
            print("Log-Values(BFGS): ", math.log10(f(xk)))
            it += 1

        return xk, it


# -

def func(val):
    x,y=val
    return x**2 + y**2


def grad(val):
    x,y=val
    return np.array([2*x, 2*y])


opt = BFGS(f=func,fd=grad,H=np.eye(2), xk=np.array([-2000,30.23]), eps=.01)

opt.work()

# # Foobar2

from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.Chain import Chain
from neb_dynamics.Node3D_TC import Node3D_TC
from neb_dynamics.Inputs import ChainInputs

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/initial_guess_msmep/")

h.output_chain.plot_chain()

from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.Node3D import Node3D
from neb_dynamics.Inputs import NEBInputs, GIInputs

new_step2 = Trajectory([step1_neb.optimized[-1].tdstructure, step2_neb.optimized[-1]]).run_geodesic(nimages=15)

new_step2_chain = Chain.from_traj(new_step2, ChainInputs(k=0.1, delta_k=0.01, step_size=3))

neb = NEB(initial_chain=new_step2_chain, parameters=NEBInputs(v=1,tol=0.001))

neb.optimize_chain()

neb.initial_chain[0].tdstructure.to_xyz("/home/jdep/fuckwittig.xyz")

neb.chain_trajectory[-1].to_trajectory()

leaves = h.ordered_leaves

step1_neb = leaves[0].data

step2_neb = leaves[1].data

from neb_dynamics.constants import BOHR_TO_ANGSTROMS


def get_chain_at_cutoff(chain_traj, cut=0.01):
    beep = None
    for chain in chain_traj:
        if chain.get_maximum_gperp() <= cut:
            beep = chain
            break
    return beep


s1_gi = step1_neb.initial_chain.to_trajectory()

s1_opt = step1_neb.optimized.to_trajectory()

s1_003 = get_chain_at_cutoff(step1_neb.chain_trajectory, cut=0.003*BOHR_TO_ANGSTROMS).to_trajectory()

s1_01 = get_chain_at_cutoff(step1_neb.chain_trajectory, cut=0.01*BOHR_TO_ANGSTROMS).to_trajectory()

s1_0065 = get_chain_at_cutoff(step1_neb.chain_trajectory, cut=0.0065*BOHR_TO_ANGSTROMS).to_trajectory()

ref = step1_neb.initial_chain[0].tdstructure

ref.tc_model_method = 'b3lyp'
ref.tc_model_basis = 'def2-svp'
ref.tc_kwds = {'reference':'uks'}

s1_gi.update_tc_parameters(ref)

s1_gi_chain = Chain.from_traj(s1_gi, ChainInputs(node_class=Node3D_TC))

s1_gi_chain.get_maximum_gperp()

s1_01.update_tc_parameters(ref)

s1_003.update_tc_parameters(ref)

s1_0065.update_tc_parameters(ref)

s1_0065_chain = Chain.from_traj(s1_0065, ChainInputs(node_class=Node3D_TC))

s1_0065_chain.get_maximum_gperp()

s1_01_chain = Chain.from_traj(s1_01, ChainInputs(node_class=Node3D_TC))

s1_01_chain.get_maximum_gperp()

s1_003_chain = Chain.from_traj(s1_003, ChainInputs(node_class=Node3D_TC))

s1_003_chain.get_maximum_gperp()

s1_003_chain.get_maximum_gperp()

s1_opt_chain = Chain.from_traj(s1_opt, ChainInputs(node_class=Node3D_TC))

s1_opt_chain.get_maximum_gperp()

labels =['gi','0.01','0.0065', '0.003','0.001']

import matplotlib.pyplot as plt

plt.plot(labels, [s1_gi_chain.get_maximum_gperp(),s1_01_chain.get_maximum_gperp(),s1_0065_chain.get_maximum_gperp(), s1_003_chain.get_maximum_gperp(),s1_opt_chain.get_maximum_gperp()],'o-')

start, end = s1_opt[0], s1_opt[-1]
start.update_tc_parameters(ref)
end.update_tc_parameters(ref)

start_opt = start.tc_geom_optimization()

end_opt = end.tc_geom_optimization()


