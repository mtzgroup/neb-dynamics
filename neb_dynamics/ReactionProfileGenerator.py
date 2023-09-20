from dataclasses import dataclass, field

import numpy as np
from chemcloud.client import CCClient

from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs
from neb_dynamics.Node3D_TC import Node3D_TC
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory


@dataclass
class ReactionProfileGenerator:
    input_chain: Chain
    method: str
    basis: str
    kwds: dict = field(default_factory=dict)
    geom_opt_kwds: dict = field(default_factory= lambda: {"program": "terachem_fe", "maxiter": 300, 
                          "transition": True, 'trust': 0.005})
    _cached_ts: TDStructure = None
    _cached_freqs: list = None
    _cached_nma: list = None
    _cached_alleged_profile: Chain = None
    _cached_pseu_irc_profile: Chain = None
    
    @property
    def transition_state(self):
        if self._cached_ts: 
            return self._cached_ts
        else:
            ts_guess = self.input_chain.get_ts_guess()
            ts_guess.tc_model_method = self.method
            ts_guess.tc_model_basis = self.basis
            ts_guess.tc_geom_opt_kwds = self.geom_opt_kwds
            
            
            inp = ts_guess._prepare_input(method='ts')
            result = self.compute_and_return_results(inp)
            if result.success:
                ts = TDStructure.from_cc_result(result)
                ts.update_tc_parameters(ts_guess) # TODO: fix this so that the params come from cc result
                
                freqs = ts.tc_freq_calculation()
                nma = ts.tc_nma_calculation() # these are sorted from min freq to max freq
                assert sum(np.array(freqs) < 0) == 1, "Not a first order saddle point."
                
                self._cached_ts = ts
                self._cached_freqs = freqs
                self._cached_nma = nma
                return ts
            else:
                print(result.error.error_message)
                return None
    
    @property
    def client(self):
        return CCClient()
    
    def create_pseudo_irc_inputs(self, dr=.1):
        ts_structure = self.transition_state
        
        unstable_mode = self._cached_nma[0]
        direction = np.array(unstable_mode).reshape(ts_structure.coords.shape)

        td_disp_plus = ts_structure.update_coords(ts_structure.coords + dr*direction)
        td_disp_plus.update_tc_parameters(ts_structure)
        
        td_disp_minus = ts_structure.update_coords(ts_structure.coords - dr*direction)
        td_disp_minus.update_tc_parameters(ts_structure)

        inps = [td_disp_plus._prepare_input("opt"), td_disp_minus._prepare_input("opt")]
        return inps
    
    def create_alleged_profile_inputs(self):
        r = self.input_chain[0].tdstructure
        p = self.input_chain[-1].tdstructure

        for td in [r, p]:
            td.tc_model_basis = self.basis
            td.tc_model_method = self.method

        inp_objs = [r._prepare_input(method='opt'), p._prepare_input(method='opt')]
        inp_opts = self.client.compute_procedure(inp_objs, procedure='geometric')
        return inp_objs

    def compute_and_return_results(self, inps):
        opts = self.client.compute_procedure(inps, procedure='geometric')
        results = opts.get()
        return results
        
    def create_profile(self, results_obj):
        r = TDStructure.from_cc_result(results_obj[0])
        p = TDStructure.from_cc_result(results_obj[1])
        t = Trajectory([r, self.transition_state, p])
        t.update_tc_parameters(self.transition_state)
        chain = Chain.from_traj(t, ChainInputs(node_class=Node3D_TC))
        return chain
        
    
    def create_all_reaction_profiles(self):
        alleged_r_p_inputs = self.create_alleged_profile_inputs()
        pseu_irc_r_p_inputs = self.create_pseudo_irc_inputs()
        all_inps = alleged_r_p_inputs+pseu_irc_r_p_inputs
        
        results = self.compute_and_return_results(all_inps)
        
        if self._cached_alleged_profile:
            alleged_profile = self._cached_alleged_profile
        else:
            alleged_profile = self.create_profile(results[:2])
            self._cached_alleged_profile = alleged_profile
            
        
        if self._cached_pseu_irc_profile:
            pseu_irc_profile = self._cached_pseu_irc_profile
        else:
            pseu_irc_profile = self.create_profile(results[2:])
            self._cached_pseu_irc_profile = pseu_irc_profile
            
        
            
        return alleged_profile, pseu_irc_profile
    
    @staticmethod
    def profiles_are_identical(alleged_profile, pseu_irc_profile):
        arp = alleged_profile
        pirp = pseu_irc_profile
        simply_true = arp[0]._is_connectivity_identical(pirp[0]) and arp[-1]._is_connectivity_identical(pirp[-1])
        flipped_but_true = arp[0]._is_connectivity_identical(pirp[-1]) and arp[-1]._is_connectivity_identical(pirp[0])
        return simply_true or flipped_but_true
    