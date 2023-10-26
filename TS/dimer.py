# @dataclass
# class Dimer:
#     initial_node: Node
#     delta_r: float
#     step_size: float
#     d_theta: float
#     optimized_dimer = None
#     en_thre: float = 1e-7


#     @property
#     def ts_node(self):
#         if self.optimized_dimer:
#             final_unit_dir = self.get_unit_dir(self.optimized_dimer)
#             r1, r2 = self.optimized_dimer
#             ts = self.initial_node.copy()
#             ts_coords = r1.coords + self.delta_r*final_unit_dir
#             ts = ts.update_coords(ts_coords)


#             return ts


#     def make_initial_dimer(self):
#         random_vec = np.random.rand(*self.initial_node.coords.shape)
#         random_unit_vec = random_vec / np.linalg.norm(random_vec)

#         r1_coords = self.initial_node.coords - self.delta_r*random_unit_vec
#         r2_coords = self.initial_node.coords + self.delta_r*random_unit_vec


#         r1 = self.initial_node.copy()
#         r1 = r1.update_coords(r1_coords)

#         r2 = self.initial_node.copy()
#         r2 = r2.update_coords(r2_coords)

#         dimer = np.array([r1, r2])

#         return dimer

#     def get_dimer_energy(self, dimer):
#         r1,r2 = dimer

#         return r1.energy + r2.energy


#     def force_func(self, node: Node):
#         return - node.gradient

#     def force_perp(self, r_vec: Node, unit_dir: np.array):
#         force_r_vec = - r_vec.gradient
#         return force_r_vec - r_vec.dot_function(force_r_vec, unit_dir)*unit_dir

#     def get_unit_dir(self, dimer):
#         r1, r2 = dimer
#         return (r2.coords - r1.coords)/np.linalg.norm(r2.coords-r1.coords)

#     def get_dimer_force_perp(self, dimer):
#         r1, r2 = dimer
#         unit_dir= self.get_unit_dir(dimer)

#         f_r1 = self.force_perp(r1, unit_dir=unit_dir)
#         f_r2 = self.force_perp(r2, unit_dir=unit_dir)
#         return f_r2 - f_r1


#     def _attempt_rot_step(self, dimer: np.array,theta_rot, t, unit_dir):

#         # update dimer endpoint
#         _, r2 = dimer
#         r2_prime_coords = r2.coords + (unit_dir*np.cos(t) + theta_rot*np.sin(t))*self.delta_r
#         r2_prime = self.initial_node.copy()
#         r2_prime = r2_prime.update_coords(r2_prime_coords)

#         # calculate new unit direction
#         midpoint_coords = r2.coords - unit_dir*self.delta_r
#         new_dir = (r2_prime_coords - midpoint_coords)
#         new_unit_dir = new_dir / np.linalg.norm(new_dir)

#         # remake dimer using new unit direciton
#         r1_prime_coords = r2_prime_coords - 2*self.delta_r*new_unit_dir
#         r1_prime = self.initial_node.copy()
#         r1_prime = r1_prime.update_coords(r1_prime_coords)

#         new_dimer = np.array([r1_prime, r2_prime])
#         new_grad = self.get_dimer_force_perp(new_dimer)

#         # en_struct_prime = np.linalg.norm(new_grad)
#         en_struct_prime = np.dot(new_grad, theta_rot)
#         # en_struct_prime = self.get_dimer_energy(new_dimer)

#         return en_struct_prime, t

#     def _rotate_img(self, r_vec: Node, unit_dir: np.array, theta_rot: float, dimer: np.array,dt,
#     alpha=0.000001, beta=0.5):

#         # max_steps = 10
#         # count = 0


#         # grad = self.get_dimer_force_perp(dimer)
#         # # en_struct = np.linalg.norm(grad)
#         # en_struct = np.dot(grad, theta_rot)
#         # # en_struct = self.get_dimer_energy(dimer)

#         # en_struct_prime, t = self._attempt_rot_step(dimer=dimer, t=self.d_theta, unit_dir=unit_dir, theta_rot=theta_rot)

#         # condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

#         # while condition and count < max_steps:
#         #     t *= beta
#         #     count += 1
#         #     en_struct = en_struct_prime

#         #     en_struct_prime, t = self._attempt_rot_step(dimer=dimer, t=self.d_theta, unit_dir=unit_dir, theta_rot=theta_rot)

#         #     condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

#         # print(f"\t\t\t{t=} {count=} || force: {np.linalg.norm(grad)}")
#         # sys.stdout.flush()


#         # return r_vec.coords + (unit_dir*np.cos(t) + theta_rot*np.sin(t))*self.delta_r
#         return r_vec.coords + (unit_dir*np.cos(dt) + theta_rot*np.sin(dt))*self.delta_r


#     def _rotate_dimer(self, dimer):
#         """
#         from this paper https://aip.scitation.org/doi/pdf/10.1063/1.480097
#         """
#         _, r2 = dimer
#         unit_dir = self.get_unit_dir(dimer)
#         midpoint_coords = r2.coords - unit_dir*self.delta_r
#         midpoint = self.initial_node.copy()
#         midpoint = midpoint.update_coords(midpoint_coords)

#         dimer_force_perp = self.get_dimer_force_perp(dimer)
#         theta_rot = dimer_force_perp / np.linalg.norm(dimer_force_perp)


#         r2_prime_coords = self._rotate_img(r_vec=midpoint, unit_dir=unit_dir, theta_rot=theta_rot, dimer=dimer, dt=self.d_theta)
#         r2_prime = self.initial_node.copy()
#         r2_prime = r2_prime.update_coords(r2_prime_coords)

#         new_dir = (r2_prime_coords - midpoint_coords)
#         new_unit_dir = new_dir / np.linalg.norm(new_dir)

#         r1_prime_coords = r2_prime_coords - 2*self.delta_r*new_unit_dir
#         r1_prime = self.initial_node.copy()
#         r1_prime = r1_prime.update_coords(r1_prime_coords)


#         dimer_prime = np.array([r1_prime, r2_prime])
#         dimer_prime_force_perp = self.get_dimer_force_perp(dimer_prime)
#         theta_rot_prime = dimer_prime_force_perp / np.linalg.norm(dimer_prime_force_perp)
#         f_prime = (np.dot(dimer_prime_force_perp, theta_rot_prime) - np.dot(dimer_force_perp, theta_rot))/self.d_theta

#         # optimal_d_theta = (np.dot(dimer_force_perp, theta_rot) + np.dot(dimer_prime_force_perp, theta_rot_prime))/(-2*f_prime)


#         r1_p, r2_p = dimer_prime
#         f_r1_p = self.force_perp(r1_p, unit_dir=unit_dir)
#         f_r2_p = self.force_perp(r2_p, unit_dir=unit_dir)

#         r1, r2 = dimer
#         f_r1 = self.force_perp(r1, unit_dir=unit_dir)
#         f_r2 = self.force_perp(r2, unit_dir=unit_dir)


#         f_val = 0.5*(np.dot(f_r2_p - f_r1_p, theta_rot_prime) + np.dot(f_r2 - f_r1, theta_rot))
#         f_val_prime = (1/self.d_theta)*(np.dot(f_r2_p - f_r1_p, theta_rot_prime) - np.dot(f_r2 - f_r1, theta_rot))

#         optimal_d_theta = 0.5*np.arctan(2*f_val/f_val_prime) - self.d_theta/2
#         print(f"\t\t{optimal_d_theta=}")
#         r2_final_coords = self._rotate_img(r_vec=midpoint, unit_dir=unit_dir, theta_rot=theta_rot, dimer=dimer, dt=optimal_d_theta)
#         r2_final = self.initial_node.copy()
#         r2_final = r2_final.update_coords(r2_final_coords)

#         final_dir = (r2_final_coords - midpoint_coords)
#         final_unit_dir = final_dir / np.linalg.norm(final_dir)

#         r1_final_coords = r2_final_coords - 2*self.delta_r*final_unit_dir
#         r1_final = self.initial_node.copy()
#         r1_final = r1_final.update_coords(r1_final_coords)

#         dimer_final = np.array([r1_final, r2_final])


#         return dimer_final

#     def _translate_dimer(self, dimer):
#         dimer_0 = dimer

#         r1,r2 = dimer_0
#         force = self.get_climb_force(dimer_0)


#         r2_prime_coords = r2.coords + self.step_size*force
#         r2_prime = self.initial_node.copy()
#         r2_prime = r2_prime.update_coords(r2_prime_coords)


#         r1_prime_coords = r1.coords + self.step_size*force
#         r1_prime = self.initial_node.copy()
#         r1_prime = r1_prime.update_coords(r1_prime_coords)


#         dimer_1 = (r1_prime, r2_prime)

#         return dimer_1


#     def fully_update_dimer(self, dimer):
#         dimer_0 = dimer
#         en_0 = self.get_dimer_energy(dimer_0)

#         dimer_0_prime = self._rotate_dimer(dimer_0)
#         dimer_1 = self._translate_dimer(dimer_0_prime)
#         en_1 = self.get_dimer_energy(dimer_1)

#         n_counts = 0
#         while np.abs(en_1 - en_0) > self.en_thre and n_counts < 100000:
#             # print(f"{n_counts=} // |∆E|: {np.abs(en_1 - en_0)}")
#             dimer_0 = dimer_1
#             en_0 = self.get_dimer_energy(dimer_0)

#             dimer_0_prime = self._rotate_dimer(dimer_0)
#             dimer_1 = self._translate_dimer(dimer_0_prime)

#             en_1 = self.get_dimer_energy(dimer_1)
#             n_counts+=1

#         if np.abs(en_1 - en_0) <= self.en_thre: print(f"Optimization converged in {n_counts} steps!")
#         else: print(f"Optimization did not converge. Final |∆E|: {np.abs(en_1 - en_0)}")
#         return dimer_1


#     def get_climb_force(self, dimer):
#         r1, r2 = dimer
#         unit_path = self.get_unit_dir(dimer)


#         f_r1 = self.force_func(r1)
#         f_r2 = self.force_func(r2)
#         F_R = f_r1 + f_r2


#         f_parallel_r1 = self.initial_node.dot_function(f_r1, unit_path)*unit_path
#         f_parallel_r2 = self.initial_node.dot_function(f_r2, unit_path)*unit_path
#         F_Par = f_parallel_r1 + f_parallel_r2


#         return F_R - 2*F_Par


#     def find_ts(self):
#         dimer = self.make_initial_dimer()
#         opt_dimer = self.fully_update_dimer(dimer)
#         self.optimized_dimer = opt_dimer
