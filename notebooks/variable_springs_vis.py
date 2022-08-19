from neb_dynamics.NEB import Chain, Node2D
import numpy as np
import matplotlib.pyplot as plt

# +
arr = np.array([[-3.77928812, -3.28320392],
       [-3.48622958, -2.45275996],
       [-3.29515736, -1.59331875],
       [-3.15605443, -0.7243703 ],
       [-3.04858521,  0.1484558 ],
       [-2.96447671,  1.02309022],
       [-2.89233606,  1.89792329],
       [-2.88436763,  2.77474502],
       [-2.06157496,  3.07518504],
       [-1.18868001,  3.00532544],
       [-0.31687816,  2.92533579],
       [ 0.55270823,  2.82572979],
       [ 1.41709825,  2.68910715],
       [ 2.26568071,  2.47566295],
       [ 3.00002182,  1.99995542]])


arr2 = np.array([[-3.77928812, -3.28320392],
       [-3.43727343, -2.26707267],
       [-3.26879576, -1.45580522],
       [-3.15754305, -0.74291069],
       [-3.07029467, -0.05706221],
       [-2.9998469 ,  0.63068196],
       [-2.93520954,  1.35406575],
       [-2.86964167,  2.19369124],
       [-2.43878091,  3.2134023 ],
       [-1.42777561,  3.02536916],
       [-0.56096579,  2.94941294],
       [ 0.27380051,  2.86245286],
       [ 1.10467218,  2.74496014],
       [ 2.00442638,  2.55539617],
       [ 3.00002182,  1.99995542]])

chain = Chain.from_list_of_coords(k=1,list_of_coords=arr,delta_k=0,step_size=0,node_class=Node2D)
chain2 = Chain.from_list_of_coords(k=1,list_of_coords=arr2,delta_k=0,step_size=0,node_class=Node2D)

# +
arr_climb = np.array([[-3.77928812, -3.28320392],
       [-3.50079526, -2.50932564],
       [-3.31606671, -1.70788463],
       [-3.18001462, -0.89677399],
       [-3.0730237 , -0.08133405],
       [-2.97465049,  0.91309691],
       [-2.89195279,  1.909357  ],
       [-2.89806688,  2.9097563 ],
       [-1.90861274,  3.06319393],
       [-0.91050584,  2.98113049],
       [ 0.08643184,  2.88428102],
       [ 0.85454803,  2.784174  ],
       [ 1.61760039,  2.65025422],
       [ 2.36408873,  2.44268316],
       [ 3.00002182,  1.99995542]])

arr2_climb = np.array([[-3.77928812, -3.28320392],
       [-3.43872377, -2.27355275],
       [-3.27064913, -1.46703888],
       [-3.15986742, -0.76007914],
       [-3.07302575, -0.08135304],
       [-3.00284432,  0.59936748],
       [-2.93870232,  1.31222336],
       [-2.87393321,  2.13476156],
       [-2.68784081,  3.19278774],
       [-1.62366514,  3.04121961],
       [-0.74089311,  2.9663616 ],
       [ 0.0866775 ,  2.8842547 ],
       [ 0.98204188,  2.76414024],
       [ 1.93559473,  2.57287473],
       [ 3.00002182,  1.99995542]])


chain_climb = Chain.from_list_of_coords(k=1,list_of_coords=arr_climb,delta_k=0,step_size=0,node_class=Node2D)
chain2_climb = Chain.from_list_of_coords(k=1,list_of_coords=arr2_climb,delta_k=0,step_size=0,node_class=Node2D)
# -

plt.plot(chain.integrated_path_length, chain.energies,'o--', label='NEB(k=100)', color='orange')
plt.plot(chain_climb.integrated_path_length, chain.energies+40,'^-', label='cNEB(k=100 )', color='orange')
plt.plot(chain2.integrated_path_length, chain2.energies+200,'o--', label='NEB(k=100 / ∆k=50)', color='skyblue')
plt.plot(chain2_climb.integrated_path_length, chain2.energies+240,'o-', label='cNEB(k=100 / ∆k=50)', color='skyblue')
plt.legend(bbox_to_anchor=(1, 1.0))

# +
arr = np.array([[-3.77928812, -3.28320392],
       [-3.31002321, -1.6334969 ],
       [-3.05787304,  0.06291247],
       [-2.9057854 ,  1.77100314],
       [-1.95882024,  3.20035069],
       [-0.26885569,  2.91574879],
       [ 1.42755118,  2.67818277],
       [ 3.00002182,  1.99995542]])

arr2 = np.array([[-3.77928812, -3.28320392],
       [-3.28914444, -1.5348907 ],
       [-3.07157778, -0.06811779],
       [-2.93397699,  1.40811196],
       [-2.64813935,  3.21901272],
       [-0.75081977,  2.95965754],
       [ 1.14935912,  2.72851899],
       [ 3.00002182,  1.99995542]])

arr_climb = np.array([[-3.77928812, -3.28320392],
       [-3.32292024, -1.7048767 ],
       [-3.07300096, -0.08108244],
       [-2.90791058,  1.75122512],
       [-1.73439054,  3.16822122],
       [ 0.08416811,  2.884373  ],
       [ 1.60333038,  2.64489421],
       [ 3.00002182,  1.99995542]])

arr2_climb = np.array([[-3.77928812, -3.28320392],
       [-3.29014786, -1.54079219],
       [-3.07301428, -0.08123443],
       [-2.93481738,  1.39824182],
       [-2.66783701,  3.21111644],
       [-0.76165829,  2.96073839],
       [ 1.14607643,  2.72907449],
       [ 3.00002182,  1.99995542]])

chain = Chain.from_list_of_coords(k=1,list_of_coords=arr,delta_k=0,step_size=0,node_class=Node2D)
chain2 = Chain.from_list_of_coords(k=1,list_of_coords=arr2,delta_k=0,step_size=0,node_class=Node2D)
chain_climb = Chain.from_list_of_coords(k=1,list_of_coords=arr_climb,delta_k=0,step_size=0,node_class=Node2D)
chain2_climb = Chain.from_list_of_coords(k=1,list_of_coords=arr2_climb,delta_k=0,step_size=0,node_class=Node2D)

plt.plot(chain.integrated_path_length, chain.energies,'o--', label='NEB(k=100 / ∆k=0)', color='skyblue')
plt.plot(chain2.integrated_path_length, chain2.energies+100,'o-', label='cNEB(k=100 / ∆k=50)', color='skyblue')

plt.plot(chain_climb.integrated_path_length, chain.energies,'^--', label='NEB(k=100 / ∆k=0)', color='orange')
plt.plot(chain2_climb.integrated_path_length, chain2.energies+100,'o-', label='cNEB(k=100 / ∆k=50)', color='orange')

plt.legend()
# -


