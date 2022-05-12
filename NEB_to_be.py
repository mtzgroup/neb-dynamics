# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt


# # Define potential

def coulomb(r, d, r0, alpha):
    return (d/2)*((3/2)*np.exp(-2*alpha*(r - r0)) - np.exp(-alpha*(r-r0)))
coulomb(d=4.746, r=1, r0=0.742, alpha=1.942)


def exchange(r, d, r0, alpha):
    return (d/4)*(np.exp(-2*alpha*(r - r0)) - 6*np.exp(-alpha*(r-r0)))
exchange(d=4.746, r=1, r0=0.742, alpha=1.942)

# +
# plt.plot([coulomb(d=4.746, r=x, r0=0.742, alpha=1.942) for x in list(range(10))])

# +


def potential(r_ab, r_bc, a=0.05, b=0.30, c=0.05, d_ab=4.746, 
                d_bc=4.746, d_ac=3.445, r0=0.742, alpha=1.942):
    Q_AB = coulomb(r=r_ab, d=d_ab, r0=r0, alpha=alpha)
    Q_BC = coulomb(r=r_bc, d=d_bc, r0=r0, alpha=alpha)
    Q_AC = coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)
    
    J_AB = exchange(r=r_ab, d=d_ab, r0=r0, alpha=alpha)
    J_BC = exchange(r=r_bc, d=d_bc, r0=r0, alpha=alpha)
    J_AC = exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)
    
    result_Qs = (Q_AB/(1 + a)) + (Q_BC/(1+b)) + (Q_AC/(1+c)) 
    result_Js_1 = ((J_AB**2) / ((1+a)**2)) + ((J_BC**2) / ((1+b)**2)) + ((J_AC**2) / ((1+c)**2))
    result_Js_2 = ((J_AB*J_BC)/((1+a)*(1+b))) + ((J_AC*J_BC)/((1+c)*(1+b)))+ ((J_AB*J_AC)/((1+a)*(1+c)))
    result_Js = result_Js_1 - result_Js_2
    
    
    result = result_Qs - (result_Js)**(1/2)
    return result
potential(1,1)


# -

def toy_potential(x, y, height=1):
    result = -x**2 + -y**2 + 10
    return np.where(result<0, 0, result)


def toy_grad(x, y ):
    if toy_potential(x, y)==0: return [0,0]
    dx = -2*x 
    dy = -2*y
    
    return dx, dy


def toy_potential_2(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 -7)**2


def toy_grad_2(x, y):
    dx = 2*(x**2 + y - 11)*(2*x) + 2*(x + y**2 - 7)
    dy = 2*(x**2 + y - 11) + 2*(x+y**2 - 7)*(2*y)
    return dx, dy



def spring_grad(x, y, neighs, k=0.1, ideal_distance=0.5, en_func=toy_potential, grad_func=toy_grad):
    
    # pe_grad = 0

    
    
    if len(neighs)==1:
        vec_tan_path = neighs[0] - np.array([x,y])
        
        
    elif len(neighs)==2:
        vec_tan_path = np.array(neighs[1]) - np.array(neighs[0])
    else: 
        raise ValueError("Wtf are you doing.")
        
    unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)
    # print(f"{unit_tan_path=}")
    
    
    
    pe_grad = grad_func(x, y)
    pe_grad_nudged_const = np.dot(pe_grad, unit_tan_path)
    pe_grad_nudged = pe_grad - pe_grad_nudged_const*unit_tan_path
    
    grads_neighs = []
    for neigh in neighs:
        neigh_x, neigh_y = neigh
        dist_x = np.abs(neigh_x - x)
        dist_y = np.abs(neigh_y - y)
        
        force_x = -k*(dist_x - ideal_distance)
        force_y = -k*(dist_y - ideal_distance)
        # print(f"\t{force_x=} {force_y=}")
        
        
        if (neigh_x > x): 
            force_x*= -1
        
        if  (neigh_y > y): 
            force_y*= -1

        
        force_spring = np.array([force_x, force_y])
        force_spring_nudged_const = np.dot(force_spring, unit_tan_path**2)
        force_spring_nudged = force_spring - force_spring_nudged_const*unit_tan_path
        
        
        grads_neighs.append(force_spring_nudged)
    
    # print(f"\t{grads_neighs=}")
        
    tot_grads_neighs = np.sum(grads_neighs, axis=0)
    
    
    
    
    
    # print(f"{tot_grads_neighs}")
    return tot_grads_neighs - pe_grad_nudged

# x = np.linspace(start=0,stop=4, num=10)
# y = x.reshape(-1,1)
# h = potential(x, y, d_ac=1)
# cs = plt.contourf(h)


def dist(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def update_points(chain, grad, dr):
    new_chain = []
    for point in chain:
        p_current = point 

    

        p_x, p_y = p_current
        grad_x, grad_y = grad(p_x, p_y)
        p_new = (p_x + (-grad_x*dr), p_y + (-grad_y*dr))



        p_current = p_new
        new_chain.append(p_current)
        
    return new_chain


def update_points_spring(chain, dr,  en_func, grad_func, k=1, ideal_dist=0.5):
    new_chain = [chain[0]]

    
    for i in range(1, len(chain)-1):

        
        point = chain[i]
        p_current = point 

    

        p_x, p_y = p_current
        if i==0:
            neighs= [chain[1]]
        elif i==len(chain)-1:
            neighs=[chain[-2]]
            
        else:
            neighs=[chain[i-1], chain[i+1]]
        grad_x, grad_y = spring_grad(p_x, p_y, neighs=neighs, k=k, ideal_distance=ideal_dist, grad_func=grad_func, en_func=en_func)
        # print(f"{grad_x=} {grad_y=}")
        p_new = (p_x + (grad_x*dr), p_y + (grad_y*dr))



        p_current = p_new
        new_chain.append(p_current)
    new_chain.append(chain[-1])
        
    return new_chain

# ### NB:
# For top left to bottom right, k=25, idealdist=10

# + tags=[]
np.random.seed(1)
fs = 14
# vars for sim
nsteps = 2000
dr=.01
nimages = 20

en_func = toy_potential_2
grad_func = toy_grad_2


# set up plot for potential
min_val = -4
max_val=4
num=10
fig=10
f,ax = plt.subplots(figsize=(1.18*fig, fig))
x = np.linspace(start=min_val,stop=max_val, num=num)
y = x.reshape(-1,1)


h = en_func(x, y)
cs = plt.contourf(x, x, h)
cbar = f.colorbar(cs)




# # set up points
# chain = np.sort([np.random.uniform(-1, 1, size=2) for n in range(nimages)])
# chain = np.linspace((min_val, max_val), (max_val, min_val), nimages)
chain = np.linspace((-3.7933036307483574, -3.103697226077475), (3, 2), nimages)
# chain = [(-2,-.1),(0,2),(2,.1)]
print(chain)


plt.plot([(point[0]) for point in chain],[(point[1]) for point in chain], '^--',c='white',  label='original')



# dynamics!
chain_current = chain.copy()
print(f"{chain_current=}\n")
for step in range(nsteps):
    new_chain = update_points_spring(chain_current, dr, k=1, ideal_dist=.1, en_func=en_func, grad_func=grad_func)
    # [plt.scatter(point[0], point[1], c='white') for point in chain_current]
    chain_current = new_chain
plt.plot([point[0] for point in chain_current],[point[1] for point in chain_current], 'o--', c='white', label='final path (NEB)')
print(f"final chain: {chain_current}")
for i in range(len(chain_current)-1):
    print(f"dist {i}-{i+1}: {dist(chain_current[i], chain_current[i+1])}")
plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()



# +
# set up plot for potential
min_val = -4
max_val=4
num=10
fig=10
f,ax = plt.subplots(figsize=(1.18*fig, fig))
x = np.linspace(start=min_val,stop=max_val, num=num)
y = x.reshape(-1,1)


h = en_func(x, y)
cs = plt.contourf(x, x, h)
cbar = f.colorbar(cs)
points_x = [point[0] for point in chain_current]
points_y = [point[1] for point in chain_current]
plt.scatter(points_x, points_y, c='white')
print(chain_current)
plt.show()
# -


