import numpy as np
def sorry_func_0(inp):
    
    x, y = inp
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

def sorry_func_1(inp): # https://theory.cm.utexas.edu/henkelman/pubs/sheppard11_1769.pdf
    x, y = inp
    A = np.cos(np.pi*x) 
    B = np.cos(np.pi*y) 
    C = np.pi*np.exp(-np.pi*x**2)
    D = (np.exp(-np.pi*(y - 0.8)**2))  + np.exp(-np.pi*(y+0.8)**2)
    return A + B + C*D


def sorry_func_2(inp):
    x, y = inp
    Ax = 1
    Ay = 1
    return -1*(Ax*np.cos(2*np.pi*x) + Ay*np.cos(2*np.pi*y))


def flower_func(inp):
    x, y = inp
    return (1./20.)*(( 1*(x**2 + y**2) - 6*np.sqrt(x**2 + y**2))**2 + 30 ) * -1*np.abs(.4 * np.cos(6  * np.arctan(x/y))+1)


####### -------

def coulomb(r, d, r0, alpha):
    return (d / 2) * ((3 / 2) * np.exp(-2 * alpha * (r - r0)) - np.exp(-alpha * (r - r0)))


coulomb(d=4.746, r=1, r0=0.742, alpha=1.942)


def exchange(r, d, r0, alpha):
    return (d / 4) * (np.exp(-2 * alpha * (r - r0)) - 6 * np.exp(-alpha * (r - r0)))


exchange(d=4.746, r=1, r0=0.742, alpha=1.942)


# +
# plt.plot([coulomb(d=4.746, r=x, r0=0.742, alpha=1.942) for x in list(range(10))])
# -


def sorry_func_3(
    inp,
    a=0.05,
    b=0.30,
    c=0.05,
    d_ab=4.746,
    d_bc=4.746,
    d_ac=3.445,
    r0=0.742,
    alpha=1.942,
):
    r_ab, r_bc = inp

    Q_AB = coulomb(r=r_ab, d=d_ab, r0=r0, alpha=alpha)
    Q_BC = coulomb(r=r_bc, d=d_bc, r0=r0, alpha=alpha)
    Q_AC = coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

    J_AB = exchange(r=r_ab, d=d_ab, r0=r0, alpha=alpha)
    J_BC = exchange(r=r_bc, d=d_bc, r0=r0, alpha=alpha)
    J_AC = exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

    result_Qs = (Q_AB / (1 + a)) + (Q_BC / (1 + b)) + (Q_AC / (1 + c))
    result_Js_1 = ((J_AB**2) / ((1 + a) ** 2)) + ((J_BC**2) / ((1 + b) ** 2)) + ((J_AC**2) / ((1 + c) ** 2))
    result_Js_2 = ((J_AB * J_BC) / ((1 + a) * (1 + b))) + ((J_AC * J_BC) / ((1 + c) * (1 + b))) + ((J_AB * J_AC) / ((1 + a) * (1 + c)))
    result_Js = result_Js_1 - result_Js_2

    result = result_Qs - (result_Js) ** (1 / 2)
    return result


def dQ_dr(d, alpha, r, r0):
    return (d / 2) * ((3 / 2) * (-2 * alpha * np.exp(-2 * alpha * (r - r0))) + alpha * np.exp(-alpha * (r - r0)))


def dJ_dr(d, alpha, r, r0):
    return (d / 4) * (np.exp(-2 * alpha * (r - r0)) * (-2 * alpha) + 6 * alpha * np.exp(-alpha * (r - r0)))


def grad_x(
    inp,
    a=0.05,
    b=0.30,
    c=0.05,
    d_ab=4.746,
    d_bc=4.746,
    d_ac=3.445,
    r0=0.742,
    alpha=1.942,
):
    r_ab, r_bc = inp

    ealpha_x = np.exp(alpha * (r0 - r_ab))
    neg_ealpha_x = np.exp(alpha * (r_ab - r0))
    ealpha_y = np.exp(alpha * (r0 - r_bc))
    neg_ealpha_y = np.exp(alpha * (r_bc - r0))

    e2alpha_x = np.exp(2 * alpha * (r0 - r_ab))
    e2alpha_y = np.exp(2 * alpha * (r0 - r_bc))

    aDenom = 1 / (1 + a)
    bDenom = 1 / (1 + b)

    Qconst = coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)
    Jconst = exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

    cDenom = 1 / (1 + c)

    d = d_ab

    dx = (
        0.25
        * aDenom**2
        * alpha
        * d
        * ealpha_x
        * (
            -2 * (1 + a) * (-1 + 3 * ealpha_x)
            + ((-3 + ealpha_x) * (2 * d * ealpha_x * (-6 + ealpha_x) - (1 + a) * d * ealpha_y * (-6 + ealpha_y) * bDenom - 4 * (1 + a) * Jconst * cDenom))
            / (
                np.sqrt(
                    (
                        ((d**2 * e2alpha_x) * (-6 + ealpha_x) ** 2 * aDenom**2)
                        + (d**2 * e2alpha_y * (-6 + ealpha_y) ** 2) * bDenom**2
                        - d**2 * np.exp(-2 * alpha * (-2 * r0 + r_ab + r_bc)) * (-1 + 6 * neg_ealpha_x) * (-1 + 6 * neg_ealpha_y) * aDenom * bDenom
                    )
                    - 4 * d * ealpha_x * (-6 + ealpha_x) * Jconst * aDenom * cDenom
                    - 4 * d * ealpha_y * (-6 + ealpha_y * Jconst * bDenom * cDenom)
                    + 16 * Jconst**2 * cDenom**2
                )
            )
        )
    )

    return dx


def grad(inp):
    return np.array([grad_x(inp), grad_y(inp)])


def grad_y(
    inp,
    a=0.05,
    b=0.30,
    c=0.05,
    d_ab=4.746,
    d_bc=4.746,
    d_ac=3.445,
    r0=0.742,
    alpha=1.942,
):
    r_ab, r_bc = inp

    ealpha_x = np.exp(alpha * (r0 - r_ab))
    neg_ealpha_x = np.exp(alpha * (r_ab - r0))
    ealpha_y = np.exp(alpha * (r0 - r_bc))
    neg_ealpha_y = np.exp(alpha * (r_bc - r0))

    e2alpha_x = np.exp(2 * alpha * (r0 - r_ab))
    e2alpha_y = np.exp(2 * alpha * (r0 - r_bc))

    aDenom = 1 / (1 + a)
    bDenom = 1 / (1 + b)

    Qconst = coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)
    Jconst = exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

    cDenom = 1 / (1 + c)

    d = d_bc

    dy = (
        0.25
        * bDenom**2
        * alpha
        * d
        * ealpha_y
        * (
            -2 * (1 + b) * (-1 + 3 * ealpha_y)
            + ((-3 + ealpha_y) * (2 * d * ealpha_y * (-6 + ealpha_y) - (1 + b) * d * ealpha_x * (-6 + ealpha_x) * aDenom - 4 * (1 + b) * Jconst * cDenom))
            / (
                np.sqrt(
                    (
                        ((d**2 * e2alpha_x) * (-6 + ealpha_x) ** 2 * aDenom**2)
                        + (d**2 * e2alpha_y * (-6 + ealpha_y) ** 2) * bDenom**2
                        - d**2 * np.exp(-2 * alpha * (-2 * r0 + r_ab + r_bc)) * (-1 + 6 * neg_ealpha_x) * (-1 + 6 * neg_ealpha_y) * aDenom * bDenom
                    )
                    - 4 * d * ealpha_x * (-6 + ealpha_x) * Jconst * aDenom * cDenom
                    - 4 * d * ealpha_y * (-6 + ealpha_y * Jconst * bDenom * cDenom)
                    + 16 * Jconst**2 * cDenom**2
                )
            )
        )
    )

    return dy