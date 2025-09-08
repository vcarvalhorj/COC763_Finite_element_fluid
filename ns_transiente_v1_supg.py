"""
Transient incompressible Navier-Stokes (lid-driven cavity)
FEniCS Legacy v2019.1.0 — Backward Euler in time

Baseado no código estático do repositório do Gabriel Barros, adaptado para solver transiente.

- Esquema temporal: Backward Euler
- SUPG opcional (aplicado ao termo convectivo do momento)
- Salvamento periódico de arquivos PVD

"""
from __future__ import print_function


import os
import random


from fenics import (
    MPI,
    CellDiameter,
    Constant,
    DirichletBC,
    File,
    FiniteElement,
    Function,
    FunctionSpace,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    SubDomain,
    TestFunctions,
    TrialFunction,
    UnitSquareMesh,
    UserExpression,
    VectorElement,
    action,
    derivative,
    div,
    dot,
    dx,
    grad,
    inner,
    nabla_grad,
    near,
    parameters,
    split,
    sqrt,
    Point,
    )
import numpy as np

# ===================== Parâmetros globais =====================
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 3

# Controle de estabilização
ENABLE_SUPG = True  # mantém SUPG (momento) como no código base

# Reynolds e malha
Re = 100
N = 40  # malha um pouco mais refinada para o transiente (ajuste à vontade)
nu = Constant(1.0 / Re)

# Tempo
T = 20.0           # tempo final (unidades adimensionais)
dt = 0.01          # passo de tempo (ajuste para estabilidade/tempo de execução)
Dt = Constant(dt)  # constante UFL
SAVE_EVERY = 10    # salva a cada SAVE_EVERY passos

# Força de corpo
f = Constant((0.0, 0.0))


# ===================== Condições iniciais =====================
class InitialConditions(UserExpression):
    """Condição inicial suave/aleatória para u e p."""

    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 0.02 * (0.5 - random.random())  # u
        values[1] = 0.0                             # v
        values[2] = 0.0                             # p

    def value_shape(self):
        return (3,)


# ===================== BCs da cavidade =====================

def cavity_boundary_condition(W):
    class Lid(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 1.0)

    class Walls(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (
                near(x[0], 0.0) or near(x[0], 1.0) or near(x[1], 0.0)
            )

    def zero_point(x):
        return near(x[0], 0.0) and near(x[1], 0.0)

    g_lid = Constant((1.0, 0.0))   # velocidade do tampo
    g_walls = Constant((0.0, 0.0))
    g_p = Constant(0.0)

    bc_lid = DirichletBC(W.sub(0), g_lid, Lid())
    bc_walls = DirichletBC(W.sub(0), g_walls, Walls())
    bc_p0 = DirichletBC(W.sub(1), g_p, zero_point, "pointwise")

    return [bc_walls, bc_lid, bc_p0]


# ===================== Malha e espaços =====================
mesh = UnitSquareMesh(N, N)
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)  # velocidade
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # pressão
TH = P2 * P1
W = FunctionSpace(mesh, TH)

# Funções de teste e incógnita
vp = TrialFunction(W)
(w, q) = TestFunctions(W)

# Funções de solução (atual e anterior)
vp_ = Function(W)   # solução no tempo n+1
v_, p_ = split(vp_)

vp_n = Function(W)  # solução no tempo n
v_n, p_n = split(vp_n)

# Inicializa com campo suave
vp_n.interpolate(InitialConditions())

# BCs
bcs = cavity_boundary_condition(W)


# ===================== Formulação variacional (Backward Euler) =====================
# Equações (forma forte):
# (v^{n+1} - v^n)/dt + (v^{n+1} · ∇) v^{n+1} - ν Δ v^{n+1} + ∇p^{n+1} = f^{n+1}
# ∇ · v^{n+1} = 0

# Termo transiente
F_time = inner((v_ - v_n) / Dt, w) * dx

# Convecção + difusão + pressão/incompressibilidade
F_mom = inner(dot(v_, nabla_grad(v_)), w) * dx + nu * inner(grad(v_), grad(w)) * dx - inner(p_, div(w)) * dx
F_cont = inner(q, div(v_)) * dx
F_body = -inner(f, w) * dx

F = F_time + F_mom + F_cont + F_body

# ===================== SUPG (opcional, momento) =====================
if ENABLE_SUPG:
    h = CellDiameter(mesh)
    vnorm = sqrt(dot(v_, v_)) + Constant(1e-12)  # evita divisão por zero
    # Resíduo forte do momento (em v_{n+1})
    Rm = (v_ - v_n) / Dt + dot(v_, nabla_grad(v_)) - nu * div(grad(v_)) + grad(p_) - f
    # tau clássico para NS (ver código base)
    tau = 0.5*((2.0 * vnorm / h) ** 2 + 9.0 * (4.0 * nu / (h * 2)) ** 2) ** (-0.5)
    F_supg = tau * inner(Rm, dot(v_, nabla_grad(w))) * dx
    F += F_supg

# Funcional e Jacobiano para Newton
F1 = F                           #F1 = action(F, vp_)
J = derivative(F1, vp_, vp)

problem = NonlinearVariationalProblem(F1, vp_, bcs, J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm["nonlinear_solver"] = "newton"
prm["newton_solver"]["absolute_tolerance"] = 1e-7
prm["newton_solver"]["relative_tolerance"] = 1e-7
prm["newton_solver"]["convergence_criterion"] = "incremental"
prm["newton_solver"]["maximum_iterations"] = 25
prm["newton_solver"]["relaxation_parameter"] = 0.9
prm["newton_solver"]["linear_solver"] = "direct" # 'direct' usa UMFPACK/MUMPS, conforme disponível     # prm["newton_solver"]["linear_solver"] = "mumps"  # se disponível; caso contrário "direct"
prm["newton_solver"]["error_on_nonconvergence"] = False

# ===================== Saída =====================
os.makedirs("results", exist_ok=True)
out_u = File("results_TF1_SUPG/velocity_transient.pvd", "compressed")
out_p = File("results_TF1_SUPG/pressure_transient.pvd", "compressed")


# ===================== Loop no tempo =====================

t = 0.0
step = 0
print("\n>>> Iniciando integração no tempo (Backward Euler) …")
print("T = {:.3f}, dt = {:.4f}, passos = {}".format(T, dt, int(T / dt)))

while t < T - 1e-12:
    t += dt
    step += 1

    # chute inicial para o Newton em t^{n+1}
    vp_.assign(vp_n)

    # Resolve em t^{n+1}
    converged, n_it = solver.solve()

    if MPI.rank(MPI.comm_world) == 0:
        status = "OK" if converged else "WARN"
        print("t = {:.4f} | Newton iters = {:2d} | {}".format(t, n_it, status))

    # Salva periodicamente
    if step % SAVE_EVERY == 0 or t >= T - 1e-12:
        v_sol, p_sol = vp_.split(True)
        out_u << v_sol
        out_p << p_sol

    # Atualiza solução anterior
    vp_n.assign(vp_)

# ======= Pós-processamento: perfis de validação (Ghia et al.) =======
# Amostra u(x=0.5, y) e v(x, y=0.5) ao final da simulação
try:
    v_last, p_last = vp_.split(True)
    ny = 201
    nx = 201
    y_vals = np.linspace(0.0, 1.0, ny)
    x_vals = np.linspace(0.0, 1.0, nx)


    # deslocamento minúsculo para evitar avaliar exatamente na fronteira
    eps = 1e-10
    x_mid = 0.5
    y_mid = 0.5


    ux_center = [] # (y, u_x)
    for y in y_vals:
        yy = min(max(y, 0.0 + eps), 1.0 - eps)
        uvec = v_last(Point(x_mid, yy))
        ux_center.append([y, uvec[0]])


    vy_center = [] # (x, v_y)
    for x in x_vals:
        xx = min(max(x, 0.0 + eps), 1.0 - eps)
        uvec = v_last(Point(xx, y_mid))
        vy_center.append([x, uvec[1]])

    np.savetxt("results_TF1_SUPG/ghia_u_xmid_supg.csv", np.array(ux_center), delimiter=",", header="y,u(x=0.5,y)", comments="")
    np.savetxt("results_TF1_SUPG/ghia_v_ymid_supg.csv", np.array(vy_center), delimiter=",", header="x,v(x,y=0.5)", comments="")
    if MPI.rank(MPI.comm_world) == 0:
        print("Perfis exportados: results/ghia_u_xmid_supg.csv e results/ghia_v_ymid_supg.csv")
except Exception as e:
    if MPI.rank(MPI.comm_world) == 0:
        print("Aviso: falha ao exportar perfis de validação:", e)

print(">>> Finalizado.")
