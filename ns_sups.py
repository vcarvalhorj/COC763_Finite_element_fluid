"""
Transient incompressible Navier-Stokes (lid-driven cavity) — SU/PSPG
FEniCS Legacy v2019.1.0  |  Backward Euler no tempo

- SUPG no momento  (opcional)
- PSPG na continuidade (acoplamento u-p)  [“SUPS” = SU/PSPG]
- Malha UnitSquare, Taylor-Hood P2-P1, BC tampo deslizante
- Parâmetros no topo (T, dt, SAVE_EVERY, Re, N)
"""

from __future__ import print_function
import os, random
import numpy as np

from fenics import (
    MPI, CellDiameter, Constant, DirichletBC, File, FiniteElement,
    Function, FunctionSpace, NonlinearVariationalProblem, NonlinearVariationalSolver,
    SubDomain, TestFunctions, TrialFunction, UnitSquareMesh, UserExpression, VectorElement,
    derivative, div, dot, dx, grad, inner, nabla_grad, near, parameters, split, sqrt, Point
)

# ===================== Parâmetros globais =====================
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 3

# Estabilizações
ENABLE_SUPG  = True   # SUPG no momento
ENABLE_PSPG  = True   # PSPG na continuidade (acoplamento u–p)
ENABLE_LSIC  = False  # (opcional) termo de compressibilidade mínima (grad-div fraco)

TAU_SCALE = 1.0       # fator para suavizar tau (ajuda Newton a convergir); (1.0 para ser mais agressivo)

# Reynolds e malha
Re = 100
N  = 40  #efeito da malha (antes era 40)
nu = Constant(1.0 / Re)

# Tempo
T  = 10.0
dt = 0.01     #0.005
Dt = Constant(dt)
SAVE_EVERY = 10

# Força de corpo
f = Constant((0.0, 0.0))

# ===================== Condições iniciais =====================
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.02*(0.5 - random.random())  # u
        values[1] = 0.0                            # v
        values[2] = 0.0                            # p
    def value_shape(self): return (3,)

# ===================== BCs da cavidade =====================
def cavity_boundary_condition(W):
    class Lid(SubDomain):
        def inside(self, x, on_boundary): return on_boundary and near(x[1], 1.0)
    class Walls(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (near(x[0],0.0) or near(x[0],1.0) or near(x[1],0.0))
    def zero_point(x): return near(x[0], 0.0) and near(x[1], 0.0)
    g_lid, g_walls, g_p = Constant((1.0,0.0)), Constant((0.0,0.0)), Constant(0.0)
    bc_lid   = DirichletBC(W.sub(0), g_lid,   Lid())
    bc_walls = DirichletBC(W.sub(0), g_walls, Walls())
    bc_p0    = DirichletBC(W.sub(1), g_p, zero_point, "pointwise")
    return [bc_walls, bc_lid, bc_p0]

# ===================== Malha e espaços =====================
mesh = UnitSquareMesh(N, N)
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)  # velocidade
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # pressão
W  = FunctionSpace(mesh, P2 * P1)

# Testes e incógnitas
vp = TrialFunction(W)
(w, q) = TestFunctions(W)

# Soluções (n+1 e n)
vp_ = Function(W);  v_, p_ = split(vp_)
vpn = Function(W);  v_n, p_n = split(vpn)

# Inicializa
vpn.interpolate(InitialConditions())
bcs = cavity_boundary_condition(W)

# ===================== Backward Euler =====================
F_time = inner((v_ - v_n)/Dt, w) * dx
F_mom  = inner(dot(v_, nabla_grad(v_)), w)*dx + nu*inner(grad(v_), grad(w))*dx - inner(p_, div(w))*dx
F_cont = inner(q, div(v_)) * dx
F_body = -inner(f, w) * dx
F = F_time + F_mom + F_cont + F_body

# ===================== Estabilizações =====================
# Resíduo do momento (em n+1)
h     = CellDiameter(mesh)
vnorm = sqrt(dot(v_, v_)) + Constant(1e-12)

# tau com termo transiente (robusto p/ dt menores): [(2/dt)^2 + (2|u|/h)^2 + C_nu*(nu/h^2)^2]^{-1/2}
# Mesmo estilo da base (ns_transiente_v1_supg.py) e foi adicionado (2/Dt)^2. O fator TAU_SCALE ajuda a amortecer.
Cnu = 9.0 * (4.0 / 2.0)**2  # segue a forma do código base
tau_raw = ((2.0/Dt)**2 + (2.0*vnorm/h)**2 + Cnu*(nu/(h*h))**2) ** (-0.5)
tau = TAU_SCALE * tau_raw

Rm = (v_ - v_n)/Dt + dot(v_, nabla_grad(v_)) - nu*div(grad(v_)) + grad(p_) - f
Rc = div(v_)  # só se for usar LSIC

# SUPG (momento)
if ENABLE_SUPG:
    F += inner(tau*Rm, dot(v_, nabla_grad(w))) * dx

# PSPG (acoplamento u–p): testa Rm com grad(q)  -> estabiliza pressão
if ENABLE_PSPG:
    F += inner(tau*Rm, grad(q)) * dx

# (Opcional) LSIC / grad-div fraco: melhora conservação de massa
if ENABLE_LSIC:
    # escolha clássica: tau_c ~ h^2 / (tau_m * |u|^2 + small) , aqui uso forma simples proporcional a h^2/(2*Dt + nu)
    tau_c = TAU_SCALE * h*h / (2.0/Dt + nu + 1e-12)
    F += inner(tau_c * Rc, div(w)) * dx

# ===================== Solver =====================
F1 = F
J  = derivative(F1, vp_, vp)

problem = NonlinearVariationalProblem(F1, vp_, bcs, J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm["nonlinear_solver"] = "newton"
prm["newton_solver"]["absolute_tolerance"] = 1e-7
prm["newton_solver"]["relative_tolerance"] = 1e-7
prm["newton_solver"]["convergence_criterion"] = "incremental"
prm["newton_solver"]["maximum_iterations"] = 25
prm["newton_solver"]["relaxation_parameter"] = 0.9
prm["newton_solver"]["linear_solver"] = "direct"
prm["newton_solver"]["error_on_nonconvergence"] = False

# ===================== Saída =====================
os.makedirs("results_TF2_SUPS", exist_ok=True)
out_u = File("results_TF2_SUPS/velocity.pvd", "compressed")
out_p = File("results_TF2_SUPS/pressure.pvd", "compressed")

# ===================== Loop no tempo =====================
t, step = 0.0, 0
print("\n>>> Iniciando integração no tempo (Backward Euler + SU/PSPG)…")
print("T = {:.3f}, dt = {:.4f}, passos = {}".format(T, dt, int(T/dt)))

while t < T - 1e-12:
    t += dt; step += 1
    vp_.assign(vpn)  # chute inicial Newton

    converged, n_it = solver.solve()
    if MPI.rank(MPI.comm_world) == 0:
        print("t = {:.4f} | Newton iters = {:2d} | {}".format(t, n_it, "OK" if converged else "WARN"))

    if step % SAVE_EVERY == 0 or t >= T - 1e-12:
        v_sol, p_sol = vp_.split(True)
        out_u << v_sol; out_p << p_sol

    vpn.assign(vp_)

# ======= Pós-processamento: perfis centrais =======
try:
    v_last, p_last = vp_.split(True)
    ny, nx = 201, 201
    y_vals = np.linspace(0.0, 1.0, ny)
    x_vals = np.linspace(0.0, 1.0, nx)
    eps, x_mid, y_mid = 1e-10, 0.5, 0.5
    ux_center, vy_center = [], []
    for y in y_vals:
        uvec = v_last(Point(x_mid, min(max(y,0.0+eps),1.0-eps)))
        ux_center.append([y, uvec[0]])
    for x in x_vals:
        uvec = v_last(Point(min(max(x,0.0+eps),1.0-eps), y_mid))
        vy_center.append([x, uvec[1]])
    np.savetxt("results_TF2_SUPS/ghia_u_xmid.csv", np.array(ux_center), delimiter=",", header="y,u(x=0.5,y)", comments="")
    np.savetxt("results_TF2_SUPS/ghia_v_ymid.csv", np.array(vy_center), delimiter=",", header="x,v(x,y=0.5)", comments="")
    if MPI.rank(MPI.comm_world) == 0:
        print("Perfis exportados: results_TF2_SUPS/ghia_*.csv")
except Exception as e:
    if MPI.rank(MPI.comm_world) == 0:
        print("Aviso: falha ao exportar perfis:", e)

print(">>> Finalizado.")
