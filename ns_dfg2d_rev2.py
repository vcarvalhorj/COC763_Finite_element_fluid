from __future__ import print_function
from dolfin import *
import numpy as np

"""
NS DFG-2D (cilindro em canal) — FEniCS Legacy 2019.1/2019.2

Recursos:
- Leitura de malha .xml
- BCs do benchmark DFG-2D: perfil parabólico na entrada, no-slip nas paredes e cilindro, pressão nula na saída (opção simples)
- Marcha temporal Backward-Euler (implícito)
- Cálculo online de forças de arrasto e sustentação via tensor de tensões
- Cálculo dos coeficientes Cd, Cl e métricas finais: \bar{C_D}, C'_L (amplitude dominante) e St (Strouhal)
- Estabilizações SUPG/PSPG (ativadas por padrão), opcional LSIC

Observações importantes:
1) INLET=2, OUTLET=3, WALLS=4, CYLINDER=45 
2) O perfil de entrada usa velocidade média U_mean. O perfil parabólico é
   u_in(y) = 4 * U_mean * y * (H - y) / H^2  (em x) e v=0 (em y).
   O valor máximo = 1.5 * U_mean.
3) Coeficientes: Cd = 2*Fx/(rho*U_mean^2*D), Cl = 2*Fy/(rho*U_mean^2*D).
   Para St usamos f_dominante*D/U_mean, estimando f pela FFT do sinal de Cl após descarte do transiente.
4) Este arquivo resolve o problema transiente. Para simulação estacionária, use T pequeno e tolere convergência a regime.
5) SUPG/PSPG são casos particulares da formulação RBVMS. Para RBVMS completa, adicionar projeções de escalas não resolvidas.

Como usar:
$ python3 ns_dfg2d.py --mesh mesh_dfg2d.xdmf --mf mf_dfg2d.xdmf --Re 100 --D 0.1 --H 0.41 \
    --Umean 1.0 --nu -1  --rho 1.0 --dt 0.001 --T 8.0 --save 1
Se --nu<0, será calculado por nu = Umean*D/Re.

Geração esperada de arquivos:
- results/velocity.pvd, results/pressure.pvd
- results/forces.csv (t, Fx, Fy, Cd, Cl)
- results/metrics.txt (\bar{C_D}, C'_L, St)
"""

from argparse import ArgumentParser
import os

parameters["form_compiler"]["quadrature_degree"] = 3    #4
parameters["allow_extrapolation"] = True

# Dimensões do domínio 
L = 2.5
H_val = 0.41  # valor usado para marcação
D_val = 0.1   # usado no marcador do cilindro (ajustar se necessário)

# --------------------- Parâmetros / CLI ---------------------
parser = ArgumentParser()
parser.add_argument('--Re', type=float, default=100.0)
parser.add_argument('--Umean', type=float, default=1.0)
parser.add_argument('--nu', type=float, default=0.001)
parser.add_argument('--rho', type=float, default=1.0)
parser.add_argument('--D', type=float, default= D_val, help='Diâmetro do cilindro')
parser.add_argument('--H', type=float, default= H_val, help='Altura do canal')
parser.add_argument('--dt', type=float, default=1e-2)
parser.add_argument('--T', type=float, default=8.0)  # 8.0 ou 10.0 para simulação completa
parser.add_argument('--save', type=int, default=1, help='salvar VTK/CSV: 1 sim, 0 não')
parser.add_argument('--output', type=str, default='/home/vrodrigues/supg_stabilization/results_teste_vsualizacao')
args = parser.parse_args()

comm = MPI.comm_world
rank = comm.Get_rank()

if rank == 0:
    print("\n=== Configuração ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

# --------------------- Leitura da malha ---------------------
mesh = Mesh("/home/vrodrigues/supg_stabilization/Trabalho_final/dfg2d.xml")


# --------------------- Marcação das Fronteiras ---------------------
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[1], H_val))

class Cylinder(SubDomain):
    def inside(self, x, on_boundary):
        # Ajuste conforme necessário se o cilindro for exato
        return on_boundary and (x[0] > 0.1 and x[0] < 0.3 and x[1] > 0.1 and x[1] < 0.3)

facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
facet_markers.set_all(0)

INLET, OUTLET, WALLS, CYLINDER = 2, 3, 4, 5
Inlet().mark(facet_markers, INLET)
Outlet().mark(facet_markers, OUTLET)
Walls().mark(facet_markers, WALLS)
Cylinder().mark(facet_markers, CYLINDER)

# Diagnóstico rápido
if rank == 0:
    print("Valores únicos nos facet_markers:", np.unique(facet_markers.array()))

outlet_faces = np.sum(facet_markers.array() == OUTLET)
print("Número de faces no OUTLET:", outlet_faces)


# --------------------- Medidas e normals ---------------------
n = FacetNormal(mesh)
ds = Measure('ds', domain=mesh, subdomain_data=facet_markers)
dx = Measure('dx', domain=mesh)

# --------------------- Espaços e funções ---------------------
V = VectorElement('Lagrange', mesh.ufl_cell(), 2)
Q = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, MixedElement([V, Q]))

U = Function(W)
U_n = Function(W)
(u, p) = split(U)
(u_n, p_n) = split(U_n)
(v, q) = TestFunctions(W)

# --------------------- Parâmetros físicos ---------------------
rho = Constant(args.rho)
H = Constant(args.H)
D = Constant(args.D)
Umean = Constant(args.Umean)
if args.nu < 0:
    nu_val = args.Umean*args.D/args.Re
else:
    nu_val = args.nu
nu = Constant(nu_val)
mu = Constant(nu_val*args.rho)

dt = Constant(args.dt)

# --------------------- Condições de contorno ---------------------
uin = Expression((
    '4.0*Umean*x[1]*(H - x[1])/(H*H)',
    '0.0'
), Umean=args.Umean, H=args.H, degree=2)

# Subespaços do espaço misto
vel_subspace = W.sub(0)
press_subspace = W.sub(1)

# Criar DirichletBCs diretamente usando o MeshFunction lido do XDMF
bc_inlet = DirichletBC(vel_subspace, uin, facet_markers, INLET)
bc_walls = DirichletBC(vel_subspace, Constant((0.0, 0.0)), facet_markers, WALLS)
bc_cyl = DirichletBC(vel_subspace, Constant((0.0, 0.0)), facet_markers, CYLINDER)
bc_p_out = DirichletBC(press_subspace, Constant(0.0), facet_markers, OUTLET)

# Lista de todas as BCs
bcs = [bc_inlet, bc_walls, bc_cyl, bc_p_out]

# --------------------- Termos e forma fraca ---------------------
Du = (u - u_n)/dt

I = Identity(mesh.geometry().dim())
Dsym = sym(grad(u))
sigma = -p*I + 2.0*mu*Dsym

f = Constant((0.0, 0.0))

F_momentum = rho*inner(Du, v)*dx \
           + rho*inner(dot(u, nabla_grad(u)), v)*dx \
           + inner(sigma, grad(v))*dx \
           - inner(f, v)*dx

F_continuity = - div(u)*q*dx
F = F_momentum + F_continuity

# -------- Estabilização SUPG/PSPG --------
TAU_SCALE = 1.0
Cnu = 9.0 * (4.0 / 2.0)**2

h = CellDiameter(mesh)
vnorm = sqrt(dot(u_n, u_n)) + Constant(1e-12)

tau_raw = ((2.0/dt)**2 + (2.0*vnorm/h)**2 + Cnu*(nu/(h*h))**2) 
tau = TAU_SCALE / sqrt(tau_raw + 1e-12)

# Inspecionar os valores de τ
tau_project = project(tau, FunctionSpace(mesh, 'CG', 1))
tau_array = tau_project.vector().get_local()
print("τ médio:", tau_array.mean(), " | máx:", tau_array.max(), " | mín:", tau_array.min())

r_m = rho * ((u - u_n)/dt + dot(u, nabla_grad(u))) - div(2.0*mu*Dsym) + grad(p) - f
r_c = div(u)

ENABLE_SUPG = False
if ENABLE_SUPG:
    F += inner(tau * r_m, dot(u, nabla_grad(v))) * dx

ENABLE_PSPG = False
if ENABLE_PSPG:
    PSPG_SCALE = 0.72 
    F += PSPG_SCALE * inner(tau *grad(q), r_m) * dx
    #F += inner(tau * r_m, grad(q)) * dx

ENABLE_LSIC = False
if ENABLE_LSIC:
    tau_c = TAU_SCALE * h*h / (2.0/dt + nu + 1e-12)
    F += inner(tau_c * r_c, div(v)) * dx

# --------------------- Solução no tempo ---------------------
t = 0.0
Tfinal = float(args.T)

if rank == 0 and args.save:
    os.makedirs(args.output, exist_ok=True)
vel_file = File(os.path.join(args.output, "velocity.pvd")) if args.save else None
pre_file = File(os.path.join(args.output, "pressure.pvd")) if args.save else None

if rank == 0 and args.save:
    fcsv = open(os.path.join(args.output, 'forces.csv'), 'w')
    fcsv.write('t,Fx,Fy,Cd,Cl\n')
else:
    fcsv = None

lift_hist = []
drag_hist = []
time_hist = []

assign(U_n.sub(0), project(Constant((0.0, 0.0)), VectorFunctionSpace(mesh, 'Lagrange', 2)))
assign(U_n.sub(1), project(Constant(0.0), FunctionSpace(mesh, 'Lagrange', 1)))

J = derivative(F, U)
problem = NonlinearVariationalProblem(F, U, bcs, J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1e-8
prm['newton_solver']['relative_tolerance'] = 1e-7
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['linear_solver'] = 'mumps'         

ex = Constant((1.0, 0.0))
ey = Constant((0.0, 1.0))

step = 0
while t < Tfinal - 1e-12:
    t += float(dt)
    step += 1
    if rank == 0:
        print(f"\n>>> t = {t:.6f} (step {step})")

    solver.solve()

    if args.save:
        U_vec, P_sc = U.split(deepcopy=True)
        U_vec.rename('velocity', 'u')
        P_sc.rename('pressure', 'p')
        if vel_file is not None:
            vel_file << (U_vec, t)
        if pre_file is not None:
            pre_file << (P_sc, t)

    stress_n = dot(sigma, n)
    Fx = assemble(dot(stress_n, ex)*ds(CYLINDER))
    Fy = assemble(dot(stress_n, ey)*ds(CYLINDER))

    Cd = 2.0*Fx/(float(rho)*args.Umean**2*args.D)
    Cl = 2.0*Fy/(float(rho)*args.Umean**2*args.D)

    if rank == 0:
        time_hist.append(t)
        drag_hist.append(Cd)
        lift_hist.append(Cl)
        if fcsv:
            fcsv.write(f"{t:.9f},{Fx:.9e},{Fy:.9e},{Cd:.9e},{Cl:.9e}\n")
            fcsv.flush()

    U_n.assign(U)

if rank == 0:
    time_hist = np.array(time_hist)
    drag_hist = np.array(drag_hist)
    lift_hist = np.array(lift_hist)

    if len(time_hist) > 10:
        t_cut = 0.5*(time_hist[0] + time_hist[-1])
        mask = time_hist >= t_cut
        t_sig = time_hist[mask]
        Cl_sig = lift_hist[mask]
        Cd_sig = drag_hist[mask]
    else:
        t_sig = time_hist
        Cl_sig = lift_hist
        Cd_sig = drag_hist

    Cd_mean = float(np.mean(Cd_sig)) if len(Cd_sig) else np.nan

    if len(Cl_sig) > 4:
        Cl_pp = float(np.max(Cl_sig) - np.min(Cl_sig))
        Cl_prime = 0.5*Cl_pp
    else:
        Cl_prime = float(np.std(Cl_sig)*np.sqrt(2.0)) if len(Cl_sig) else np.nan

    if len(t_sig) > 4:
        dt_series = np.diff(t_sig)
        dtm = float(np.mean(dt_series)) if len(dt_series) else float(args.dt)
        fs = 1.0/dtm
        nfft = int(2**np.ceil(np.log2(len(Cl_sig))))
        Cl_hat = np.fft.rfft(Cl_sig - np.mean(Cl_sig), n=nfft)
        freqs = np.fft.rfftfreq(nfft, d=dtm)
        if len(freqs) > 1:
            k = np.argmax(np.abs(Cl_hat[1:])) + 1
            f_dominante = float(freqs[k])
        else:
            f_dominante = np.nan
    else:
        f_dominante = np.nan

    St = float(f_dominante*float(D)/float(Umean)) if (not np.isnan(f_dominante)) else np.nan

    if args.save:
        with open(os.path.join(args.output, 'metrics.txt'), 'w') as fm:
            fm.write(f"Cd_mean = {Cd_mean}\n")
            fm.write(f"Cl_prime = {Cl_prime}\n")
            fm.write(f"f_dominante = {f_dominante}\n")
            fm.write(f"St = {St}\n")

    print("\n=== Métricas finais (pós-transiente) ===")
    print(f"Cd_mean = {Cd_mean}")
    print(f"Cl' (amplitude) = {Cl_prime}")
    print(f"f_dominante = {f_dominante}")
    print(f"St = {St}")

    if fcsv:
        fcsv.close()

print("\nConcluído.")
