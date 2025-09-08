// Dimensões
L = 2.2;
H = 0.41;
r = 0.05;
xc = 0.2;
yc = 0.2;

// Pontos do retângulo externo
Point(1) = {0, 0, 0, 0.02};
Point(2) = {L, 0, 0, 0.02};
Point(3) = {L, H, 0, 0.02};
Point(4) = {0, H, 0, 0.02};

// Contorno externo
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};

// Cilindro (definido por arcos de círculo)
Point(5) = {xc, yc, 0, 0.02};
p1 = newp; Point(p1) = {xc + r, yc, 0, 0.02};
p2 = newp; Point(p2) = {xc, yc + r, 0, 0.02};
p3 = newp; Point(p3) = {xc - r, yc, 0, 0.02};
p4 = newp; Point(p4) = {xc, yc - r, 0, 0.02};

Circle(7) = {p1, 5, p2};
Circle(8) = {p2, 5, p3};
Circle(9) = {p3, 5, p4};
Circle(10) = {p4, 5, p1};

Line Loop(11) = {7, 8, 9, 10};

// Domínio com furo (cilindro)
Plane Surface(12) = {5, 11};

// Marcação de fronteiras para FEniCS
Physical Line("inlet")    = {4};
Physical Line("walls")    = {1, 3};
Physical Line("outlet")   = {2};
Physical Line("cylinder") = {7, 8, 9, 10};
Physical Surface("fluid") = {12};

// Marcar os pontos-cantos do retângulo (remove marcador -1)
Physical Point("corners") = {1, 2, 3, 4};
