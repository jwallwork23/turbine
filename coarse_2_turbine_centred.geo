W=200.;   // width of channel
D=18.;    // turbine diameter
xt1=325.; // x location of turbine1
xt2=675.; // x location of turbine2
dx1=10.;  // Outer resolution
dx2=2.;   // Inner resolution
L=1e3;    // length of channel
yt1=W/2;  // y location of turbine1
yt2=W/2;  // y location of turbine2
Point(1) = {0., 0., 0., dx1};
Point(2) = {L,  0., 0., dx1};
Point(3) = {L,  W,  0., dx1};
Point(4) = {0., W,  0., dx1};
Point(5) = {xt1-D/2, yt1-D/2, 0., dx2};
Point(6) = {xt1+D/2, yt1-D/2, 0., dx2};
Point(7) = {xt1+D/2, yt1+D/2, 0., dx2};
Point(8) = {xt1-D/2, yt1+D/2, 0., dx2};
Point(9) = {xt2-D/2, yt2-D/2, 0., dx2};
Point(10) = {xt2+D/2, yt2-D/2, 0., dx2};
Point(11) = {xt2+D/2, yt2+D/2, 0., dx2};
Point(12) = {xt2-D/2, yt2+D/2, 0., dx2};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) =  {9,  10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 9};
Physical Line(1) = {4};   // Left boundary
Physical Line(2) = {2};   // Right boundary
Physical Line(3) = {1,3}; // Sides
// outside loop
Line Loop(1) = {1, 2, 3, 4};
// inside loop1
Line Loop(2) = {5, 6, 7, 8};
// inside loop2
Line Loop(3) = {9, 10, 11, 12};
Plane Surface(1) = {1,2,3};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
// id outside turbine
Physical Surface(1) = {1};
// id inside turbine1
Physical Surface(2) = {2};
// id inside turbine2
Physical Surface(3) = {3};
