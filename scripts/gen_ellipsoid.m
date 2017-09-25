#! /usr/bin/env octave

args = argv();

if nargin < 4
	printf("Usage: a b c n\n  (a b c) are ellipsoid axes,\n  n is number of particles");
	exit;
end

myaxes = [str2num(args{1}), str2num(args{2}), str2num(args{3})];
smallax = myaxes - 1.5;

n = str2num(args{4});

printf("%d\n", n);
printf("# Generated ellipsoid with axes %f %f %f\n", myaxes(1), myaxes(2), myaxes(3));

for i = 1:n
	ang = rand(1, 2);
	u = 2*pi*ang(1);
	v = pi*ang(2);

	len = (myaxes - smallax).*rand(1, 3) + smallax;
	x = len(1)*cos(u)*sin(v);
	y = len(2)*sin(u)*sin(v);
	z = len(3)*cos(v);
	
	printf("0 %f %f %f\n", x, y, z);
end

