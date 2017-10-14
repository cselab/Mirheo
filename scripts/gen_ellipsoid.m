#! /usr/bin/env octave

args = argv();

if nargin < 4
	printf("Usage: a b c n\n  (a b c) are ellipsoid axes,\n  n is number of particles");
	exit;
end

%rand("seed", 424242);

myaxes = [str2num(args{1}), str2num(args{2}), str2num(args{3})];
smallax = myaxes - 1.5;

n = str2num(args{4});

printf("%d\n", n);
printf("# Generated ellipsoid with axes %f %f %f\n", myaxes(1), myaxes(2), myaxes(3));

com = [0 0 0];
for i = 1:n-1

	while true
		ang = rand(1, 2);
		u = 2*pi*ang(1);
		v = pi*ang(2);

		len = (myaxes - smallax).*rand(1, 3) + smallax;
		x = len(1)*cos(u)*sin(v);
		y = len(2)*sin(u)*sin(v);
		z = len(3)*cos(v);
		
		candcom = com + [x y z];
		
		if sum(candcom.*candcom ./ (smallax .* smallax)) > 1 && sum(candcom.*candcom ./ (myaxes .* myaxes)) < 1
			break;
		end
	end
	
	com = com + [x y z];
	printf("0 %f %f %f\n", x, y, z);
end

printf("0 %f %f %f\n", -com(1), -com(2), -com(3));
