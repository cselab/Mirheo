#!/usr/bin/env perl

use strict;
use List::Util qw(min max);
use Text::Tabs;

chomp(my @text = <>);

$_ =~ s/^(.*?)\s+\\$/$1/ foreach (@text);

print map {"// ".$_."\n"} @text;
print "\n";

# Change // ... comments to /* ... */, othw everything will fail
$_ =~ s/\/\/(.*)$/\/* $1 *\// foreach (@text);
$tabstop = 4;
my $maxLen = max (map {length(expand($_))} @text);
$_ = $_ . " "x($maxLen - length(expand($_))+4) . "\\\n" foreach (@text[0..$#text-1]);

print @text;
print "\n";
