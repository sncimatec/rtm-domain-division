#!/bin/sh
cd ../../src 
make allclean 
make
cd -
rm -f dir.*
time ../rtm_main par=input.dat

exit 0
