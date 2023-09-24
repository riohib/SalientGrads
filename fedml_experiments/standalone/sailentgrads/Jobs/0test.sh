#!/bin/bash

for alpha in 0.1 0.2 0.3 0.5 1
do
    recip=$(echo "1-$alpha" | bc -l)
    echo "alpha: $alpha | recip: $recip"
done