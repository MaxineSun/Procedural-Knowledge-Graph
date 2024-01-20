#!/bin/bash

total_time=0

for i in {0..21}
do
  filename="sorted_5_gpt2-xl_16_$i.out"
  time_string=$(grep -a "it took [0-9.]* seconds" "$filename" | sed -n 's/.*it took \([0-9.]*\) seconds.*/\1/p' | tail -n 1)
  total_time=$(echo "$total_time + $time_string" | bc)
done

average_time=$(echo "scale=6; $total_time / 28 / 50" | bc)

echo "Average time: $average_time seconds"

