#!/bin/bash

total_time=0

for i in {0..5}
do
  filename="bubble_2_gpt2-xl_8_$i.out"
  time_string=$(grep -a "it took [0-9.]* seconds" "$filename" | sed -n 's/.*it took \([0-9.]*\) seconds.*/\1/p')
  total_time=$(echo "$total_time + $time_string" | bc)
done

average_time=$(echo "scale=6; $total_time / 12 / 50" | bc)

echo "Average time: $average_time seconds"

