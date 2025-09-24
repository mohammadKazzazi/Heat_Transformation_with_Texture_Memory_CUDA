# Heat_Transformation_with_Texture_Memory_CUDA

Consider a two-dimensional room that is divided into cells, with each cell having a temperature
randomly ranging between 20 to 30 degrees Celsius. As each cell has a different temperature
from its neighbors, heat transformation occurs between these cells and their neighbors to establish thermal balance. See the cuda_htt.pdf for a complete explanation.

Compile: 
```console
nvcc -O2 htt_main.cu htt.cu -o htt
```
Execute: 
```console
./htt M
```

Note that $N=2^M$ and $M = 10$, $11$, $12$ and $13$.
