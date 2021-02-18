#define swapUInt(__a,__b) { unsigned int __t = __a; __a = __b; __b = __t; }
#define swapReal(__a,__b) { real __t = __a; __a = __b; __b = __t; }

__local unsigned int *getKN(__local unsigned int *kn, int id);
__local real *getKNDist(__local real *kndist, int id);
__local unsigned int *getKN(__local unsigned int *kn, int id) { return &kn[get_local_size(0)*id]; }
__local real *getKNDist(__local real *kndist, int id) { return &kndist[get_local_size(0)*id]; }

void heapify(__local unsigned int *keys, __local real *vals, int node, int size);
void heapsort(__local unsigned int *keys, __local real *vals, int size);

void heapify(__local unsigned int *keys, __local real *vals, int node, int size) {
  int j = node;
  while (true) {
    int left  = 2*j+1;
    int right = 2*j+2;
    int largest = j;
    if ( left<size && (*getKNDist(vals,left) > *getKNDist(vals,largest)))
      largest = left;
    if (right<size && (*getKNDist(vals,right) > *getKNDist(vals,largest)))
      largest = right;
    if (largest==j) return;
    swapReal(*getKNDist(vals,j), *getKNDist(vals,largest));
    swapUInt(*getKN(keys,j), *getKN(keys,largest));
    j = largest;
  }
}
void heapsort(__local unsigned int *keys, __local real *vals, int size) {
  while (size) {
    swapReal(*getKNDist(vals,0), *getKNDist(vals,size - 1));
    swapUInt(*getKN(keys,0), *getKN(keys,size - 1));
    heapify(keys, vals, 0, --size);
  }
}

__kernel void build_knearests(__local real *lMemory, int dim, __global const int *counters, __global const real4 *stored_points,
                              int num_cell_offsets, __global const int *cell_offsets, __global const real *cell_offset_distances,
                              int __global const * heap, int num, int offset, __global unsigned int *g_knearests)
{
  int gid = get_global_id(0);
  if (gid >= num) return;
  int point_in = heap[gid+offset];
  
  real4 p = stored_points[point_in];
  int cell_in = cellFromPoint(dim, p);
  int lid = get_local_id(0);
  __local real *kn_dists = &lMemory[lid];
  __local unsigned int *kn = (__local unsigned int *) (&lMemory[get_local_size(0)*K]);
  kn+=lid;
  
  for (int i = 0; i < K; i++) {
    *getKNDist(kn_dists,i) = FLT_MAX;
    *getKN(kn,i) = UINT_MAX;
  }
  int o = 0;
  int maxDim=dim*dim*dim;
  do {
    real min_dist = cell_offset_distances[o];
    if (*getKNDist(kn_dists,0) < min_dist) break;
    int cell = cell_in + cell_offsets[o];
    if (cell >= 0 && cell < maxDim) {
    for (int ptr = counters[cell]; ptr < counters[cell+1]; ptr++) {
      if (ptr == point_in) continue;
        real4 p_cmp = stored_points[ptr];
        real d = (p_cmp.x - p.x)*(p_cmp.x - p.x) + (p_cmp.y - p.y)*(p_cmp.y - p.y) + (p_cmp.z - p.z)*(p_cmp.z - p.z);
        if (d < *getKNDist(kn_dists,0)) {
          *getKNDist(kn_dists,0) = d;
          *getKN(kn,0) = ptr;
          heapify(kn, kn_dists, 0, K);
        }
      }
    }
  } while (o++ < num_cell_offsets);
  heapsort(kn, kn_dists, K);
  
  for (int i = 0; i < K; i++) 
     g_knearests[gid*K + i] = *getKN(kn,i);
}
