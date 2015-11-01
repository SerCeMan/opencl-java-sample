#define SWAP(a,b) {__local int * tmp=a; a=b; b=tmp;}

__kernel void block_scan(__global float * input, __global float * output, __global float *summand, __local float * a, __local float * b)
{
    uint lid = get_local_id(0);
    uint grid = get_group_id(0);
    uint block_size = get_local_size(0);
    uint idx = lid + grid * block_size;

    a[lid] = b[lid] = input[idx];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint s = 1; s < block_size; s <<= 1)
    {
        if(lid > (s-1))
        {
            b[lid] = a[lid] + a[lid-s];
        }
        else
        {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a,b);
    }
    if (lid == block_size - 1) {
        summand[grid] = a[lid];
    }
    output[idx] = a[lid];
}

__kernel void sum_scan(__global float * input, __global float * output, __local float * a, __local float * b)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);

    a[lid] = b[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint s = 1; s < block_size; s <<= 1)
    {
        if(lid > (s-1))
        {
            b[lid] = a[lid] + a[lid-s];
        }
        else
        {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a,b);
    }
    output[gid] = a[lid];
}

__kernel void add_scan(__global float * input, __global float * sum)
{
    uint lid = get_local_id(0);
    uint grid = get_group_id(0);
    uint block_size = get_local_size(0);


    if (grid > 0) {
        input[lid + grid * block_size] += sum[grid - 1];
    }
}