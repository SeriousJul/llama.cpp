#include <cstdio>
int main(){
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("device: %s  sm_%d%d\n", p.name, p.major, p.minor);
    printf("  sharedMemPerBlock            %8zu B (%.1f KiB)   static cap\n", p.sharedMemPerBlock, p.sharedMemPerBlock/1024.0);
    printf("  sharedMemPerBlockOptin       %8d B (%.1f KiB)   per block, after cudaFuncSetAttribute\n", p.sharedMemPerBlockOptin, p.sharedMemPerBlockOptin/1024.0);
    printf("  sharedMemPerMultiprocessor   %8zu B (%.1f KiB)   per SM\n", p.sharedMemPerMultiprocessor, p.sharedMemPerMultiprocessor/1024.0);
    printf("  regsPerBlock %d  regsPerMultiprocessor %d  maxThreadsPerMP %d\n\n", p.regsPerBlock, p.regsPerMultiprocessor, p.maxThreadsPerMultiProcessor);
    const int DK=128, DV=128;
    printf("%-6s %-10s %-10s %-10s   %-9s %-9s\n","C","state","Q+K tiles","V or T","no alias","alias V/T");
    for (int C : {16,32,64,128}) {
        double state=DK*DV*2.0, tile=C*DK*2.0, vtile=C*DV*2.0;
        printf("%-6d %-10.0f %-10.0f %-10.0f   %6.1f KiB %6.1f KiB\n", C, state/1024, 2*tile/1024, vtile/1024,
               (state+2*tile+vtile)/1024, (state+2*tile+ (vtile> C*C*4.0 ? vtile : C*C*4.0))/1024);
    }
    printf("\nblocks per SM at the aliased layout: state 32 KiB + 2 tiles + max(V, T)\n");
    return 0;
}
