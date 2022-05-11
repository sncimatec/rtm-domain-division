#ifndef FD_H
#define FD_H

#include<CL/sycl.hpp>

void fd_init_org(int order, int nx, int nz, float dx, float dz, float dt);

void fd_init(int order, int nx, int nz, int nxb, int nzb, int nt, int ns, float fac, float dx, float dz, float dt); 

void fd_step(int order, float **p, float **pp, float **v2, int nz, int nx);
void write_buffers(float **p, float **pp, float **v2, float *taperx, float *taperz, 
                float ***d_obs, float ***wf, int flag);
void fd_forward(int order, float **p, float **pp, float **v2, float ***swf,  
			    int nx, int nz, int nt, int is, int sz, int *sx, float *srce, int flag); 
void fd_backward(int order, float **p, float **pp, float **v2, float ***rwf, float ***dobs,  
			   int nx, int nz, int nt, int ns, int gz, int is, int it, int sz, int *sx, float *srce, int flag); 

void fd_destroy();
float *calc_coefs(int order);

#endif
