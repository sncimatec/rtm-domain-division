#include "cwp.h"
#include "fd.h"

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#define sizeblock 64 // Best number to iris 

static void makeo2 (float *coef,int order);

float *d_p, *d_pr, *d_pp, *d_ppr, *d_swap;
float *d_laplace, *d_v2, *d_coefs_x, *d_coefs_z;
float *d_taperx, *d_taperz, *d_swf, *d_rwf;

size_t mtxBufferLength, brdBufferLength;
size_t imgBufferLength, obsBufferLength;
size_t coefsBufferLength;

float *taper_x, *taper_z;
int nxbin, nzbin;

int gridx, gridz;
int gridBorder_x, gridBorder_z;

static float dx2inv,dz2inv,dt2;
static float **laplace = NULL;
static float *coefs = NULL;
static float *coefs_z = NULL;
static float *coefs_x = NULL;

float *calc_coefs(int order);

void fd_init(int order, int nx, int nz, float dx, float dz, float dt){
	dx2inv = (1./dx)*(1./dx);
        dz2inv = (1./dz)*(1./dz);
	dt2 = dt*dt;

	coefs = calc_coefs(order);
	laplace = alloc2float(nz,nx);
	
	memset(*laplace,0,nz*nx*sizeof(float));

	return;
}

void fd_step(int order, float **p, float **pp, float **v2, int nz, int nx){
	int ix,iz,io;
	float acm = 0;

	for(ix=order/2;ix<nx-order/2;ix++){
		for(iz=order/2;iz<nz-order/2;iz++){
			for(io=0;io<=order;io++){
				acm += p[ix][iz+io-order/2]*coefs[io]*dz2inv;
				acm += p[ix+io-order/2][iz]*coefs[io]*dx2inv;
			}
			laplace[ix][iz] = acm;
			acm = 0.0;
		}
	}

	for(ix=0;ix<nx;ix++){
		for(iz=0;iz<nz;iz++){
			pp[ix][iz] = 2.*p[ix][iz] - pp[ix][iz] + v2[ix][iz]*dt2*laplace[ix][iz];
		}
	}

	return;
}

void fd_destroy(){
	free2float(laplace);
	free1float(coefs);
	return;
}

float *calc_coefs(int order){
	float *coef;

	coef = (float*) calloc(order+1,sizeof(float));

	switch(order){
		case 2:
			coef[0] = 1.;
			coef[1] = -2.;
			coef[2] = 1.;
			break;
		case 4:
			coef[0] = -1./12.;
			coef[1] = 4./3.;
			coef[2] = -5./2.;
			coef[3] = 4./3.;
			coef[4] = -1./12.;
			break;
		case 6:
			coef[0] = 1./90.;
			coef[1] = -3./20.;
			coef[2] = 3./2.;
			coef[3] = -49./18.;
			coef[4] = 3./2.;
			coef[5] = -3./20.;
			coef[6] = 1./90.;
			break;
		case 8:
			coef[0] = -1./560.;
			coef[1] = 8./315.;
			coef[2] = -1./5.;
			coef[3] = 8./5.;
			coef[4] = -205./72.;
			coef[5] = 8./5.;
			coef[6] = -1./5.;
			coef[7] = 8./315.;
			coef[8] = -1./560.;
			break;
		default:
			makeo2(coef,order);
	}

	return coef;
}

static void makeo2 (float *coef,int order){
	float h_beta, alpha1=0.0;
	float alpha2=0.0;
	float  central_term=0.0; 
	float coef_filt=0; 
	float arg=0.0; 
	float  coef_wind=0.0;
	int msign,ix; 
	float alpha = .54;
	float beta = 6.;

	h_beta = 0.5*beta;
	alpha1=2.*alpha-1.0;
	alpha2=2.*(1.0-alpha);
	central_term=0.0;

	msign=-1;

	for (ix=1; ix <= order/2; ix++){      
		msign=-msign ;            
		coef_filt = (2.*msign)/(ix*ix); 
		arg = PI*ix/(2.*(order/2+2));
		coef_wind=pow((alpha1+alpha2*cos(arg)*cos(arg)),h_beta);
		coef[order/2+ix] = coef_filt*coef_wind;
		central_term = central_term + coef[order/2+ix]; 
		coef[order/2-ix] = coef[order/2+ix]; 
	}
	
	coef[order/2]  = -2.*central_term;

	return; 
}

// =========================== Init Input ==========================
void fd_init_cuda(int order, int nxe, int nze, int nxb, int nzb, int nt, int ns, float fac)
{
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
 	sycl::queue &q_ct1 = dev_ct1.default_queue();
	float dfrac;
   	nxbin=nxb; nzbin=nzb;
   	brdBufferLength = nxb*sizeof(float);
   	mtxBufferLength = (nxe*nze)*sizeof(float);
   	coefsBufferLength = (order+1)*sizeof(float);
	obsBufferLength = nt*(nxe-(2*nxb))*sizeof(float);
   	imgBufferLength = (nxe-(2*nxb))*(nze-(2*nzb))*sizeof(float);

	taper_x = alloc1float(nxb);
	taper_z = alloc1float(nzb);

        dfrac = sqrt(-log(fac)) / (1. * nxb);
        for(int i=0;i<nxb;i++)
          taper_x[i] = exp(-pow((dfrac * (nxb - i)), 2));

        dfrac = sqrt(-log(fac)) / (1. * nzb);
        for(int i=0;i<nzb;i++)
          taper_z[i] = exp(-pow((dfrac * (nzb - i)), 2));

        // Create a Device pointers
        d_v2 = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
        d_p = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
        d_pp = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
        d_pr = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
        d_ppr = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
        d_swap = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
        d_laplace = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);

        d_rwf = (float *)sycl::malloc_device(obsBufferLength, q_ct1);
        d_swf = (float *)sycl::malloc_device(imgBufferLength, q_ct1);
        d_coefs_x = (float *)sycl::malloc_device(coefsBufferLength, q_ct1);
        d_coefs_z = (float *)sycl::malloc_device(coefsBufferLength, q_ct1);
        d_taperx = (float *)sycl::malloc_device(brdBufferLength, q_ct1);
        d_taperz = (float *)sycl::malloc_device(brdBufferLength, q_ct1);

        int div_x, div_z;
	// Set a Grid for the execution on the device
	div_x = (float) nxe/(float) sizeblock;
	div_z = (float) nze/(float) sizeblock;
        gridx = (int)ceil(div_x);
        gridz = (int)ceil(div_z);

        div_x = (float) nxb/(float) sizeblock;
	div_z = (float) nzb/(float) sizeblock;
        gridBorder_x = (int)ceil(div_x);
        gridBorder_z = (int)ceil(div_z);

        div_x = (float) 8/(float) sizeblock;
}

// ============================ Kernels ============================
void kernel_lap(int order, int nx, int nz, float * __restrict__ p, float * __restrict__ lap, float * __restrict__ coefsx, float * __restrict__ coefsz,
                sycl::nd_item<3> item_ct1){
	int half_order=order/2;
	int i = half_order +
			item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
			item_ct1.get_local_id(2); // Global row index
	int j = half_order +
			item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
			item_ct1.get_local_id(1); // Global column index
	int mult = i*nz;
	int aux;
	float acmx = 0, acmz = 0;

	if(i<nx - half_order)
	{
		if(j<nz - half_order)
		{
			for(int io=0;io<=order;io++)
			{
				aux = io-half_order;
				acmz += p[mult + j+aux]*coefsz[io];
				acmx += p[(i+aux)*nz + j]*coefsx[io];
			}
			lap[mult +j] = acmz + acmx;
			acmx = 0.0;
			acmz = 0.0;
		}
	}

}

void kernel_time(int nx, int nz, float *__restrict__ p, float *__restrict__ pp, float *__restrict__ v2, float *__restrict__ lap, float dt2,
                 sycl::nd_item<3> item_ct1){

	int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
			item_ct1.get_local_id(2); // Global row index
	int j = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
			item_ct1.get_local_id(1); // Global column index
	int mult = i*nz;

  	if(i<nx){
  		if(j<nz){
			 pp[mult+j] = 2.*p[mult+j] - pp[mult+j] + v2[mult+j]*dt2*lap[mult+j];
		}
  	}
}

void kernel_tapper(int nx, int nz, int nxb, int nzb, float *__restrict__ p, float *__restrict__ pp, float *__restrict__ taperx, float *__restrict__ taperz,
                   sycl::nd_item<3> item_ct1){

	int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
			item_ct1.get_local_id(2); // nx index
	int j = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
			item_ct1.get_local_id(1); // nzb index
	int itxr = nx - 1, mult = i*nz;

	if(i<nx){
		if(j<nzb){
			p[mult+j] *= taperz[j];
			pp[mult+j] *= taperz[j];
		}
	}

	if(i<nxb){
		if(j<nzb){
			p[mult+j] *= taperx[i];
			pp[mult+j] *= taperx[i];

			p[(itxr-i)*nz+j] *= taperx[i];
			pp[(itxr-i)*nz+j] *= taperx[i];
		}
	}
}

void kernel_src(int nz, float * __restrict__ pp, int sx, int sz, float srce){
 	pp[sx*nz+sz] += srce;
}

void fd_foward(int order, float *P, float *PP, float *vel2,  
			   int nxe, int nze, int nt, int is, int sz, int *sx, float *srce){

}