#include "cwp.h"
#include "fd.h"

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

using namespace std; 
using namespace sycl; 

#define sizeblock 16 // Best number to iris 

static void makeo2 (float *coef,int order);

float *d_p, *d_pr, *d_pp, *d_ppr, *d_swap;
float *d_laplace, *d_v2, *d_coefs_x, *d_coefs_z;
float *d_taperx, *d_taperz, *d_swf, *d_rwf, *d_dobs;

size_t mtxBufferLength, brdBufferLength;
size_t imgBufferLength, dobsBufferLength;
size_t coefsBufferLength, waveBufferLength;

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

void fd_init_org(int order, int nx, int nz, float dx, float dz, float dt){
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
void fd_init_sycl(int order, int nxe, int nze, int nxb, int nzb, int nt, int ns, float fac)
{
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
 	sycl::queue &q_ct1 = dev_ct1.default_queue();
	
   	nxbin=nxb; nzbin=nzb;
   	
	brdBufferLength = nxb * sizeof(float);
   	mtxBufferLength = (nxe * nze) * sizeof(float);
	waveBufferLength = nt * (nxe - (2 * nxb)) * (nze - (2 * nzb)) * sizeof(float);
   	coefsBufferLength = (order + 1) * sizeof(float);
	dobsBufferLength = nt * (nxe - (2 * nxb)) * ns * sizeof(float);
	taper_x = alloc1float(nxb);
	taper_z = alloc1float(nzb);

	for(int i=0;i<nxb;i++)
		taper_x[i] = exp(-pow((fac * (nxb - i)), 2));

	for(int i=0;i<nzb;i++)
		taper_z[i] = exp(-pow((fac * (nzb - i)), 2));

	// Create a Device pointers
	d_p = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
	d_pp = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
	d_pr = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
	d_ppr = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
	
	d_v2 = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
	
	d_swap = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
	d_laplace = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);

	d_rwf = (float *)sycl::malloc_device(waveBufferLength, q_ct1);
	d_swf = (float *)sycl::malloc_device(waveBufferLength, q_ct1);
	d_dobs = (float *)sycl::malloc_device(dobsBufferLength, q_ct1);
	
	d_coefs_x = (float *)sycl::malloc_device(coefsBufferLength, q_ct1);
	d_coefs_z = (float *)sycl::malloc_device(coefsBufferLength, q_ct1);
	
	d_taperx = (float *)sycl::malloc_device(brdBufferLength, q_ct1);
	d_taperz = (float *)sycl::malloc_device(brdBufferLength, q_ct1);

	int div_x, div_z;
	
	// Set a Grid for the execution on the device
	
	gridx = (float) (((nxe / sizeblock) + 1) * sizeblock)/(float) sizeblock;
	gridz = (float) (((nze / sizeblock) + 1) * sizeblock)/(float) sizeblock;
	gridBorder_x = (float) (((nxb / sizeblock) + 1) * sizeblock)/(float) sizeblock;
	gridBorder_z = (float) (((nzb / sizeblock) + 1) * sizeblock)/(float) sizeblock;
}

void fd_init(int order, int nx, int nz, int nxb, int nzb, int nt, int ns, float fac, float dx, float dz, float dt)
{
	int io;
	dx2inv = (1./dx)*(1./dx);
	dz2inv = (1./dz)*(1./dz);
	dt2 = dt*dt;

	coefs = calc_coefs(order);
	laplace = alloc2float(nz,nx);

	coefs_z = calc_coefs(order);
	coefs_x = calc_coefs(order);

	// pre calc coefs 8 d2 inv
	for (io = 0; io <= order; io++) {
		coefs_z[io] = dz2inv * coefs[io];
		coefs_x[io] = dx2inv * coefs[io];
	}

	memset(*laplace,0,nz*nx*sizeof(float));

    fd_init_sycl(order,nx,nz,nxb,nzb,nt,ns,fac);

    return;
}

void write_buffers(float **p, float **pp, float **v2, float *taperx, float *taperz, float ***dobs, float ***wf, int flag)
{
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue &q_ct1 = dev_ct1.default_queue();
	// gettimeofday(&startCopyMem, NULL);
	if(flag == 0){
		q_ct1.memcpy(d_p, p[0], mtxBufferLength).wait();
		q_ct1.memcpy(d_pp, pp[0], mtxBufferLength).wait();
		q_ct1.memcpy(d_v2, v2[0], mtxBufferLength).wait();
		q_ct1.memcpy(d_laplace, laplace[0], mtxBufferLength).wait();
		
		q_ct1.memcpy(d_coefs_x, coefs_x, coefsBufferLength).wait();
		q_ct1.memcpy(d_coefs_z, coefs_z, coefsBufferLength).wait();
		
		q_ct1.memcpy(d_taperx, taperx, brdBufferLength).wait();
		q_ct1.memcpy(d_taperz, taperz, brdBufferLength).wait();
		q_ct1.memcpy(d_swf, wf[0][0], waveBufferLength).wait();
	}
	if(flag == 1){
		q_ct1.memcpy(d_p, p[0], mtxBufferLength).wait();
		q_ct1.memcpy(d_pp, pp[0], mtxBufferLength).wait();
		q_ct1.memcpy(d_v2, v2[0], mtxBufferLength).wait();
		q_ct1.memcpy(d_laplace, laplace[0], mtxBufferLength).wait();
		
		q_ct1.memcpy(d_coefs_x, coefs_x, coefsBufferLength).wait();
		q_ct1.memcpy(d_coefs_z, coefs_z, coefsBufferLength).wait();
		
		q_ct1.memcpy(d_taperx, taperx, brdBufferLength).wait();
		q_ct1.memcpy(d_taperz, taperz, brdBufferLength).wait();
		q_ct1.memcpy(d_rwf, wf[0][0], waveBufferLength).wait();
		q_ct1.memcpy(d_dobs, dobs[0][0], dobsBufferLength).wait();
	}
	// gettimeofday(&endCopyMem, NULL);
	// execTimeMem += ((endCopyMem.tv_sec - startCopyMem.tv_sec)*1000000 + (endCopyMem.tv_usec - startCopyMem.tv_usec))/1000;
}

// ============================ Kernels ============================
void kernel_lap(int order, int nx, int nz, float * __restrict__ p, float * __restrict__ lap, 
	float * __restrict__ coefsx, float * __restrict__ coefsz, sycl::nd_item<2> item_ct1){
	int half_order=order/2;
	int i = half_order +
			item_ct1.get_group(0) * item_ct1.get_local_range().get(0) +
			item_ct1.get_local_id(0); // Global row index
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
                 sycl::nd_item<2> item_ct1){

	int i = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) +
			item_ct1.get_local_id(0); // Global row index
	int j = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
			item_ct1.get_local_id(1); // Global column index
	int mult = i*nz;

  	if(i<nx){
  		if(j<nz){
			 pp[mult+j] = 2.*p[mult+j] - pp[mult+j] + v2[mult+j]*dt2*lap[mult+j];
		}
  	}
}

void kernel_tapper(int nx, int nz, int nxb, int nzb, float *__restrict__ p, float *__restrict__ pp,
					float *__restrict__ taperx, float *__restrict__ taperz, sycl::nd_item<2> item_ct1){

	int i = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) +
			item_ct1.get_local_id(0); // nx index
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
 	pp[sx*nz + sz] += srce;
}

void kernel_sism(int nx, int nz, int nzb, int nt, int ns, int it, int is, int gz, 
		float * __restrict__ pp, float * __restrict__ dobs, sycl::nd_item<1> item_ct1){
	int i = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) +
			item_ct1.get_local_id(0); // nx index
	int nx_aux = nx - (2*nzb); 
	if(i<nx_aux)
	 	pp[(i+nzb)*nz + gz] += dobs[is*nx_aux*nt + i*nt + (nt-it)]; 
}


void kernel_updt_wfd(float *wf, float *p, int nx, int nz, int it, int nxb,
                 sycl::nd_item<2> item_ct1){

	int i = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) +
			item_ct1.get_local_id(0); // Global row index
	int j = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
			item_ct1.get_local_id(1); // Global column index

  	if(i<nx){
  		if(j<nz){
			 wf[(it*nx*nz) + (i*nz) + j] = p[((i+nxb)*(nz+(2*nxb)))+(j+nxb)]; 
		}
  	}
}

void fd_forward(int order, float **p, float **pp, float **v2, float ***swf,   
			   int nx, int nz, int nt, int is, int sz, int *sx, float *srce){
	
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue &q_ct1 = dev_ct1.default_queue();
	sycl::range<2> dimGrid(gridx, gridz);
	sycl::range<2> dimGridTaper(gridx, gridBorder_z);

	sycl::range<2> dimBlock(sizeblock, sizeblock);
	write_buffers(p, pp, v2, taper_x, taper_z, NULL, swf, 0);
	   	
   	for (int it = 0; it < nt; it++){
		/*
	 	DPCT1049:1: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/

		q_ct1.submit([&](sycl::handler &cgh) {
			auto d_p_ct3 = d_p;
			auto d_laplace_ct4 = d_laplace;
			auto d_coefs_x_ct5 = d_coefs_x;
			auto d_coefs_z_ct6 = d_coefs_z;

			cgh.parallel_for(
				sycl::nd_range<2>(dimGrid*dimBlock, dimBlock),
				[=](sycl::nd_item<2> item_ct1) {
						kernel_lap(order, nx, nz, d_p_ct3,
									d_laplace_ct4, d_coefs_x_ct5,
									d_coefs_z_ct6, item_ct1);
				});
		});
	
		/*
	 	DPCT1049:2: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/
		
		q_ct1.submit([&](sycl::handler &cgh) {
			auto d_p_ct2 = d_p;
			auto d_pp_ct3 = d_pp;
			auto d_v2_ct4 = d_v2;
			auto d_laplace_ct5 = d_laplace;
			auto dt2_ct6 = dt2;

			cgh.parallel_for(
				sycl::nd_range<2>(dimGrid * dimBlock, dimBlock),
				[=](sycl::nd_item<2> item_ct1) {
						kernel_time(nx, nz, d_p_ct2, d_pp_ct3,
									d_v2_ct4, d_laplace_ct5,
									dt2_ct6, item_ct1);
				});
		});
		
		/*
	 	DPCT1049:3: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/
		q_ct1.submit([&](sycl::handler &cgh) {
			auto d_pp_ct1 = d_pp;
			auto sx_is_ct2 = sx[is];
			auto srce_it_ct4 = srce[it];

			cgh.single_task([=]() {
					kernel_src(
					nz, d_pp_ct1, sx_is_ct2, sz	,srce_it_ct4);
				});
		});

				// /*
	 	// DPCT1049:0: The workgroup size passed to the
        //          * SYCL kernel may exceed the limit. To get the device limit,
        //          * query info::device::max_work_group_size. Adjust the workgroup
        //          * size if needed.
	 	// */

		q_ct1.submit([&](sycl::handler &cgh) {
			auto nxbin_ct2 = nxbin;
			auto nzbin_ct3 = nzbin;
			auto d_p_ct4 = d_p;
			auto d_pp_ct5 = d_pp;
			auto d_taperx_ct6 = d_taperx;
			auto d_taperz_ct7 = d_taperz;

			cgh.parallel_for(
				sycl::nd_range<2>(dimGridTaper * dimBlock,
									dimBlock),
				[=](sycl::nd_item<2> item_ct1) {
						kernel_tapper(nx, nz, nxbin_ct2, nzbin_ct3,
										d_p_ct4, d_pp_ct5,
										d_taperx_ct6, d_taperz_ct7,
										item_ct1);
				});
		}).wait();

		// /*
	 	// DPCT1049:0: The workgroup size passed to the
        //          * SYCL kernel may exceed the limit. To get the device limit,
        //          * query info::device::max_work_group_size. Adjust the workgroup
        //          * size if needed.
	 	// */

		q_ct1.submit([&](sycl::handler &cgh) {
			auto d_swf_ct1 = d_swf;
			auto d_p_ct2 = d_p;
			auto it_ct3 = it; 
			auto nxbin_ct4 = nxbin;
			auto nx_ct5 = (nx - (2*nxbin)); 
			auto nz_ct6 = (nz - (2*nzbin)); 

			cgh.parallel_for(
				sycl::nd_range<2>(dimGrid * dimBlock, dimBlock),
				[=](sycl::nd_item<2> item_ct1) {
						kernel_updt_wfd(d_swf_ct1, d_p_ct2, 
						nx_ct5, nz_ct6, it_ct3, nxbin_ct4, 
						item_ct1);
				});
		});
      

		d_swap  = d_pp;
	 	d_pp = d_p;
	 	d_p = d_swap;

		if((it+1)%100 == 0){fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);fprintf(stdout,"\n");}
		
 	}
	// q_ct1.memcpy(p[0], d_p, mtxBufferLength).wait();
	// q_ct1.memcpy(pp[0], d_pp, mtxBufferLength).wait();
	q_ct1.memcpy(swf[0][0], d_swf, waveBufferLength).wait();
}

void fd_backward(int order, float **p, float **pp, float **v2, float ***rwf, float ***dobs,  
			   int nx, int nz, int nt, int ns, int gz, int is, int it, int sz, int *sx, float *srce){
	
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue &q_ct1 = dev_ct1.default_queue();
	
	sycl::range<1> dimGridSingle(gridx);
	sycl::range<2> dimGrid(gridx, gridz);
	sycl::range<2> dimGridTaper(gridx, gridBorder_z);

	sycl::range<1> dimBlockSingle(sizeblock);
	sycl::range<2> dimBlock(sizeblock, sizeblock);
	write_buffers(p, pp, v2, taper_x, taper_z, dobs, rwf, 1);
	   	
   	for (int it = 0; it < nt; it++){
		/*
	 	DPCT1049:1: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/

		q_ct1.submit([&](sycl::handler &cgh) {
			auto d_p_ct3 = d_p;
			auto d_laplace_ct4 = d_laplace;
			auto d_coefs_x_ct5 = d_coefs_x;
			auto d_coefs_z_ct6 = d_coefs_z;

			cgh.parallel_for(
				sycl::nd_range<2>(dimGrid*dimBlock, dimBlock),
				[=](sycl::nd_item<2> item_ct1) {
						kernel_lap(order, nx, nz, d_p_ct3,
									d_laplace_ct4, d_coefs_x_ct5,
									d_coefs_z_ct6, item_ct1);
				});
		});
	
		/*
	 	DPCT1049:2: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/
		
		q_ct1.submit([&](sycl::handler &cgh) {
			auto d_p_ct2 = d_p;
			auto d_pp_ct3 = d_pp;
			auto d_v2_ct4 = d_v2;
			auto d_laplace_ct5 = d_laplace;
			auto dt2_ct6 = dt2;

			cgh.parallel_for(
				sycl::nd_range<2>(dimGrid * dimBlock, dimBlock),
				[=](sycl::nd_item<2> item_ct1) {
						kernel_time(nx, nz, d_p_ct2, d_pp_ct3,
									d_v2_ct4, d_laplace_ct5,
									dt2_ct6, item_ct1);
				});
		});
		
		/*
	 	DPCT1049:3: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/
		q_ct1.submit([&](sycl::handler &cgh) {
			auto d_dobs_ct1 = d_dobs;
			auto d_pp_ct2 = d_pp;
			auto it_ct3 = it; 
			auto is_ct4 = is; 
			auto nzbin_ct5 = nzbin;

			cgh.parallel_for(
				sycl::nd_range<1>(dimGridSingle * dimBlockSingle, dimBlockSingle),
				[=](sycl::nd_item<1> item_ct1) {
					kernel_sism(nx, nz, nzbin_ct5, nt, ns, it_ct3, is_ct4, 
						gz, d_pp_ct2, d_dobs_ct1, item_ct1);
				});
		});

		// /*
	 	// DPCT1049:0: The workgroup size passed to the
        //          * SYCL kernel may exceed the limit. To get the device limit,
        //          * query info::device::max_work_group_size. Adjust the workgroup
        //          * size if needed.
	 	// */

		q_ct1.submit([&](sycl::handler &cgh) {
			auto nxbin_ct2 = nxbin;
			auto nzbin_ct3 = nzbin;
			auto d_p_ct4 = d_p;
			auto d_pp_ct5 = d_pp;
			auto d_taperx_ct6 = d_taperx;
			auto d_taperz_ct7 = d_taperz;

			cgh.parallel_for(
				sycl::nd_range<2>(dimGridTaper * dimBlock,
									dimBlock),
				[=](sycl::nd_item<2> item_ct1) {
						kernel_tapper(nx, nz, nxbin_ct2, nzbin_ct3,
										d_p_ct4, d_pp_ct5,
										d_taperx_ct6, d_taperz_ct7,
										item_ct1);
				});
		}).wait();
		
		// // // /*
	 	// // // DPCT1049:0: The workgroup size passed to the
        // // //          * SYCL kernel may exceed the limit. To get the device limit,
        // // //          * query info::device::max_work_group_size. Adjust the workgroup
        // // //          * size if needed.
	 	// // // */

		q_ct1.submit([&](sycl::handler &cgh) {
			auto d_rwf_ct1 = d_rwf;
			auto d_p_ct2 = d_p;
			auto it_ct3 = it; 
			auto nxbin_ct4 = nxbin;
			auto nx_ct5 = (nx - (2*nxbin)); 
			auto nz_ct6 = (nz - (2*nzbin)); 

			cgh.parallel_for(
				sycl::nd_range<2>(dimGrid * dimBlock, dimBlock),
				[=](sycl::nd_item<2> item_ct1) {
						kernel_updt_wfd(d_rwf_ct1, d_p_ct2, 
						nx_ct5, nz_ct6, it_ct3, nxbin_ct4, 
						item_ct1);
				});
		});

		d_swap  = d_pp;
	 	d_pp = d_p;
	 	d_p = d_swap;

		if((it+1)%100 == 0){fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);fprintf(stdout,"\n");}
		
 	}
	q_ct1.memcpy(rwf[0][0], d_rwf, waveBufferLength).wait();
}