#include<iostream>

#include "cwp.h"
#include "fd.h"

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

using namespace std; 
using namespace sycl; 

#define NGPUS 2 
#define SIZEBLOCK 16 

static void makeo2 (float *coef,int order);

std::vector<sycl::queue> qs;

float *d_p[2], *d_pp[2], *d_swap[2];
float *d_laplace[2], *d_v2[2], *d_coefs_x[2], *d_coefs_z[2];
float *d_taperx[2], *d_taperz[2], *d_swf[2], *d_rwf[2], *d_dobs[2];

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
	// Discover GPUs 

  	auto platforms = sycl::platform::get_platforms();
	for (auto & p : platforms) {
		auto pname = p.get_info<sycl::info::platform::name>();
		auto devices = p.get_devices();
		for (auto & d : devices ) {
			if ( (d.is_gpu() || d.is_cpu()) && qs.size() < NGPUS) {
				std::cout << "Platform: " << pname << std::endl;
				std::cout << " Device: " << d.get_info<sycl::info::device::name>() << std::endl;
				qs.push_back(sycl::queue(d));
			}
		}
	}
	
	if (qs.size() != NGPUS){
		std::cout << "Number of GPUs insuficient, platform only has " << qs.size() << " but is necessary 2 or more." << std::endl; 
		exit(0); 
	}

	// End Discover 
	
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

	// Create a Device pointers for both GPUs
	for(int i =0; i<NGPUS; i++){
		d_p[i] = (float *)sycl::malloc_device(mtxBufferLength, qs[i]);
		d_pp[i] = (float *)sycl::malloc_device(mtxBufferLength, qs[i]);
		
		d_v2[i] = (float *)sycl::malloc_device(mtxBufferLength, qs[i]);
		
		d_swap[i] = (float *)sycl::malloc_device(mtxBufferLength, qs[i]);
		d_laplace[i] = (float *)sycl::malloc_device(mtxBufferLength, qs[i]);

		d_rwf[i] = (float *)sycl::malloc_device(waveBufferLength, qs[i]);
		d_swf[i] = (float *)sycl::malloc_device(waveBufferLength, qs[i]);
		d_dobs[i] = (float *)sycl::malloc_device(dobsBufferLength, qs[i]);
		
		d_coefs_x[i] = (float *)sycl::malloc_device(coefsBufferLength, qs[i]);
		d_coefs_z[i] = (float *)sycl::malloc_device(coefsBufferLength, qs[i]);
		
		d_taperx[i] = (float *)sycl::malloc_device(brdBufferLength, qs[i]);
		d_taperz[i] = (float *)sycl::malloc_device(brdBufferLength, qs[i]);
	}
	

	int div_x, div_z;
	
	// Set a Grid for the execution on the device
	// The grid size has to be the next multiple nearest the original grid size
	
	gridx = (float) (((nxe / SIZEBLOCK) + 1) * SIZEBLOCK)/(float) SIZEBLOCK;
	gridz = (float) (((nze / SIZEBLOCK) + 1) * SIZEBLOCK)/(float) SIZEBLOCK;
	gridBorder_x = (float) (((nxb / SIZEBLOCK) + 1) * SIZEBLOCK)/(float) SIZEBLOCK;
	gridBorder_z = (float) (((nzb / SIZEBLOCK) + 1) * SIZEBLOCK)/(float) SIZEBLOCK;
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
	// gettimeofday(&startCopyMem, NULL);
		qs[flag].memcpy(d_p[flag], p[0], mtxBufferLength).wait();
		qs[flag].memcpy(d_pp[flag], pp[0], mtxBufferLength).wait();
		qs[flag].memcpy(d_v2[flag], v2[0], mtxBufferLength).wait();
		qs[flag].memcpy(d_laplace[flag], laplace[0], mtxBufferLength).wait();
		
		qs[flag].memcpy(d_coefs_x[flag], coefs_x, coefsBufferLength).wait();
		qs[flag].memcpy(d_coefs_z[flag], coefs_z, coefsBufferLength).wait();
		
		qs[flag].memcpy(d_taperx[flag], taperx, brdBufferLength).wait();
		qs[flag].memcpy(d_taperz[flag], taperz, brdBufferLength).wait();
	
	if(flag == 0)
		qs[flag].memcpy(d_swf[flag], wf[0][0], waveBufferLength).wait();
	if(flag == 1){
			qs[flag].memcpy(d_rwf[flag], wf[0][0], waveBufferLength).wait();
		qs[flag].memcpy(d_dobs[flag], dobs[0][0], dobsBufferLength).wait();
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
			   int nx, int nz, int nt, int is, int sz, int *sx, float *srce, int flag){
	
	sycl::range<2> dimGrid(gridx, gridz);
	sycl::range<2> dimGridTaper(gridx, gridBorder_z);

	sycl::range<2> dimBlock(SIZEBLOCK, SIZEBLOCK);
	write_buffers(p, pp, v2, taper_x, taper_z, NULL, swf, 0);
	   	
   	for (int it = 0; it < nt; it++){
		/*
	 	DPCT1049:1: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/

		qs[flag].submit([&](sycl::handler &cgh) {
			auto d_p_ct3 = d_p[flag];
			auto d_laplace_ct4 = d_laplace[flag];
			auto d_coefs_x_ct5 = d_coefs_x[flag];
			auto d_coefs_z_ct6 = d_coefs_z[flag];

			cgh.parallel_for(
				sycl::nd_range<2>(dimGrid*dimBlock, dimBlock),
				[=](sycl::nd_item<2> item_ct1) {
						kernel_lap(order, nx, nz, d_p_ct3,
									d_laplace_ct4, d_coefs_x_ct5,
									d_coefs_z_ct6, item_ct1);
				});
		}).wait();
	
		/*
	 	DPCT1049:2: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/
		
		qs[flag].submit([&](sycl::handler &cgh) {
			auto d_p_ct2 = d_p[flag];
			auto d_pp_ct3 = d_pp[flag];
			auto d_v2_ct4 = d_v2[flag];
			auto d_laplace_ct5 = d_laplace[flag];
			auto dt2_ct6 = dt2;

			cgh.parallel_for(
				sycl::nd_range<2>(dimGrid * dimBlock, dimBlock),
				[=](sycl::nd_item<2> item_ct1) {
						kernel_time(nx, nz, d_p_ct2, d_pp_ct3,
									d_v2_ct4, d_laplace_ct5,
									dt2_ct6, item_ct1);
				});
		}).wait();
		
		/*
	 	DPCT1049:3: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/
		qs[flag].submit([&](sycl::handler &cgh) {
			auto d_pp_ct1 = d_pp[flag];
			auto sx_is_ct2 = sx[is];
			auto srce_it_ct4 = srce[it];

			cgh.single_task([=]() {
					kernel_src(
					nz, d_pp_ct1, sx_is_ct2, sz	,srce_it_ct4);
				});
		}).wait();

		/*
	 	DPCT1049:0: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/

		qs[flag].submit([&](sycl::handler &cgh) {
			auto nxbin_ct2 = nxbin;
			auto nzbin_ct3 = nzbin;
			auto d_p_ct4 = d_p[flag];
			auto d_pp_ct5 = d_pp[flag];
			auto d_taperx_ct6 = d_taperx[flag];
			auto d_taperz_ct7 = d_taperz[flag];

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

		/*
	 	DPCT1049:0: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/

		qs[flag].submit([&](sycl::handler &cgh) {
			auto d_swf_ct1 = d_swf[flag];
			auto d_p_ct2 = d_p[flag];
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
		}).wait();
      

		d_swap[flag]  = d_pp[flag];
	 	d_pp[flag] = d_p[flag];
	 	d_p[flag] = d_swap[flag];

		// if((it+1)%100 == 0){fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);fprintf(stdout,"\n");}
		
 	}
	// qs[flag].memcpy(p[0], d_p, mtxBufferLength).wait();
	// qs[flag].memcpy(pp[0], d_pp, mtxBufferLength).wait();
	qs[flag].memcpy(swf[0][0], d_swf[flag], waveBufferLength).wait();
}

void fd_backward(int order, float **p, float **pp, float **v2, float ***rwf, float ***dobs,  
			   int nx, int nz, int nt, int ns, int gz, int is, int it, int sz, int *sx, float *srce, int flag){
	
	sycl::range<1> dimGridSingle(gridx);
	sycl::range<2> dimGrid(gridx, gridz);
	sycl::range<2> dimGridTaper(gridx, gridBorder_z);

	sycl::range<1> dimBlockSingle(SIZEBLOCK);
	sycl::range<2> dimBlock(SIZEBLOCK, SIZEBLOCK);
	write_buffers(p, pp, v2, taper_x, taper_z, dobs, rwf, 1);
	   	
   	for (int it = 0; it < nt; it++){
		/*
	 	DPCT1049:1: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/

		qs[flag].submit([&](sycl::handler &cgh) {
			auto d_p_ct3 = d_p[flag];
			auto d_laplace_ct4 = d_laplace[flag];
			auto d_coefs_x_ct5 = d_coefs_x[flag];
			auto d_coefs_z_ct6 = d_coefs_z[flag];

			cgh.parallel_for(
				sycl::nd_range<2>(dimGrid*dimBlock, dimBlock),
				[=](sycl::nd_item<2> item_ct1) {
						kernel_lap(order, nx, nz, d_p_ct3,
									d_laplace_ct4, d_coefs_x_ct5,
									d_coefs_z_ct6, item_ct1);
				});
		}).wait();;
	
		/*
	 	DPCT1049:2: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/
		
		qs[flag].submit([&](sycl::handler &cgh) {
			auto d_p_ct2 = d_p[flag];
			auto d_pp_ct3 = d_pp[flag];
			auto d_v2_ct4 = d_v2[flag];
			auto d_laplace_ct5 = d_laplace[flag];
			auto dt2_ct6 = dt2;

			cgh.parallel_for(
				sycl::nd_range<2>(dimGrid * dimBlock, dimBlock),
				[=](sycl::nd_item<2> item_ct1) {
						kernel_time(nx, nz, d_p_ct2, d_pp_ct3,
									d_v2_ct4, d_laplace_ct5,
									dt2_ct6, item_ct1);
				});
		}).wait();;
		
		/*
	 	DPCT1049:3: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/
		qs[flag].submit([&](sycl::handler &cgh) {
			auto d_dobs_ct1 = d_dobs[flag];
			auto d_pp_ct2 = d_pp[flag];
			auto it_ct3 = it; 
			auto is_ct4 = is; 
			auto nzbin_ct5 = nzbin;

			cgh.parallel_for(
				sycl::nd_range<1>(dimGridSingle * dimBlockSingle, dimBlockSingle),
				[=](sycl::nd_item<1> item_ct1) {
					kernel_sism(nx, nz, nzbin_ct5, nt, ns, it_ct3, is_ct4, 
						gz, d_pp_ct2, d_dobs_ct1, item_ct1);
				});
		}).wait();;

		/*
	 	DPCT1049:0: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/

		qs[flag].submit([&](sycl::handler &cgh) {
			auto nxbin_ct2 = nxbin;
			auto nzbin_ct3 = nzbin;
			auto d_p_ct4 = d_p[flag];
			auto d_pp_ct5 = d_pp[flag];
			auto d_taperx_ct6 = d_taperx[flag];
			auto d_taperz_ct7 = d_taperz[flag];

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
		
		/*
	 	DPCT1049:0: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
	 	*/

		qs[flag].submit([&](sycl::handler &cgh) {
			auto d_rwf_ct1 = d_rwf[flag];
			auto d_p_ct2 = d_p[flag];
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
		}).wait();;

		d_swap[flag] = d_pp[flag];
	 	d_pp[flag] = d_p[flag];
	 	d_p[flag] = d_swap[flag];

		// if((it+1)%100 == 0){fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);fprintf(stdout,"\n");}
		
 	}
	qs[flag].memcpy(rwf[0][0], d_rwf[flag], waveBufferLength).wait();
}