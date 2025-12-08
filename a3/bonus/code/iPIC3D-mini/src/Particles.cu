#include "Alloc.h"
#include "Particles.h"
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>

#define USE_GPU

/** allocate particle arrays */
void particle_allocate(struct parameters *param, struct particles *part,
                       int is) {

  // set species ID
  part->species_ID = is;
  // number of particles
  part->nop = param->np[is];
  // maximum number of particles
  part->npmax = param->npMax[is];

  // choose a different number of mover iterations for ions and electrons
  if (param->qom[is] < 0) { // electrons
    part->NiterMover = param->NiterMover;
    part->n_sub_cycles = param->n_sub_cycles;
  } else { // ions: only one iteration
    part->NiterMover = 1;
    part->n_sub_cycles = 1;
  }

  // particles per cell
  part->npcelx = param->npcelx[is];
  part->npcely = param->npcely[is];
  part->npcelz = param->npcelz[is];
  part->npcel = part->npcelx * part->npcely * part->npcelz;

  // cast it to required precision
  part->qom = (FPpart)param->qom[is];

  long npmax = part->npmax;

  // initialize drift and thermal velocities
  // drift
  part->u0 = (FPpart)param->u0[is];
  part->v0 = (FPpart)param->v0[is];
  part->w0 = (FPpart)param->w0[is];
  // thermal
  part->uth = (FPpart)param->uth[is];
  part->vth = (FPpart)param->vth[is];
  part->wth = (FPpart)param->wth[is];

  //////////////////////////////
  /// ALLOCATION PARTICLE ARRAYS
  //////////////////////////////
  part->x = new FPpart[npmax];
  part->y = new FPpart[npmax];
  part->z = new FPpart[npmax];
  // allocate velocity
  part->u = new FPpart[npmax];
  part->v = new FPpart[npmax];
  part->w = new FPpart[npmax];
  // allocate charge = q * statistical weight
  part->q = new FPinterp[npmax];
}

/** deallocate */
void particle_deallocate(struct particles *part) {
  // deallocate particle variables
  delete[] part->x;
  delete[] part->y;
  delete[] part->z;
  delete[] part->u;
  delete[] part->v;
  delete[] part->w;
  delete[] part->q;
}

#ifndef USE_GPU

/** particle mover */
int mover_PC(struct particles *part, struct EMfield *field, struct grid *grd,
             struct parameters *param) {
  // print species and subcycling
  std::cout << "***  MOVER with SUBCYCLYING " << param->n_sub_cycles
            << " - species " << part->species_ID << " ***" << std::endl;

  // auxiliary variables
  FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);
  FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;
  FPpart omdtsq, denom, ut, vt, wt, udotb;

  // local (to the particle) electric and magnetic field
  FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

  // interpolation densities
  int ix, iy, iz;
  FPfield weight[2][2][2];
  FPfield xi[2], eta[2], zeta[2];

  // intermediate particle position and velocity
  FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

  // start subcycling
  for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
    // move each particle with new fields
    for (int i = 0; i < part->nop; i++) {
      xptilde = part->x[i];
      yptilde = part->y[i];
      zptilde = part->z[i];
      // calculate the average velocity iteratively
      for (int innter = 0; innter < part->NiterMover; innter++) {
        // interpolation G-->P
        ix = 2 + int((part->x[i] - grd->xStart) * grd->invdx);
        iy = 2 + int((part->y[i] - grd->yStart) * grd->invdy);
        iz = 2 + int((part->z[i] - grd->zStart) * grd->invdz);

        // calculate weights
        xi[0] = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0] = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1] = grd->XN[ix][iy][iz] - part->x[i];
        eta[1] = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        for (int ii = 0; ii < 2; ii++)
          for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
              weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

        // set to zero local electric and magnetic field
        Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

        for (int ii = 0; ii < 2; ii++)
          for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
              Exl += weight[ii][jj][kk] * field->Ex[ix - ii][iy - jj][iz - kk];
              Eyl += weight[ii][jj][kk] * field->Ey[ix - ii][iy - jj][iz - kk];
              Ezl += weight[ii][jj][kk] * field->Ez[ix - ii][iy - jj][iz - kk];
              Bxl += weight[ii][jj][kk] * field->Bxn[ix - ii][iy - jj][iz - kk];
              Byl += weight[ii][jj][kk] * field->Byn[ix - ii][iy - jj][iz - kk];
              Bzl += weight[ii][jj][kk] * field->Bzn[ix - ii][iy - jj][iz - kk];
            }

        // end interpolation
        omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
        denom = 1.0 / (1.0 + omdtsq);
        // solve the position equation
        ut = part->u[i] + qomdt2 * Exl;
        vt = part->v[i] + qomdt2 * Eyl;
        wt = part->w[i] + qomdt2 * Ezl;
        udotb = ut * Bxl + vt * Byl + wt * Bzl;
        // solve the velocity equation
        uptilde = (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) *
                  denom;
        vptilde = (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) *
                  denom;
        wptilde = (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) *
                  denom;
        // update position
        part->x[i] = xptilde + uptilde * dto2;
        part->y[i] = yptilde + vptilde * dto2;
        part->z[i] = zptilde + wptilde * dto2;

      } // end of iteration
      // update the final position and velocity
      part->u[i] = 2.0 * uptilde - part->u[i];
      part->v[i] = 2.0 * vptilde - part->v[i];
      part->w[i] = 2.0 * wptilde - part->w[i];
      part->x[i] = xptilde + uptilde * dt_sub_cycling;
      part->y[i] = yptilde + vptilde * dt_sub_cycling;
      part->z[i] = zptilde + wptilde * dt_sub_cycling;

      //////////
      //////////
      ////////// BC

      // X-DIRECTION: BC particles
      if (part->x[i] > grd->Lx) {
        if (param->PERIODICX == true) { // PERIODIC
          part->x[i] = part->x[i] - grd->Lx;
        } else { // REFLECTING BC
          part->u[i] = -part->u[i];
          part->x[i] = 2 * grd->Lx - part->x[i];
        }
      }

      if (part->x[i] < 0) {
        if (param->PERIODICX == true) { // PERIODIC
          part->x[i] = part->x[i] + grd->Lx;
        } else { // REFLECTING BC
          part->u[i] = -part->u[i];
          part->x[i] = -part->x[i];
        }
      }

      // Y-DIRECTION: BC particles
      if (part->y[i] > grd->Ly) {
        if (param->PERIODICY == true) { // PERIODIC
          part->y[i] = part->y[i] - grd->Ly;
        } else { // REFLECTING BC
          part->v[i] = -part->v[i];
          part->y[i] = 2 * grd->Ly - part->y[i];
        }
      }

      if (part->y[i] < 0) {
        if (param->PERIODICY == true) { // PERIODIC
          part->y[i] = part->y[i] + grd->Ly;
        } else { // REFLECTING BC
          part->v[i] = -part->v[i];
          part->y[i] = -part->y[i];
        }
      }

      // Z-DIRECTION: BC particles
      if (part->z[i] > grd->Lz) {
        if (param->PERIODICZ == true) { // PERIODIC
          part->z[i] = part->z[i] - grd->Lz;
        } else { // REFLECTING BC
          part->w[i] = -part->w[i];
          part->z[i] = 2 * grd->Lz - part->z[i];
        }
      }

      if (part->z[i] < 0) {
        if (param->PERIODICZ == true) { // PERIODIC
          part->z[i] = part->z[i] + grd->Lz;
        } else { // REFLECTING BC
          part->w[i] = -part->w[i];
          part->z[i] = -part->z[i];
        }
      }

    } // end of subcycling
  } // end of one particle

  return (0); // exit succcesfully
} // end of the mover
#endif

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles *part, struct interpDensSpecies *ids,
               struct grid *grd) {

  // arrays needed for interpolation
  FPpart weight[2][2][2];
  FPpart temp[2][2][2];
  FPpart xi[2], eta[2], zeta[2];

  // index of the cell
  int ix, iy, iz;

  for (register long long i = 0; i < part->nop; i++) {

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((part->x[i] - grd->xStart) * grd->invdx));
    iy = 2 + int(floor((part->y[i] - grd->yStart) * grd->invdy));
    iz = 2 + int(floor((part->z[i] - grd->zStart) * grd->invdz));

    // distances from node
    xi[0] = part->x[i] - grd->XN[ix - 1][iy][iz];
    eta[0] = part->y[i] - grd->YN[ix][iy - 1][iz];
    zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
    xi[1] = grd->XN[ix][iy][iz] - part->x[i];
    eta[1] = grd->YN[ix][iy][iz] - part->y[i];
    zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];

    // calculate the weights for different nodes
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          weight[ii][jj][kk] =
              part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

    //////////////////////////
    // add charge density
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->rhon[ix - ii][iy - jj][iz - kk] +=
              weight[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add current density - Jx
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add current density - Jy
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add current density - Jz
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add pressure pxx
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    ////////////////////////////
    // add pressure pxy
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    /////////////////////////////
    // add pressure pxz
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    /////////////////////////////
    // add pressure pyy
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    /////////////////////////////
    // add pressure pyz
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    /////////////////////////////
    // add pressure pzz
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++)
        for (int kk = 0; kk < 2; kk++)
          ids->pzz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
  }
}

#ifdef USE_GPU
__global__ void move_particle_gpu(struct particles *part, struct EMfield *field,
                                  struct grid *grd, struct parameters *param) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= part->nop)
    return;

// Helper macro to convert 3D indices to flat 1D index
// For a grid of size nxn x nyn x nzn, index is: i*nyn*nzn + j*nzn + k
#define IDX(i, j, k) ((i) * grd->nyn * grd->nzn + (j) * grd->nzn + (k))

  // auxiliary variables
  FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);
  FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;
  FPpart omdtsq, denom, ut, vt, wt, udotb;

  // local (to the particle) electric and magnetic field
  FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

  // interpolation densities
  int ix, iy, iz;
  FPfield weight[2][2][2];
  FPfield xi[2], eta[2], zeta[2];

  // intermediate particle position and velocity
  FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

  // start subcycling
  for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
    xptilde = part->x[i];
    yptilde = part->y[i];
    zptilde = part->z[i];
    // calculate the average velocity iteratively
    for (int innter = 0; innter < part->NiterMover; innter++) {
      // interpolation G-->P
      ix = 2 + int((part->x[i] - grd->xStart) * grd->invdx);
      iy = 2 + int((part->y[i] - grd->yStart) * grd->invdy);
      iz = 2 + int((part->z[i] - grd->zStart) * grd->invdz);

      // calculate weights
      xi[0] = part->x[i] - grd->XN_flat[IDX(ix - 1, iy, iz)];
      eta[0] = part->y[i] - grd->YN_flat[IDX(ix, iy - 1, iz)];
      zeta[0] = part->z[i] - grd->ZN_flat[IDX(ix, iy, iz - 1)];
      xi[1] = grd->XN_flat[IDX(ix, iy, iz)] - part->x[i];
      eta[1] = grd->YN_flat[IDX(ix, iy, iz)] - part->y[i];
      zeta[1] = grd->ZN_flat[IDX(ix, iy, iz)] - part->z[i];
      for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
          for (int kk = 0; kk < 2; kk++)
            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

      // set to zero local electric and magnetic field
      Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

      for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
          for (int kk = 0; kk < 2; kk++) {
            Exl += weight[ii][jj][kk] *
                   field->Ex_flat[IDX(ix - ii, iy - jj, iz - kk)];
            Eyl += weight[ii][jj][kk] *
                   field->Ey_flat[IDX(ix - ii, iy - jj, iz - kk)];
            Ezl += weight[ii][jj][kk] *
                   field->Ez_flat[IDX(ix - ii, iy - jj, iz - kk)];
            Bxl += weight[ii][jj][kk] *
                   field->Bxn_flat[IDX(ix - ii, iy - jj, iz - kk)];
            Byl += weight[ii][jj][kk] *
                   field->Byn_flat[IDX(ix - ii, iy - jj, iz - kk)];
            Bzl += weight[ii][jj][kk] *
                   field->Bzn_flat[IDX(ix - ii, iy - jj, iz - kk)];
          }

      // end interpolation
      omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
      denom = 1.0 / (1.0 + omdtsq);
      // solve the position equation
      ut = part->u[i] + qomdt2 * Exl;
      vt = part->v[i] + qomdt2 * Eyl;
      wt = part->w[i] + qomdt2 * Ezl;
      udotb = ut * Bxl + vt * Byl + wt * Bzl;
      // solve the velocity equation
      uptilde =
          (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
      vptilde =
          (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
      wptilde =
          (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;
      // update position
      part->x[i] = xptilde + uptilde * dto2;
      part->y[i] = yptilde + vptilde * dto2;
      part->z[i] = zptilde + wptilde * dto2;

    } // end of iteration
    // update the final position and velocity
    part->u[i] = 2.0 * uptilde - part->u[i];
    part->v[i] = 2.0 * vptilde - part->v[i];
    part->w[i] = 2.0 * wptilde - part->w[i];
    part->x[i] = xptilde + uptilde * dt_sub_cycling;
    part->y[i] = yptilde + vptilde * dt_sub_cycling;
    part->z[i] = zptilde + wptilde * dt_sub_cycling;

    //////////
    //////////
    ////////// BC

    // X-DIRECTION: BC particles
    if (part->x[i] > grd->Lx) {
      if (param->PERIODICX == true) { // PERIODIC
        part->x[i] = part->x[i] - grd->Lx;
      } else { // REFLECTING BC
        part->u[i] = -part->u[i];
        part->x[i] = 2 * grd->Lx - part->x[i];
      }
    }

    if (part->x[i] < 0) {
      if (param->PERIODICX == true) { // PERIODIC
        part->x[i] = part->x[i] + grd->Lx;
      } else { // REFLECTING BC
        part->u[i] = -part->u[i];
        part->x[i] = -part->x[i];
      }
    }

    // Y-DIRECTION: BC particles
    if (part->y[i] > grd->Ly) {
      if (param->PERIODICY == true) { // PERIODIC
        part->y[i] = part->y[i] - grd->Ly;
      } else { // REFLECTING BC
        part->v[i] = -part->v[i];
        part->y[i] = 2 * grd->Ly - part->y[i];
      }
    }

    if (part->y[i] < 0) {
      if (param->PERIODICY == true) { // PERIODIC
        part->y[i] = part->y[i] + grd->Ly;
      } else { // REFLECTING BC
        part->v[i] = -part->v[i];
        part->y[i] = -part->y[i];
      }
    }

    // Z-DIRECTION: BC particles
    if (part->z[i] > grd->Lz) {
      if (param->PERIODICZ == true) { // PERIODIC
        part->z[i] = part->z[i] - grd->Lz;
      } else { // REFLECTING BC
        part->w[i] = -part->w[i];
        part->z[i] = 2 * grd->Lz - part->z[i];
      }
    }

    if (part->z[i] < 0) {
      if (param->PERIODICZ == true) { // PERIODIC
        part->z[i] = part->z[i] + grd->Lz;
      } else { // REFLECTING BC
        part->w[i] = -part->w[i];
        part->z[i] = -part->z[i];
      }
    }
  }

#undef IDX
}

int mover_PC_gpu(struct particles *part, struct EMfield *field,
                 struct grid *grd, struct parameters *param) {
  // print species and subcycling
  std::cout << "***  MOVER with SUBCYCLYING " << param->n_sub_cycles
            << " - species " << part->species_ID << " ***" << std::endl;

  int TPB = 1024;
  int NBlocks = 1 + ((part->nop - 1) / TPB); // assumes part->nop != 0

  int size_part = part->nop * sizeof(FPpart);
  int size_grd = grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield);

  struct particles *d_part = nullptr;
  struct EMfield *d_field = nullptr;
  struct grid *d_grd = nullptr;
  struct parameters *d_param = nullptr;

  struct particles l_part = *part;
  struct EMfield l_field = *field;
  struct grid l_grd = *grd;

  FPpart *d_part_x = nullptr;
  FPpart *d_part_y = nullptr;
  FPpart *d_part_z = nullptr;
  FPpart *d_part_u = nullptr;
  FPpart *d_part_v = nullptr;
  FPpart *d_part_w = nullptr;

  cudaMalloc(&d_part_x, size_part);
  cudaMalloc(&d_part_y, size_part);
  cudaMalloc(&d_part_z, size_part);
  cudaMalloc(&d_part_u, size_part);
  cudaMalloc(&d_part_v, size_part);
  cudaMalloc(&d_part_w, size_part);

  cudaMemcpy(d_part_x, part->x, size_part, cudaMemcpyHostToDevice);
  cudaMemcpy(d_part_y, part->y, size_part, cudaMemcpyHostToDevice);
  cudaMemcpy(d_part_z, part->z, size_part, cudaMemcpyHostToDevice);
  cudaMemcpy(d_part_u, part->u, size_part, cudaMemcpyHostToDevice);
  cudaMemcpy(d_part_v, part->v, size_part, cudaMemcpyHostToDevice);
  cudaMemcpy(d_part_w, part->w, size_part, cudaMemcpyHostToDevice);

  l_part.x = d_part_x;
  l_part.y = d_part_y;
  l_part.z = d_part_z;
  l_part.u = d_part_u;
  l_part.v = d_part_v;
  l_part.w = d_part_w;

  FPfield *d_grd_XN_flat = nullptr;
  FPfield *d_grd_YN_flat = nullptr;
  FPfield *d_grd_ZN_flat = nullptr;

  cudaMalloc(&d_grd_XN_flat, size_grd);
  cudaMalloc(&d_grd_YN_flat, size_grd);
  cudaMalloc(&d_grd_ZN_flat, size_grd);

  cudaMemcpy(d_grd_XN_flat, grd->XN_flat, size_grd, cudaMemcpyHostToDevice);
  cudaMemcpy(d_grd_YN_flat, grd->YN_flat, size_grd, cudaMemcpyHostToDevice);
  cudaMemcpy(d_grd_ZN_flat, grd->ZN_flat, size_grd, cudaMemcpyHostToDevice);

  l_grd.XN_flat = d_grd_XN_flat;
  l_grd.YN_flat = d_grd_YN_flat;
  l_grd.ZN_flat = d_grd_ZN_flat;

  FPfield *d_field_Ex_flat = nullptr;
  FPfield *d_field_Ey_flat = nullptr;
  FPfield *d_field_Ez_flat = nullptr;
  FPfield *d_field_Bxn_flat = nullptr;
  FPfield *d_field_Byn_flat = nullptr;
  FPfield *d_field_Bzn_flat = nullptr;

  cudaMalloc(&d_field_Ex_flat, size_grd);
  cudaMalloc(&d_field_Ey_flat, size_grd);
  cudaMalloc(&d_field_Ez_flat, size_grd);
  cudaMalloc(&d_field_Bxn_flat, size_grd);
  cudaMalloc(&d_field_Byn_flat, size_grd);
  cudaMalloc(&d_field_Bzn_flat, size_grd);

  cudaMemcpy(d_field_Ex_flat, field->Ex_flat, size_grd, cudaMemcpyHostToDevice);
  cudaMemcpy(d_field_Ey_flat, field->Ey_flat, size_grd, cudaMemcpyHostToDevice);
  cudaMemcpy(d_field_Ez_flat, field->Ez_flat, size_grd, cudaMemcpyHostToDevice);
  cudaMemcpy(d_field_Bxn_flat, field->Bxn_flat, size_grd,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_field_Byn_flat, field->Byn_flat, size_grd,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_field_Bzn_flat, field->Bzn_flat, size_grd,
             cudaMemcpyHostToDevice);

  l_field.Ex_flat = d_field_Ex_flat;
  l_field.Ey_flat = d_field_Ey_flat;
  l_field.Ez_flat = d_field_Ez_flat;
  l_field.Bxn_flat = d_field_Bxn_flat;
  l_field.Byn_flat = d_field_Byn_flat;
  l_field.Bzn_flat = d_field_Bzn_flat;

  cudaMalloc(&d_part, sizeof(struct particles));
  cudaMalloc(&d_field, sizeof(struct EMfield));
  cudaMalloc(&d_grd, sizeof(struct grid));
  cudaMalloc(&d_param, sizeof(struct parameters));

  cudaMemcpy(d_part, &l_part, sizeof(struct particles), cudaMemcpyHostToDevice);
  cudaMemcpy(d_grd, &l_grd, sizeof(struct grid), cudaMemcpyHostToDevice);
  cudaMemcpy(d_field, &l_field, sizeof(struct EMfield), cudaMemcpyHostToDevice);
  cudaMemcpy(d_param, param, sizeof(struct parameters), cudaMemcpyHostToDevice);

  move_particle_gpu<<<NBlocks, TPB>>>(d_part, d_field, d_grd, d_param);

  cudaMemcpy(part->x, d_part_x, size_part, cudaMemcpyDeviceToHost);
  cudaMemcpy(part->y, d_part_y, size_part, cudaMemcpyDeviceToHost);
  cudaMemcpy(part->z, d_part_z, size_part, cudaMemcpyDeviceToHost);
  cudaMemcpy(part->u, d_part_u, size_part, cudaMemcpyDeviceToHost);
  cudaMemcpy(part->v, d_part_v, size_part, cudaMemcpyDeviceToHost);
  cudaMemcpy(part->w, d_part_w, size_part, cudaMemcpyDeviceToHost);

  cudaFree(d_part_x);
  cudaFree(d_part_y);
  cudaFree(d_part_z);
  cudaFree(d_part_u);
  cudaFree(d_part_v);
  cudaFree(d_part_w);

  cudaFree(d_grd_XN_flat);
  cudaFree(d_grd_YN_flat);
  cudaFree(d_grd_ZN_flat);

  cudaFree(d_field_Ex_flat);
  cudaFree(d_field_Ey_flat);
  cudaFree(d_field_Ez_flat);
  cudaFree(d_field_Bxn_flat);
  cudaFree(d_field_Byn_flat);
  cudaFree(d_field_Bzn_flat);

  cudaFree(d_part);
  cudaFree(d_field);
  cudaFree(d_grd);
  cudaFree(d_param);

  return 0;
}
#endif
