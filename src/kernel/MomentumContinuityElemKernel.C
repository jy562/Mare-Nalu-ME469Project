/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumContinuityElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<class AlgTraits>
MomentumContinuityElemKernel<AlgTraits>::MomentumContinuityElemKernel(
  const stk::mesh::BulkData &bulkData,
  const SolutionOptions &solnOpts,
  VectorFieldType *velocity,
  ScalarFieldType *density,
  ScalarFieldType *viscosity,
  const bool lumpedMass,
  ElemDataRequests &dataPreReqs)
  : Kernel(),
    viscosity_(viscosity),
    includeDivU_(solnOpts.includeDivU_),
    lrscv_(sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_)->adjacentNodes()),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_)->ipNodeMap()),
    dofSize_(AlgTraits::nDim_+1)
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));
  densityN_ = &(density->field_of_state(stk::mesh::StateN));
  if (density->number_of_states() == 2)
    densityNm1_ = densityN_;
  else
    densityNm1_ = &(density->field_of_state(stk::mesh::StateNM1));

  velocityNp1_ = &(velocity->field_of_state(stk::mesh::StateNP1));
  velocityN_ = &(velocity->field_of_state(stk::mesh::StateN));
  if (velocity->number_of_states() == 2)
    velocityNm1_ = velocityN_;
  else
    velocityNm1_ = &(velocity->field_of_state(stk::mesh::StateNM1));
  
  Gjp_ = metaData.get_field<double>(
    stk::topology::NODE_RANK, "dpdx");
  GjpOld_ = metaData.get_field<double>(
    stk::topology::NODE_RANK, "dpdx_old");
  
  coordinates_ = metaData.get_field<double>(
    stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  pressure_ = metaData.get_field<double>(
    stk::topology::NODE_RANK, "pressure");

  // fields and data; mdot not gathered as element data
  dataPreReqs.add_gathered_nodal_field(*densityNm1_, 1);
  dataPreReqs.add_gathered_nodal_field(*densityN_, 1);
  dataPreReqs.add_gathered_nodal_field(*densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*velocityN_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*velocityNm1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*Gjp_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*GjpOld_, AlgTraits::nDim_);
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(*pressure_, 1);
  dataPreReqs.add_gathered_nodal_field(*viscosity_, 1);
  
  // master element registrations; surface
  MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_);
  dataPreReqs.add_cvfem_surface_me(meSCS);
  dataPreReqs.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(SCS_GRAD_OP, CURRENT_COORDINATES);

  // master element registrations; volume
  MasterElement* meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_);
  dataPreReqs.add_cvfem_volume_me(meSCV);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);

  // compute shape functions
  get_scs_shape_fn_data<AlgTraits>([&](double* ptr){meSCS->shape_fcn(ptr);}, v_shape_function_scs_);
  if ( lumpedMass )
    get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shifted_shape_fcn(ptr);}, v_shape_function_scv_);
  else
    get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shape_fcn(ptr);}, v_shape_function_scv_);
  
  // error checks - we are not supporting skew symmetric or shifted grad-ops
  const bool skewSymmetric = solnOpts.get_skew_symmetric(velocity->name());
  const bool shiftedGradOpU = solnOpts.get_shifted_grad_op(velocity->name());
  const bool shiftedGradOpP = solnOpts.get_shifted_grad_op(pressure_->name());
  if ( skewSymmetric || shiftedGradOpU || shiftedGradOpP )
    throw std::runtime_error("MomentumContinuityElemKernel::error: skewSymmetric || shiftedGradOpU || shiftedGradOpP");

  /* Notes:

  Matrix layout is in row major. For a npe = 4 (quad) and nDim = 2: dofSize_ = 2 + 1 = 3

  RHS = (resUx0, resUy0, resP0, resUx1, resUy1, resP1, resUx2, resUy2, resP2, resUx3, resUy3, resP3)

  where Uik = velocity_i_node_k and Pk = pressure_node_k

  The LHS is, therefore,

  row 0: d/dUx0(ResUx0), d/dUy0(ResUx0), d/dP0(ResUx0), ... , d/dUx3(ResUx0), d/dUy3(ResUx0), d/dP3(ResP0)

  */

}
  
template<class AlgTraits>
MomentumContinuityElemKernel<AlgTraits>::~MomentumContinuityElemKernel()
{}

template<typename AlgTraits>
void
MomentumContinuityElemKernel<AlgTraits>::setup(const TimeIntegrator& timeIntegrator)
{
  dt_ = timeIntegrator.get_time_step();
  gamma1_ = timeIntegrator.get_gamma1();
  gamma2_ = timeIntegrator.get_gamma2();
  gamma3_ = timeIntegrator.get_gamma3();
  projTimeScale_ = dt_/gamma1_;
}
  
template<class AlgTraits>
void
MomentumContinuityElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType **>& lhs,
  SharedMemView<DoubleType *>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  DoubleType w_uNm1Ip[AlgTraits::nDim_];
  DoubleType w_uNIp[AlgTraits::nDim_];
  DoubleType w_uNp1Ip[AlgTraits::nDim_];
  DoubleType w_rhoUNp1Ip[AlgTraits::nDim_];
  DoubleType w_rhoUNIp[AlgTraits::nDim_];
  DoubleType w_dpdxIp[AlgTraits::nDim_];
  DoubleType w_GjpIp[AlgTraits::nDim_];
  DoubleType w_GjpOldIp[AlgTraits::nDim_];

  // add scalings that allow the precise option being run
  const double includePstabInMom = 1.0;
  const double includeGjp = 1.0;
  const double includeOld = 1.0;
  const double om_includeOld = 1.0 - includeOld;
  
  // all required fields
  SharedMemView<DoubleType*>& v_densityNm1 = scratchViews.get_scratch_view_1D(*densityNm1_);
  SharedMemView<DoubleType*>& v_densityN = scratchViews.get_scratch_view_1D(*densityN_);
  SharedMemView<DoubleType*>& v_densityNp1 = scratchViews.get_scratch_view_1D(*densityNp1_);
  SharedMemView<DoubleType**>& v_velocityNm1 = scratchViews.get_scratch_view_2D(*velocityNm1_);
  SharedMemView<DoubleType**>& v_velocityN = scratchViews.get_scratch_view_2D(*velocityN_);
  SharedMemView<DoubleType**>& v_velocityNp1 = scratchViews.get_scratch_view_2D(*velocityNp1_);
  SharedMemView<DoubleType**>& v_Gjp = scratchViews.get_scratch_view_2D(*Gjp_);
  SharedMemView<DoubleType**>& v_GjpOld = scratchViews.get_scratch_view_2D(*GjpOld_);
  SharedMemView<DoubleType*>& v_pressure = scratchViews.get_scratch_view_1D(*pressure_);
  SharedMemView<DoubleType*>& v_viscosity = scratchViews.get_scratch_view_1D(*viscosity_);

  // volume
  SharedMemView<DoubleType*>& v_scv_volume = scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;
  
  // surface
  SharedMemView<DoubleType**>& v_scs_areav = scratchViews.get_me_views(CURRENT_COORDINATES).scs_areav;
  SharedMemView<DoubleType***>& v_dndx = scratchViews.get_me_views(CURRENT_COORDINATES).dndx;
    
  //==========================================================
  // volume (time) terms first
  //==========================================================

  for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {
    const int nearestNode = ipNodeMap_[ip];

    DoubleType rhoNm1Ip = 0.0;
    DoubleType rhoNIp   = 0.0;
    DoubleType rhoNp1Ip = 0.0;
    for (int j = 0; j < AlgTraits::nDim_; j++) {
      w_uNm1Ip[j] = 0.0;
      w_uNIp[j] = 0.0;
      w_uNp1Ip[j] = 0.0;
    }
    
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {
      const DoubleType r = v_shape_function_scv_(ip, ic);
      rhoNm1Ip += r * v_densityNm1(ic);
      rhoNIp   += r * v_densityN(ic);
      rhoNp1Ip += r * v_densityNp1(ic);
      for (int j = 0; j < AlgTraits::nDim_; j++) {
        w_uNm1Ip[j] += r*v_velocityNm1(ic,j);
        w_uNIp[j]   += r*v_velocityN(ic,j);
        w_uNp1Ip[j] += r*v_velocityNp1(ic,j);
      }
    }
    
    const DoubleType scV = v_scv_volume(ip);
    const int nnDofSize = nearestNode*dofSize_;

    // RHS (time, ui)
    for (int j = 0; j < AlgTraits::nDim_; ++j) {
      rhs(nnDofSize + j) -=
        (gamma1_*rhoNp1Ip*w_uNp1Ip[j] + gamma2_*rhoNIp*w_uNIp[j] + gamma3_*rhoNm1Ip*w_uNm1Ip[j])*scV/dt_;
    }
    
    // RHS (time, p)
    rhs(nnDofSize+AlgTraits::nDim_) -=
      (gamma1_*rhoNp1Ip + gamma2_*rhoNIp + gamma3_*rhoNm1Ip)*scV/dt_;
    
    // Compute LHS (ui only)
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {
      const int icDofSize = ic*dofSize_;
      const DoubleType r = v_shape_function_scv_(ip, ic);
      const DoubleType lhsfac = r*gamma1_*rhoNp1Ip*scV/dt_;
      for (int j = 0; j < AlgTraits::nDim_; ++j) {
        const int indexNN = nnDofSize + j;
        lhs(indexNN,icDofSize+j) += lhsfac;
      }
    }  
  }
  
  //==========================================================
  // surface (adv+pressure, diffusion) second
  //==========================================================

  for ( int ip = 0; ip < AlgTraits::numScsIp_; ++ip ) {

    // left and right nodes for this ip
    const int il = lrscv_[2*ip];
    const int ir = lrscv_[2*ip+1];
    
    // save off some offsets
    const int ilDofSize = il*dofSize_;
    const int irDofSize = ir*dofSize_;
    
    // compute scs integration point values
    DoubleType pIp = 0.0;
    DoubleType muIp = 0.0;
    DoubleType divU = 0.0;
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
      w_uNp1Ip[i] = 0.0;
      w_rhoUNp1Ip[i] = 0.0;
      w_rhoUNIp[i] = 0.0;
      w_dpdxIp[i] = 0.0;
      w_GjpIp[i] = 0.0;
      w_GjpOldIp[i] = 0.0;
    }
    
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

      const int icDofSize = ic*dofSize_;

      // save off Ic values
      const DoubleType pIc = v_pressure(ic);
      const DoubleType rhoIc = v_densityNp1(ic);

      const DoubleType r = v_shape_function_scs_(ip,ic);
      pIp += r*pIc;
      muIp += r*v_viscosity(ic);

      DoubleType lhsfacCont = 0.0;
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        const DoubleType uj = v_velocityNp1(ic,j);
        const DoubleType ujN = v_velocityN(ic,j);
        w_uNp1Ip[j] += r*uj;
        w_rhoUNp1Ip[j] += r*rhoIc*uj;
        w_rhoUNIp[j] += r*rhoIc*ujN;
        divU += uj*v_dndx(ip,ic,j);
        w_dpdxIp[j] += pIc*v_dndx(ip,ic,j);
        w_GjpIp[j] += r*v_Gjp(ic,j);
        w_GjpOldIp[j] += r*v_GjpOld(ic,j);
        lhsfacCont += -v_dndx(ip,ic,j)*v_scs_areav(ip,j);
      }

      // [1] d(Res^P)/dP; d/dP(-projTimeScale_*dp/dxj*nj*dS); (do not normalize by projTimeScale_)
      lhs(ilDofSize+AlgTraits::nDim_,icDofSize+AlgTraits::nDim_) += lhsfacCont*projTimeScale_;
      lhs(irDofSize+AlgTraits::nDim_,icDofSize+AlgTraits::nDim_) -= lhsfacCont*projTimeScale_;

      for ( int i = 0; i < AlgTraits::nDim_; ++i ) {    
        // d(Res^Ui)/dP; d/dP(-projTimeScale_*dp/dxj*ui*nj*dS)
        lhs(ilDofSize+i,icDofSize+AlgTraits::nDim_) += includePstabInMom*projTimeScale_*lhsfacCont*w_uNp1Ip[i];
        lhs(irDofSize+i,icDofSize+AlgTraits::nDim_) -= includePstabInMom*projTimeScale_*lhsfacCont*w_uNp1Ip[i];
        // d/ui(Res^P); d/dui(rho*ui*nidS);
        lhs(ilDofSize+AlgTraits::nDim_,icDofSize+i) += r*rhoIc*v_scs_areav(ip,i);
        lhs(irDofSize+AlgTraits::nDim_,icDofSize+i) -= r*rhoIc*v_scs_areav(ip,i);        
      }
    }
    
    // compute mdot
    DoubleType mdot = 0.0;
    DoubleType mdotMom = 0.0;
    for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
      mdot += includeOld*
        (w_rhoUNp1Ip[j] - projTimeScale_*(w_dpdxIp[j] - includeGjp*w_GjpOldIp[j]))*v_scs_areav(ip,j);
      mdotMom += includeOld*
        (w_rhoUNp1Ip[j] - includePstabInMom*projTimeScale_*(w_dpdxIp[j] - includeGjp*w_GjpOldIp[j]))*v_scs_areav(ip,j);
      mdot += om_includeOld*
        (w_rhoUNp1Ip[j] - projTimeScale_*(w_dpdxIp[j] - includeGjp*w_GjpIp[j]))*v_scs_areav(ip,j);
      mdotMom += om_includeOld*
        (w_rhoUNp1Ip[j] - includePstabInMom*projTimeScale_*(w_dpdxIp[j] - includeGjp*w_GjpIp[j]))*v_scs_areav(ip,j);
    }
    
    // rhs, Res^P
    rhs(ilDofSize+AlgTraits::nDim_) -= mdot;
    rhs(irDofSize+AlgTraits::nDim_) += mdot;
    
    // assemble advection; rhs only; add in divU stress (explicit)
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
      
      // 2nd order central total advection
      const DoubleType aflux = mdotMom*w_uNp1Ip[i] + pIp*v_scs_areav(ip,i);
      
      // divU stress term
      const DoubleType divUstress = 2.0/3.0*muIp*divU*v_scs_areav(ip,i)*includeDivU_;

      // right hand side; L and R
      rhs(ilDofSize+i) -= aflux + divUstress;
      rhs(irDofSize+i) += aflux + divUstress;
    }
    
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

      const int icDofSize = ic*dofSize_;

      const DoubleType rhoIc = v_densityNp1(ic);

      // advection
      const DoubleType lhsfacAdv = mdot*v_shape_function_scs_(ip,ic);
      const DoubleType lhsfacAdvP = v_shape_function_scs_(ip,ic);
      
      for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
        
        const int indexL = ilDofSize + i;
        const int indexR = irDofSize + i;
        
        // d(Res^Ui)/d(Ui); mdot*ui
        lhs(indexL,icDofSize+i) += lhsfacAdv;
        lhs(indexR,icDofSize+i) -= lhsfacAdv;
        
        // d(Res^Ui)/d(P); p*ni*dS
        lhs(indexL,icDofSize+AlgTraits::nDim_) += lhsfacAdvP*v_scs_areav(ip,i);
        lhs(indexR,icDofSize+AlgTraits::nDim_) -= lhsfacAdvP*v_scs_areav(ip,i);
        
        // viscous stress and non-linear advection (mdot)
        const DoubleType uiIp = w_uNp1Ip[i];
        
        DoubleType lhs_riC_i = 0.0;
        for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
          
          const DoubleType axj = v_scs_areav(ip,j);
          const DoubleType uj = v_velocityNp1(ic,j);

          // d(Res^Ui)/d(Uj); rho*uj*ui*nj*dS
          lhs(indexL,icDofSize+j) += rhoIc*lhsfacAdvP*uiIp*axj;
          lhs(indexR,icDofSize+j) -= rhoIc*lhsfacAdvP*uiIp*axj;

          // -mu*dui/dxj*A_j; fixed i over j loop; see below..
          const DoubleType lhsfacDiff_i = -muIp*v_dndx(ip,ic,j)*axj;
          // lhs; il then ir
          lhs_riC_i += lhsfacDiff_i;

          // -mu*duj/dxi*A_j
          const DoubleType lhsfacDiff_j = -muIp*v_dndx(ip,ic,i)*axj;

          // d(Res^Ui)/d(Uj)
          lhs(indexL,icDofSize+j) += lhsfacDiff_j;
          lhs(indexR,icDofSize+j) -= lhsfacDiff_j;
          // rhs on the spot
          rhs(indexL) -= lhsfacDiff_j*uj;
          rhs(indexR) += lhsfacDiff_j*uj;
        }

        // d(Res^Ui)/d(Ui); -mu*dui/dxj*Aj
        lhs(indexL,icDofSize+i) += lhs_riC_i;
        lhs(indexR,icDofSize+i) -= lhs_riC_i;
        const DoubleType ui = v_velocityNp1(ic,i);
        rhs(indexL) -= lhs_riC_i*ui;
        rhs(indexR) += lhs_riC_i*ui;
      }
    }
  }
}

INSTANTIATE_KERNEL(MomentumContinuityElemKernel);

}  // nalu
}  // sierra
