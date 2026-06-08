/*------------------------------------------------------------------------*/
/*  Copyright 2025 COMERI.                                                */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Mare-Nalu     */
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
  // throw std::runtime_error("MomentumContinuityElemKernel::execute() Error: Not implemented!"); // was here originally.
  // NOTE: Jun Yamasaki's monolithic implementation starts here.
  // TODO: check if the above code is the same from _4J.C?

  // Work arrays:
  NALU_ALIGNED DoubleType w_uNm1Ip[AlgTraits::nDim_]; // TOASK: what is NALU_ALIGNED?
  NALU_ALIGNED DoubleType w_uNIp[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_uNp1Ip[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_rhoUNp1Ip[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_rhoUNIp[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_dpdxIp[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_GjpIp[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_GjpOldIp[AlgTraits::nDim_];

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

  // ------ IMPLEMENTATION ------:
  
  // TODO: time integration?
  
  // Create diagonal entries first...

  // ------------------ Continuity equation-related ------------------
  // 1) LHS: d(ResP)/dP and RHS: - ResP
  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    const int il = lrscv_[2*ip];
    const int ir = lrscv_[2*ip+1];

    // Offsets, since memory locs are: 
    // 0 u
    // 1 v
    // 2 (=nDim-1) w
    // 3 (nDim) p, continuity

    // save off some offsets
    const int ilNdim = il*(AlgTraits::nDim_+1); // modified because uwv and p.
    const int irNdim = ir*(AlgTraits::nDim_+1);

    // continuity index:
    const int ilCont = ilNdim + AlgTraits::nDim;
    const int irCont = irNdim + AlgTraits::nDim;

    // ready the worker variables.
    for (int j = 0; j < AlgTraits::nDim_; ++j) { 
      w_rho_uIp[j] = 0.0; 
      w_Gpdx_Ip[j] = 0.0;
      w_dpdxIp[j] = 0.0;
    }

    // loop over nodes (total npe)
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) { 
      const DoubleType r = v_shape_function_(ip, ic);
      const DoubleType nodalPressure = v_pressure(ic);
      const DoubleType nodalRho = v_densityNp1(ic);

      DoubleType lhsfac = 0.0;
      for (int j = 0; j < AlgTraits::nDim_; ++j) {
        w_Gpdx_Ip[j] += r * v_Gjp(ic, j); // interpolate values to subcontrol-surface integration points using the Lagrange functions at this node, ic.
        w_rho_uIp[j] += r * nodalRho * v_velocity(ic, j);
        w_dpdxIp[j]  += v_dndx(ip, ic, j) * nodalPressure;
        lhsfac += - projTimeScale_* v_dndx_lhs(ip, ic, j) * v_scs_areav(ip, j); // v_scs_areav is A_j^ip.
      }

      lhs(ilCont,ic) += lhsfac;
      lhs(irCont,ic) -= lhsfac;
    }

    // assemble mdot
    DoubleType mdot = 0.0; //TOASK: does this stay in the scope, i.e. can I use it later? in another loop?
    for (int j = 0; j < AlgTraits::nDim_; ++j) {
      mdot += (w_rho_uIp[j] - projTimeScale_*(w_dpdxIp[j] - w_Gpdx_Ip[j]))*v_scs_areav(ip,j);
    }

    // residuals
    rhs(ilCont) -= mdot;
    rhs(irCont) += mdot;
  }

  // ------------------ Momentum equation-related ------------------:
  for ( int ip = 0; ip < AlgTraits::numScsIp_; ++ip ) {

    // left and right nodes for this ip
    const int il = lrscv_[2*ip];
    const int ir = lrscv_[2*ip+1];

    // save off some offsets
    const int ilNdim = il*AlgTraits::nDim_;
    const int irNdim = ir*AlgTraits::nDim_;

    // save off mdot
    const DoubleType tmdot = v_mdot(ip);

    // compute scs point values; sneak in divU
    DoubleType muIp = 0.0;
    DoubleType divU = 0.0;
    for ( int i = 0; i < AlgTraits::nDim_; ++i )
      w_uIp[i] = 0.0;

    // ingredients for mdot, ready it here.
    for (int j = 0; j < AlgTraits::nDim_; ++j) {
      w_rho_uIp[j] = 0.0;
      w_Gpdx_Ip[j] = 0.0;
      w_dpdxIp[j] = 0.0;
    }

    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

      const DoubleType r = v_shape_function_(ip,ic);
      const DoubleType rAdv = v_adv_shape_function_(ip,ic);
      const DoubleType nodalRho = v_densityNp1(ic);
      const DoubleType nodalPressure = v_pressure(ic);
      const DoubleType nodalRho = v_densityNp1(ic);

      muIp += r*v_viscosity(ic);

      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        const DoubleType uj = v_uNp1(ic,j);
        w_uIp[j] += rAdv*uj;
        divU += uj*v_dndx(ip,ic,j);

        // for mdot:
        w_Gpdx_Ip[j] += r * v_Gpdx(ic, j);
        w_rho_uIp[j] += r * nodalRho * uj;
        w_dpdxIp[j]  += v_dndx(ip, ic, j) * nodalPressure;
      }

    }

    // assemble mdot
    DoubleType tmdot = 0.0;
    for (int j = 0; j < AlgTraits::nDim_; ++j) {
      tmdot += (w_rho_uIp[j] - projTimeScale_*(w_dpdxIp[j] - w_Gpdx_Ip[j]))*v_scs_areav(ip,j);
    }

    // assemble advection; rhs only; add in divU stress (explicit)
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {

      // 2nd order central
      const DoubleType uiIp = w_uIp[i];      

      // total advection; (pressure contribution in time term) // TOASK ???
      const DoubleType aflux =tmdot*uiIp;

      // divU stress term
      const DoubleType divUstress = 2.0/3.0*muIp*divU*v_scs_areav(ip,i)*includeDivU_;

      const int indexL = ilNdim + i;
      const int indexR = irNdim + i;

      // right hand side; L and R
      rhs(indexL) -= aflux + divUstress;
      rhs(indexR) += aflux + divUstress;
    }

    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

      const int icNdim = ic*AlgTraits::nDim_;

      // advection and diffusion
      const DoubleType lhsfacAdv = v_adv_shape_function_(ip,ic)*tmdot;

      for ( int i = 0; i < AlgTraits::nDim_; ++i ) {

        const int indexL = ilNdim + i;
        const int indexR = irNdim + i;

        // advection operator lhs; rhs handled above
        // lhs; il then ir
        lhs(indexL,icNdim+i) += lhsfacAdv;
        lhs(indexR,icNdim+i) -= lhsfacAdv;

        // viscous stress
        DoubleType lhs_riC_i = 0.0;
        for ( int j = 0; j < AlgTraits::nDim_; ++j ) {

          const DoubleType axj = v_scs_areav(ip,j);
          const DoubleType uj = v_uNp1(ic,j);

          // -mu*dui/dxj*A_j; fixed i over j loop; see below..
          const DoubleType lhsfacDiff_i = -muIp*v_dndx(ip,ic,j)*axj;
          // lhs; il then ir
          lhs_riC_i += lhsfacDiff_i;

          // -mu*duj/dxi*A_j
          const DoubleType lhsfacDiff_j = -muIp*v_dndx(ip,ic,i)*axj;
          // lhs; il then ir
          lhs(indexL,icNdim+j) += lhsfacDiff_j;
          lhs(indexR,icNdim+j) -= lhsfacDiff_j;
          // rhs; il then ir
          rhs(indexL) -= lhsfacDiff_j*uj;
          rhs(indexR) += lhsfacDiff_j*uj;
        }

        // deal with accumulated lhs and flux for -mu*dui/dxj*Aj
        lhs(indexL,icNdim+i) += lhs_riC_i;
        lhs(indexR,icNdim+i) -= lhs_riC_i;
        const DoubleType ui = v_uNp1(ic,i);
        rhs(indexL) -= lhs_riC_i*ui;
        rhs(indexR) += lhs_riC_i*ui;
      }
    }
  }

  // --------------- NOW THE OFF-DIAGONALS --------------- 
  // TODO: figure out the indexing?

  





  // OTHERs... TO-IMPLEMENT...





}

  


  


}

INSTANTIATE_KERNEL(MomentumContinuityElemKernel);

}  // nalu
}  // sierra
