/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "KerrBHLevel.hpp"
#include "BoxLoops.hpp"
#include "CCZ4.hpp"
#include "ChiExtractionTaggingCriterion.hpp"
#include "ComputePack.hpp"
#include "Constraints.hpp"
#include "KerrBHLevel.hpp"
#include "NanCheck.hpp"
#include "PositiveChiAndAlpha.hpp"
#include "SetValue.hpp"
#include "TraceARemoval.hpp"

// Initial data
#include "GammaCalculator.hpp"
#include "KerrBH.hpp"

#include "ADMMass.hpp"
#include "ADMMassExtraction.hpp"

void KerrBHLevel::specificAdvance()
{
    // Enforce the trace free A_ij condition and positive chi and alpha
    BoxLoops::loop(make_compute_pack(TraceARemoval(), PositiveChiAndAlpha()),
                   m_state_new, m_state_new, INCLUDE_GHOST_CELLS);

    // Check for nan's
    if (m_p.nan_check)
        BoxLoops::loop(NanCheck(), m_state_new, m_state_new,
                       EXCLUDE_GHOST_CELLS, disable_simd());
}

void KerrBHLevel::initialData()
{
    CH_TIME("KerrBHLevel::initialData");
    if (m_verbosity)
        pout() << "KerrBHLevel::initialData " << m_level << endl;

    // First set everything to zero (to avoid undefinded values on constraints)
    // then calculate initial data  Get the Kerr solution in the variables, then
    // calculate the \tilde\Gamma^i numerically as these  are non zero and not
    // calculated in the Kerr ICs
    BoxLoops::loop(
        make_compute_pack(SetValue(0.), KerrBH(m_p.kerr_params, m_dx)),
        m_state_new, m_state_new, INCLUDE_GHOST_CELLS);

    fillAllGhosts();
    BoxLoops::loop(GammaCalculator(m_dx), m_state_new, m_state_new,
                   EXCLUDE_GHOST_CELLS);
}

void KerrBHLevel::preCheckpointLevel()
{
    fillAllGhosts();
    BoxLoops::loop(Constraints(m_dx), m_state_new, m_state_new,
                   EXCLUDE_GHOST_CELLS);
}

void KerrBHLevel::specificEvalRHS(GRLevelData &a_soln, GRLevelData &a_rhs,
                                  const double a_time)
{
    // Enforce the trace free A_ij condition and positive chi and alpha
    BoxLoops::loop(make_compute_pack(TraceARemoval(), PositiveChiAndAlpha()),
                   a_soln, a_soln, INCLUDE_GHOST_CELLS);

    // Calculate CCZ4 right hand side and set constraints to zero to avoid
    // undefined values
    BoxLoops::loop(make_compute_pack(
                       CCZ4(m_p.ccz4_params, m_dx, m_p.sigma, m_p.formulation),
                       SetValue(0, Interval(c_Ham, NUM_VARS - 1))),
                   a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
}

void KerrBHLevel::specificUpdateODE(GRLevelData &a_soln,
                                    const GRLevelData &a_rhs, Real a_dt)
{
    // Enforce the trace free A_ij condition
    BoxLoops::loop(TraceARemoval(), a_soln, a_soln, INCLUDE_GHOST_CELLS);
}

void KerrBHLevel::computeTaggingCriterion(FArrayBox &tagging_criterion,
                                          const FArrayBox &current_state)
{
    BoxLoops::loop(ChiExtractionTaggingCriterion(m_dx, m_level, m_p.max_level,
                                                 m_p.extraction_params,
                                                 m_p.activate_extraction),
                   current_state, tagging_criterion);
}

void KerrBHLevel::specificPostTimeStep()
{
    CH_TIME("KerrBHLevel::specificPostTimeStep");
    // Do the extraction on the min extraction level
    if (m_p.activate_extraction == 1)
    {
        auto min_level = m_p.extraction_params.min_extraction_level();

        // ensure ADM Mass is only calculated on full cycles of 'min_level'
        double extraction_dt = m_dt * pow(2., m_level - min_level);
        double finest_dt = m_dt * pow(2., m_level - m_p.max_level);
        if (fabs(remainder(m_time, extraction_dt)) > finest_dt / 2.)
            return;

        // Populate the ADM Mass and Spin values on the grid
        fillAllGhosts();
        BoxLoops::loop(ADMMass(m_p.extraction_params.center, m_dx), m_state_new,
                       m_state_new, EXCLUDE_GHOST_CELLS);

        if (m_level == min_level)
        {
            CH_TIME("ADMExtraction");

            // Now refresh the interpolator and do the interpolation
            m_gr_amr.m_interpolator->refresh();
            ADMMassExtraction my_extraction(m_p.extraction_params, m_dt, m_time,
                                            m_restart_time);
            my_extraction.execute_query(m_gr_amr.m_interpolator);
        }
    }
}