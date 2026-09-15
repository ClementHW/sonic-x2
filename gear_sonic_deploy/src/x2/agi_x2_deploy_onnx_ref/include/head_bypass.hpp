/**
 * @file head_bypass.hpp
 * @brief Surgical override of SONIC's head target for the 2 head DOFs.
 *
 * SONIC's head tracking saturates at ~+-10 deg: the policy was trained on
 * motion data whose head range is small, so a larger pad head-look
 * reference is simply clipped away by the policy's output distribution.
 * The pad/VR head-look reference therefore drives the two head DOFs
 * directly (mirror of the wrist bypass for the 4 broken wrist DOFs; see
 * wrist_bypass.hpp for the full rationale and telemetry methodology).
 *
 * The bypass is CLI-gated (``--head-bypass {off,ref}`` on the deploy
 * binary, default off). When ``ref`` is selected, ``OnControl`` calls
 * ``ApplyHeadBypass`` BEFORE the safety stack so soft-start blend,
 * ``--max-target-dev`` clamp, and the tilt-trip force-to-default branch
 * still apply uniformly to the reference-driven target.
 *
 * Lives in its own header so unit tests can exercise the override loop
 * without pulling in ROS 2 / ONNX runtime.
 */

#ifndef AGI_X2_HEAD_BYPASS_HPP
#define AGI_X2_HEAD_BYPASS_HPP

#include "policy_parameters.hpp"
#include "reference_motion.hpp"

#include <array>
#include <cmath>

namespace agi_x2 {

/// MJ-order joint indices the bypass overrides:
///   29 = head_yaw_joint
///   30 = head_pitch_joint
///
/// IMPORTANT: keep in sync with ``policy_parameters.hpp``'s
/// ``mujoco_joint_names`` ordering (both head joints are the last two
/// slots of the 31-DOF MJ list and have ``default_angles == 0.0``,
/// which is a slot-validity invariant on top of the index numbers
/// themselves).
inline constexpr std::array<int, 2> kBypassedHeadMjDofs = {29, 30};

/// Override ``target_pos_mj[mj]`` with ``ref.joint_pos_mj[mj]`` for every
/// joint index in ``kBypassedHeadMjDofs``. Returns the largest absolute
/// delta between the original target and the reference across the
/// overridden slots, so the deploy can surface it on the periodic status
/// line as a "how hard is SONIC being overruled" indicator.
///
/// All other slots of ``target_pos_mj`` are left untouched. The function
/// has no global state; safe to call from the deploy's 50 Hz control
/// loop and from unit tests interchangeably.
inline double ApplyHeadBypass(std::array<double, NUM_DOFS>& target_pos_mj,
                              const ReferenceFrame&         ref)
{
  double max_delta = 0.0;
  for (const int mj : kBypassedHeadMjDofs) {
    const double delta = std::fabs(target_pos_mj[mj] - ref.joint_pos_mj[mj]);
    if (delta > max_delta) max_delta = delta;
    target_pos_mj[mj] = ref.joint_pos_mj[mj];
  }
  return max_delta;
}

}  // namespace agi_x2

#endif  // AGI_X2_HEAD_BYPASS_HPP
