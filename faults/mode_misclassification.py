"""Mode Misclassification Fault Injection.

Simulates a control/service mode mismatch:
e.g. the asset is contracted/settled as DC, but the controller behaves like DR (or vice versa).

This is a penalty generator because:
- Contract expects one response envelope (e.g., DC = fast, steep)
- Controller actually runs a different envelope (e.g., DR = slower, gentler)
- Power delivery "looks plausible" but fails contract scoring

Real-world causes:
- Configuration error in PPC
- Firmware update changes default mode
- Operator mistake during mode switch
- Communication failure leaving controller in wrong state
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from models.energy import ServiceMode


@dataclass(frozen=True)
class ModeMisclassificationFault:
    """Simulates a control/service mode mismatch.
    
    The asset is contracted/settled as expected_mode, but the controller
    behaves like actual_mode. This creates a mismatch where:
    - Penalties are evaluated against expected_mode envelope
    - Controller response follows actual_mode characteristics
    
    Usage:
        fault = ModeMisclassificationFault(
            expected_mode=ServiceMode.DC,  # Contract expects DC
            actual_mode=ServiceMode.DR,    # But controller runs as DR
        )
        
        # In simulation loop:
        mode_for_controller = fault.apply_mode(contract_mode)
        power = controller.respond(freq, dt, service_mode=mode_for_controller)
    """
    expected_mode: ServiceMode   # How penalties are evaluated / contract expects
    actual_mode: ServiceMode     # How the control loop actually behaves

    def apply_mode(self, mode: ServiceMode) -> ServiceMode:
        """Apply mode misclassification.
        
        If the harness says "run expected_mode", force the control loop
        to run as actual_mode instead.
        
        Args:
            mode: The mode the system thinks it should run
            
        Returns:
            The mode the controller will actually use
        """
        if mode == self.expected_mode:
            return self.actual_mode
        return mode

    def is_misclassified(self) -> bool:
        """Check if there is a mode mismatch."""
        return self.expected_mode != self.actual_mode

    def get_penalty_metrics(self) -> Dict:
        """Get metrics about the misclassification for reporting."""
        return {
            "expected_mode": self.expected_mode.value,
            "actual_mode": self.actual_mode.value,
            "is_misclassified": self.is_misclassified(),
            "mismatch_description": self._describe_mismatch(),
        }

    def _describe_mismatch(self) -> str:
        """Generate human-readable description of the mismatch."""
        if not self.is_misclassified():
            return "No mismatch - modes are aligned"
        
        # Describe the impact
        mode_severity = {
            ServiceMode.DC: 3,  # Most aggressive
            ServiceMode.DM: 2,  # Moderate
            ServiceMode.DR: 1,  # Most gentle
        }
        
        expected_severity = mode_severity.get(self.expected_mode, 0)
        actual_severity = mode_severity.get(self.actual_mode, 0)
        
        if expected_severity > actual_severity:
            return (
                f"Under-response risk: Contract expects {self.expected_mode.value} "
                f"(aggressive) but running {self.actual_mode.value} (gentler). "
                "Response will be slower/weaker than required."
            )
        else:
            return (
                f"Over-response risk: Contract expects {self.expected_mode.value} "
                f"(gentler) but running {self.actual_mode.value} (aggressive). "
                "Response may overshoot or oscillate."
            )


def apply_mode_faults(mode: ServiceMode, faults: list) -> ServiceMode:
    """Apply all mode-related faults to determine actual controller mode.
    
    This is the hook to use in the harness simulation loop.
    
    Args:
        mode: The expected/contract service mode
        faults: List of fault objects (may include ModeMisclassificationFault)
        
    Returns:
        The mode the controller should actually use
    """
    mode_for_controller = mode
    for f in faults:
        if hasattr(f, "apply_mode"):
            mode_for_controller = f.apply_mode(mode_for_controller)
    return mode_for_controller
