from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Literal, Optional
from enum import Enum
import math


class EnergySource(str, Enum):
    SOLAR = "solar"
    WIND = "wind"
    COAL = "coal"
    GAS = "natural_gas"
    HYDRO = "hydro"
    NUCLEAR = "nuclear"
    BATTERY = "battery"


class ServiceMode(str, Enum):
    """Frequency response service modes.
    
    Each mode has different response characteristics:
    - DC: Dynamic Containment - aggressive, near-instant response (steepest slope)
    - DR: Dynamic Regulation - gentler, more tracking/regulated behavior
    - DM: Dynamic Moderation - moderate response characteristics
    """
    DC = "DC"   # Dynamic Containment
    DR = "DR"   # Dynamic Regulation
    DM = "DM"   # Dynamic Moderation


@dataclass
class EnergyReading:
    timestamp: datetime
    value_kw: float
    source: EnergySource
    location: str
    efficiency: float = 1.0


@dataclass
class EnergyGridState:
    timestamp: datetime
    producers: Dict[str, float]  # producer_id -> current_output_kw
    consumers: Dict[str, float]  # consumer_id -> current_demand_kw
    grid_frequency: float  # Hz
    voltage: float  # V
    total_generation: float = 0.0
    total_demand: float = 0.0
    
    def __post_init__(self):
        self.total_generation = sum(self.producers.values())
        self.total_demand = sum(self.consumers.values())


@dataclass
class SimulationResult:
    start_time: datetime
    end_time: datetime
    total_energy_produced: float  # kWh
    total_energy_consumed: float  # kWh
    energy_by_source: Dict[EnergySource, float]  # source -> energy_kwh
    grid_states: List[EnergyGridState]
    peak_demand: float = 0.0
    peak_generation: float = 0.0
    average_frequency: float = 50.0
    
    def __post_init__(self):
        # Calculate derived metrics
        if self.grid_states:
            self.peak_demand = max(state.total_demand for state in self.grid_states)
            self.peak_generation = max(state.total_generation for state in self.grid_states)
            self.average_frequency = sum(state.grid_frequency for state in self.grid_states) / len(self.grid_states)


@dataclass
class ProducerConfig:
    producer_id: str
    source_type: EnergySource
    capacity_kw: float
    location: str
    efficiency: float = 1.0
    maintenance_schedule: Optional[List[datetime]] = None
    cost_per_mwh: float = 0.0


@dataclass
class ConsumerConfig:
    consumer_id: str
    max_demand_kw: float
    location: str
    consumer_type: str = "residential"  # residential, commercial, industrial
    priority: int = 1  # 1=high, 2=medium, 3=low
    flexibility: float = 0.0  # 0=none, 1=fully flexible


@dataclass(frozen=True)
class ObservedSignal:
    """Describes how a signal is observed / measured.

    This does NOT model production monitoring.
    It models the properties of observation that affect penalties.
    
    Attributes:
        source: Semantic label for signal origin (simulated, scada, nimbus)
        sample_rate_hz: Sampling frequency of the observation
        jitter_ms: Timing jitter in milliseconds
        latency_ms: Observation latency in milliseconds
        drop_probability: Probability of signal drop (0.0 to 1.0)
    
    Notes:
        - frozen=True ensures immutability (important for test determinism)
        - source is semantic, not functional
        - No dependency on Nimbus code (important politically and architecturally)
    """
    source: Literal["simulated", "scada", "nimbus"]
    sample_rate_hz: float
    jitter_ms: float = 0.0
    latency_ms: float = 0.0
    drop_probability: float = 0.0
