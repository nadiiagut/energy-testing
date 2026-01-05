import asyncio
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..models.energy import (
    EnergySource,
    EnergyGridState,
    SimulationResult,
    ProducerConfig,
    ConsumerConfig,
)


class EnergyGridSimulator:
    def __init__(self, time_step_seconds: int = 300):
        self.producers: Dict[str, ProducerConfig] = {}
        self.consumers: Dict[str, ConsumerConfig] = {}
        self.time_step = timedelta(seconds=time_step_seconds)
        self.grid_frequency = 50.0  # Hz
        self.nominal_voltage = 230.0  # V
        self.timezone_offset = 0  # Hours from UTC

    def add_producer(
        self,
        producer_id: str,
        source_type: EnergySource,
        capacity_kw: float,
        location: str = "default",
        efficiency: float = 1.0,
        **kwargs
    ) -> None:
        """Add a power producer to the grid."""
        config = ProducerConfig(
            producer_id=producer_id,
            source_type=source_type,
            capacity_kw=capacity_kw,
            location=location,
            efficiency=efficiency,
            **kwargs
        )
        self.producers[producer_id] = config

    def add_consumer(
        self,
        consumer_id: str,
        max_demand_kw: float,
        location: str = "default",
        consumer_type: str = "residential",
        priority: int = 1,
        flexibility: float = 0.0,
        **kwargs
    ) -> None:
        """Add a power consumer to the grid."""
        config = ConsumerConfig(
            consumer_id=consumer_id,
            max_demand_kw=max_demand_kw,
            location=location,
            consumer_type=consumer_type,
            priority=priority,
            flexibility=flexibility,
            **kwargs
        )
        self.consumers[consumer_id] = config

    async def simulate_async(
        self,
        duration_hours: float = 24.0,
        start_time: Optional[datetime] = None,
    ) -> SimulationResult:
        """Run an asynchronous simulation of the energy grid."""
        start_time = start_time or datetime.utcnow()
        end_time = start_time + timedelta(hours=duration_hours)
        current_time = start_time
        
        total_energy_produced = 0.0
        total_energy_consumed = 0.0
        energy_by_source = {source: 0.0 for source in EnergySource}
        grid_states = []

        while current_time < end_time:
            # Simulate one time step
            producers_output = self._simulate_producers(current_time)
            consumers_demand = self._simulate_consumers(current_time)
            
            # Calculate grid state
            total_production = sum(producers_output.values())
            total_demand = sum(consumers_demand.values())
            
            # Update energy totals
            time_step_hours = self.time_step.total_seconds() / 3600
            step_energy_produced = total_production * time_step_hours
            step_energy_consumed = min(total_demand, total_production) * time_step_hours
            
            total_energy_produced += step_energy_produced
            total_energy_consumed += step_energy_consumed
            
            # Update energy by source
            for producer_id, output in producers_output.items():
                source = self.producers[producer_id].source_type
                energy_by_source[source] += output * time_step_hours
            
            # Record grid state
            grid_state = EnergyGridState(
                timestamp=current_time,
                producers=producers_output,
                consumers=consumers_demand,
                grid_frequency=self._calculate_frequency(total_production, total_demand),
                voltage=self.nominal_voltage,
            )
            grid_states.append(grid_state)
            
            # Move to next time step
            current_time += self.time_step
            await asyncio.sleep(0)  # Yield control to event loop

        return SimulationResult(
            start_time=start_time,
            end_time=end_time,
            total_energy_produced=total_energy_produced,
            total_energy_consumed=total_energy_consumed,
            energy_by_source=energy_by_source,
            grid_states=grid_states,
        )

    def simulate(
        self,
        duration_hours: float = 24.0,
        start_time: Optional[datetime] = None,
    ) -> SimulationResult:
        """Synchronous wrapper for simulate_async."""
        return asyncio.run(self.simulate_async(duration_hours, start_time))

    def _simulate_producers(self, timestamp: datetime) -> Dict[str, float]:
        """Simulate power output for all producers."""
        outputs = {}
        local_time = timestamp + timedelta(hours=self.timezone_offset)
        hour = local_time.hour
        
        for producer_id, producer in self.producers.items():
            if producer.source_type == EnergySource.SOLAR:
                # Solar output model based on time of day
                if 6 <= hour < 18:  # Daytime
                    # Sinusoidal curve peaking at noon
                    solar_factor = max(0, math.sin((hour - 6) * math.pi / 12))
                    # Add some weather variability
                    weather_factor = random.uniform(0.7, 1.0)
                    outputs[producer_id] = (
                        producer.capacity_kw * 
                        solar_factor * 
                        weather_factor * 
                        producer.efficiency
                    )
                else:
                    outputs[producer_id] = 0.0
                    
            elif producer.source_type == EnergySource.WIND:
                # Wind output with some randomness
                wind_factor = random.uniform(0.2, 1.0)
                outputs[producer_id] = (
                    producer.capacity_kw * 
                    wind_factor * 
                    producer.efficiency
                )
                
            elif producer.source_type == EnergySource.HYDRO:
                # Hydro is relatively stable
                hydro_factor = random.uniform(0.8, 1.0)
                outputs[producer_id] = (
                    producer.capacity_kw * 
                    hydro_factor * 
                    producer.efficiency
                )
                
            elif producer.source_type in [EnergySource.COAL, EnergySource.GAS, EnergySource.NUCLEAR]:
                # These are baseload plants - relatively stable
                baseload_factor = random.uniform(0.85, 1.0)
                outputs[producer_id] = (
                    producer.capacity_kw * 
                    baseload_factor * 
                    producer.efficiency
                )
                
            elif producer.source_type == EnergySource.BATTERY:
                # Battery can charge or discharge based on grid needs
                # For now, assume it can discharge up to capacity
                outputs[producer_id] = producer.capacity_kw * producer.efficiency
                
            else:
                # Default for unknown types
                outputs[producer_id] = producer.capacity_kw * producer.efficiency
                
        return outputs

    def _simulate_consumers(self, timestamp: datetime) -> Dict[str, float]:
        """Simulate power demand for all consumers."""
        demands = {}
        local_time = timestamp + timedelta(hours=self.timezone_offset)
        hour = local_time.hour
        day_of_week = local_time.weekday()
        
        for consumer_id, consumer in self.consumers.items():
            base_demand = consumer.max_demand_kw
            
            if consumer.consumer_type == "residential":
                # Residential demand pattern
                if 7 <= hour < 9 or 17 <= hour < 21:  # Morning and evening peaks
                    demand_factor = random.uniform(0.7, 1.0)
                elif 23 <= hour or hour < 5:  # Night
                    demand_factor = random.uniform(0.1, 0.3)
                else:  # Daytime
                    demand_factor = random.uniform(0.3, 0.6)
                    
            elif consumer.consumer_type == "commercial":
                # Commercial demand (business hours)
                if 8 <= hour < 18 and day_of_week < 5:  # Weekday business hours
                    demand_factor = random.uniform(0.8, 1.0)
                else:
                    demand_factor = random.uniform(0.1, 0.3)
                    
            elif consumer.consumer_type == "industrial":
                # Industrial demand (24/7 but with some variation)
                demand_factor = random.uniform(0.7, 1.0)
                
            else:
                # Default pattern
                demand_factor = random.uniform(0.5, 1.0)
            
            # Apply flexibility (demand response)
            if consumer.flexibility > 0:
                # Flexible consumers can reduce demand during peaks
                flexibility_reduction = random.uniform(0, consumer.flexibility * 0.3)
                demand_factor *= (1 - flexibility_reduction)
            
            demands[consumer_id] = base_demand * demand_factor
            
        return demands

    def _calculate_frequency(self, total_production: float, total_demand: float) -> float:
        """Calculate grid frequency based on production/demand balance."""
        if total_demand == 0:
            return 50.0  # Hz
        
        # Frequency deviation based on imbalance
        imbalance_ratio = (total_production - total_demand) / total_demand
        
        # Simple model: frequency varies between 49.5 and 50.5 Hz
        # In reality, this would be much more complex with governor response
        frequency_deviation = imbalance_ratio * 2.0  # Max deviation of ±2Hz
        frequency = 50.0 + frequency_deviation
        
        # Clamp to realistic bounds
        return max(49.0, min(51.0, frequency))

    def get_summary_stats(self, result: SimulationResult) -> Dict[str, float]:
        """Get summary statistics from simulation results."""
        if not result.grid_states:
            return {}
        
        frequencies = [state.grid_frequency for state in result.grid_states]
        generations = [state.total_generation for state in result.grid_states]
        demands = [state.total_demand for state in result.grid_states]
        
        return {
            "avg_frequency": sum(frequencies) / len(frequencies),
            "min_frequency": min(frequencies),
            "max_frequency": max(frequencies),
            "avg_generation": sum(generations) / len(generations),
            "avg_demand": sum(demands) / len(demands),
            "generation_efficiency": result.total_energy_consumed / result.total_energy_produced if result.total_energy_produced > 0 else 0,
        }
