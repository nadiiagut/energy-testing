#!/usr/bin/env python3
"""
Basic energy grid simulation example.

This example demonstrates how to set up and run a simple energy grid simulation
with multiple producers and consumers.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path so we can import energy_testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from energy_testing.simulators.grid import EnergyGridSimulator
from energy_testing.models.energy import EnergySource


def main():
    """Run a basic energy grid simulation."""
    print("Energy Grid Simulation Example")
    print("=" * 40)
    
    # Create simulator with 5-minute time steps
    simulator = EnergyGridSimulator(time_step_seconds=300)
    
    # Add producers
    simulator.add_producer(
        "solar_farm_1",
        EnergySource.SOLAR,
        capacity_kw=2000,
        location="desert_area"
    )
    
    simulator.add_producer(
        "wind_farm_1", 
        EnergySource.WIND,
        capacity_kw=1500,
        location="coastal_area"
    )
    
    simulator.add_producer(
        "coal_plant_1",
        EnergySource.COAL,
        capacity_kw=1000,
        location="industrial_zone"
    )
    
    # Add consumers
    simulator.add_consumer(
        "residential_area_1",
        max_demand_kw=800,
        consumer_type="residential",
        location="suburbs"
    )
    
    simulator.add_consumer(
        "commercial_district_1",
        max_demand_kw=1200,
        consumer_type="commercial",
        location="downtown"
    )
    
    simulator.add_consumer(
        "industrial_complex_1",
        max_demand_kw=1500,
        consumer_type="industrial",
        location="industrial_zone",
        flexibility=0.1
    )
    
    print(f"Producers: {len(simulator.producers)}")
    print(f"Consumers: {len(simulator.consumers)}")
    print()
    
    # Run simulation for 24 hours
    print("Running 24-hour simulation...")
    start_time = datetime(2024, 6, 15, 0, 0)  # Start at midnight
    result = simulator.simulate(duration_hours=24, start_time=start_time)
    
    # Display results
    print("\nSimulation Results:")
    print(f"Duration: {result.start_time} to {result.end_time}")
    print(f"Total Energy Produced: {result.total_energy_produced:.2f} kWh")
    print(f"Total Energy Consumed: {result.total_energy_consumed:.2f} kWh")
    print(f"Peak Generation: {result.peak_generation:.2f} kW")
    print(f"Peak Demand: {result.peak_demand:.2f} kW")
    print(f"Average Frequency: {result.average_frequency:.2f} Hz")
    
    print("\nEnergy by Source:")
    for source, energy in result.energy_by_source.items():
        if energy > 0:
            percentage = (energy / result.total_energy_produced) * 100
            print(f"  {source.value}: {energy:.2f} kWh ({percentage:.1f}%)")
    
    # Get summary statistics
    stats = simulator.get_summary_stats(result)
    print("\nGrid Statistics:")
    print(f"  Average Generation: {stats['avg_generation']:.2f} kW")
    print(f"  Average Demand: {stats['avg_demand']:.2f} kW")
    print(f"  Frequency Range: {stats['min_frequency']:.2f} - {stats['max_frequency']:.2f} Hz")
    print(f"  Generation Efficiency: {stats['generation_efficiency']:.2%}")
    
    # Show sample grid states
    print("\nSample Grid States:")
    sample_interval = len(result.grid_states) // 6  # Show 6 samples
    for i in range(0, len(result.grid_states), sample_interval):
        state = result.grid_states[i]
        print(f"  {state.timestamp.strftime('%H:%M')}: "
              f"Gen={state.total_generation:.0f}kW, "
              f"Demand={state.total_demand:.0f}kW, "
              f"Freq={state.grid_frequency:.2f}Hz")


if __name__ == "__main__":
    main()
