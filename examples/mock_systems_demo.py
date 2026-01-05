#!/usr/bin/env python3
"""
Demo of mock energy system components.

This example shows how to use the mock components to test various scenarios
including outages, price responses, and market trading.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from energy_testing.simulators.mocks import (
    MockEnergyProducer,
    MockEnergyConsumer,
    MockGridOperator,
    MockEnergyMarket,
)
from energy_testing.models.energy import EnergySource


def demo_producer_behavior():
    """Demonstrate producer behavior under different conditions."""
    print("1. Producer Behavior Demo")
    print("-" * 30)
    
    # Create a solar producer
    solar = MockEnergyProducer("solar_1", EnergySource.SOLAR, capacity_kw=1000)
    
    # Test output at different times
    times = [
        datetime(2024, 6, 15, 6, 0),   # 6 AM - sunrise
        datetime(2024, 6, 15, 12, 0),  # Noon - peak
        datetime(2024, 6, 15, 18, 0),  # 6 PM - sunset
        datetime(2024, 6, 15, 23, 0),  # 11 PM - night
    ]
    
    for time in times:
        output = solar.get_current_output(time)
        print(f"  {time.strftime('%H:%M')} - Output: {output:.1f} kW")
    
    # Test failure scenario
    print("\n  Testing failure scenario:")
    solar.force_offline()
    output = solar.get_current_output(datetime(2024, 6, 15, 12, 0))
    print(f"  Forced offline - Output: {output:.1f} kW")
    
    print()


def demo_consumer_price_response():
    """Demonstrate consumer response to price signals."""
    print("2. Consumer Price Response Demo")
    print("-" * 35)
    
    # Create consumers with different flexibility
    rigid_consumer = MockEnergyConsumer(
        "factory", 
        max_demand_kw=500,
        consumer_type="industrial",
        flexibility=0.0
    )
    
    flexible_consumer = MockEnergyConsumer(
        "smart_home",
        max_demand_kw=10,
        consumer_type="residential", 
        flexibility=0.5
    )
    
    timestamp = datetime(2024, 6, 15, 18, 0)  # Evening peak
    
    # Test different price levels
    prices = [30, 50, 80, 120]  # $/MWh
    
    print(f"  Base demand at {timestamp.strftime('%H:%M')}:")
    base_rigid = rigid_consumer.get_current_demand(timestamp)
    base_flexible = flexible_consumer.get_current_demand(timestamp)
    print(f"    Rigid consumer: {base_rigid:.1f} kW")
    print(f"    Flexible consumer: {base_flexible:.1f} kW")
    
    print("\n  Response to price signals:")
    for price in prices:
        rigid_demand = rigid_consumer.get_current_demand(timestamp, price)
        flexible_demand = flexible_consumer.get_current_demand(timestamp, price)
        
        rigid_change = ((base_rigid - rigid_demand) / base_rigid) * 100
        flexible_change = ((base_flexible - flexible_demand) / base_flexible) * 100
        
        print(f"    Price ${price}/MWh:")
        print(f"      Rigid: {rigid_demand:.1f} kW ({rigid_change:+.1f}%)")
        print(f"      Flexible: {flexible_demand:.1f} kW ({flexible_change:+.1f}%)")
    
    print()


def demo_grid_operations():
    """Demonstrate grid operator functions."""
    print("3. Grid Operations Demo")
    print("-" * 25)
    
    grid_op = MockGridOperator("test_grid")
    
    # Test frequency under different conditions
    scenarios = [
        (1000, 950, "Excess generation"),
        (950, 1000, "Excess demand"),
        (1000, 1000, "Balanced"),
        (900, 1000, "Large deficit"),
        (1100, 1000, "Large excess"),
    ]
    
    for generation, demand, description in scenarios:
        freq = grid_op.calculate_grid_frequency(generation, demand)
        status = "STABLE" if grid_op.is_stable else "UNSTABLE"
        print(f"  {description}:")
        print(f"    Generation: {generation} MW, Demand: {demand} MW")
        print(f"    Frequency: {freq:.2f} Hz ({status})")
    
    # Test price signals
    print("\n  Price signals by demand level:")
    demand_levels = [0.6, 0.8, 0.9, 0.95, 1.0]
    timestamp = datetime.utcnow()
    
    for level in demand_levels:
        price = grid_op.get_price_signal(timestamp, level)
        print(f"    Demand {level*100:.0f}%: ${price:.1f}/MWh")
    
    # Test outage management
    print("\n  Outage management:")
    grid_op.trigger_outage("downtown", 30)  # 30-minute outage
    print(f"    Outage triggered for downtown area")
    
    check_time = datetime.utcnow() + timedelta(minutes=15)
    is_active = grid_op.is_outage_active("downtown", check_time)
    print(f"    Outage active in 15 minutes: {is_active}")
    
    print()


def demo_energy_market():
    """Demonstrate energy market trading."""
    print("4. Energy Market Demo")
    print("-" * 20)
    
    market = MockEnergyMarket("day_ahead_market")
    
    # Submit bids (buyers)
    bids = [
        ("utility_a", 50, 45),  # Buy 50MW at max $45
        ("utility_b", 30, 55),  # Buy 30MW at max $55
        ("industrial_c", 20, 40),  # Buy 20MW at max $40
    ]
    
    # Submit offers (sellers)
    offers = [
        ("generator_x", 40, 35),  # Sell 40MW at min $35
        ("generator_y", 60, 50),  # Sell 60MW at min $50
        ("generator_z", 30, 60),  # Sell 30MW at min $60
    ]
    
    print("  Submitted Bids:")
    for buyer, quantity, price in bids:
        bid_id = market.submit_bid(buyer, quantity, price)
        print(f"    {buyer}: {quantity}MW at max ${price}/MWh")
    
    print("\n  Submitted Offers:")
    for seller, quantity, price in offers:
        offer_id = market.submit_offer(seller, quantity, price)
        print(f"    {seller}: {quantity}MW at min ${price}/MWh")
    
    # Clear the market
    print("\n  Market Clearing Results:")
    trades = market.clear_market()
    
    if trades:
        print(f"    Number of trades: {len(trades)}")
        print(f"    Clearing price: ${market.clearing_price:.1f}/MWh")
        
        total_volume = sum(trade["quantity_mw"] for trade in trades)
        print(f"    Total volume: {total_volume} MW")
        
        print("\n    Trade Details:")
        for trade in trades:
            print(f"      Trade {trade['trade_id']}: {trade['quantity_mw']}MW at ${trade['price']:.1f}/MWh")
    else:
        print("    No trades executed")
    
    print()


def main():
    """Run all demos."""
    print("Energy System Mock Components Demo")
    print("=" * 50)
    print()
    
    demo_producer_behavior()
    demo_consumer_price_response()
    demo_grid_operations()
    demo_energy_market()
    
    print("Demo completed!")


if __name__ == "__main__":
    main()
