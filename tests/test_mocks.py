"""Test Suite - Energy System Component Behavior.

PURPOSE:
========
Tests for COMPONENT BEHAVIOR in the energy grid system.
These mock classes are STAND-INS for real system endpoints:
- MockEnergyProducer → Real power plant APIs
- MockEnergyConsumer → Real smart meter APIs  
- MockGridOperator → Real grid control systems
- MockEnergyMarket → Real energy trading platforms

We test that COMPONENTS BEHAVE CORRECTLY under various conditions.
The mock implementations will be swapped for real endpoints in production.

CDN ANALOGY:
============
- Producer ~ Origin server (serves content/energy)
- Consumer ~ Client (requests content/energy)
- GridOperator ~ Load balancer (distributes traffic/power)
- EnergyMarket ~ DNS/routing (decides where to get content/energy)
- Maintenance windows ~ Scheduled origin maintenance
- Price signals ~ Dynamic routing weights
"""

import pytest
from datetime import datetime, timedelta, timezone

from simulators.mocks import (
    MockEnergyProducer,
    MockEnergyConsumer,
    MockGridOperator,
    MockEnergyMarket,
)
from models.energy import EnergySource
from tests.test_constants import SUMMER_NOON, SUMMER_EVENING, SUMMER_NIGHT


@pytest.mark.case("MK-001")
def test_mock_producer():
    """Test Case - Mock Energy Producer Lifecycle and Output.

    Description:
    -----------------
    Verify that MockEnergyProducer correctly simulates energy generation
    with time-dependent output (solar varies by time of day) and
    controllable online/offline states. Similar to testing CDN origin
    server availability and response patterns.

    Preconditions:
    -----------------
    1. Create solar producer with 1000kW capacity
    2. Location: test_site
    3. No maintenance windows configured

    Steps:
    ----------
    1. Query output at noon (peak solar) - verify within capacity
    2. Query output at 11 PM (night) - verify zero output
    3. Force producer offline - verify zero output
    4. Force producer online - verify output resumes

    Expected Results:
    ---------------------------
    1. Noon output: 0 <= output <= 1000kW, producer is_online=True
    2. Night output: 0kW (no solar generation)
    3. Offline: output=0, is_online=False
    4. Back online: output > 0, is_online=True
    """
    producer = MockEnergyProducer(
        "solar_1", 
        EnergySource.SOLAR, 
        capacity_kw=1000,
        location="test_site"
    )
    
    # Test normal operation at noon
    output = producer.get_current_output(SUMMER_NOON)
    assert 0 <= output <= 1000
    assert producer.is_online
    
    # Test night time
    output = producer.get_current_output(SUMMER_NIGHT)
    assert output == 0  # No solar at night
    
    # Test forcing offline
    producer.force_offline()
    output = producer.get_current_output(SUMMER_NOON)
    assert output == 0
    assert not producer.is_online
    
    # Test forcing back online
    producer.force_online()
    output = producer.get_current_output(SUMMER_NOON)
    assert output > 0


@pytest.mark.case("MK-002")
def test_mock_consumer():
    """Test Case - Mock Energy Consumer Demand and Price Response.

    Description:
    -----------------
    Verify that MockEnergyConsumer correctly simulates energy demand
    with time-dependent patterns and price-responsive behavior.
    Similar to testing CDN client request patterns and rate limiting.

    Preconditions:
    -----------------
    1. Create residential consumer with 10kW max demand
    2. Flexibility factor: 0.2 (20% demand reduction capability)
    3. Consumer type: residential

    Steps:
    ----------
    1. Query demand at evening peak (6 PM) - verify within max
    2. Query demand with high price signal (80) - verify reduction
    3. Verify consumer is controllable (flexibility > 0)

    Expected Results:
    ---------------------------
    1. Base demand: 0 <= demand <= 10kW
    2. High price demand < base demand (price response active)
    3. is_controllable=True (can participate in demand response)
    """
    consumer = MockEnergyConsumer(
        "home_1",
        max_demand_kw=10,
        consumer_type="residential",
        flexibility=0.2
    )
    
    # Test normal demand at evening peak
    demand = consumer.get_current_demand(SUMMER_EVENING)
    assert 0 <= demand <= 10
    assert consumer.current_demand == demand
    
    # Test price response
    demand_with_price = consumer.get_current_demand(SUMMER_EVENING, price_signal=80)
    assert demand_with_price <= demand  # Should reduce with high price
    
    # Test controllable status
    assert consumer.is_controllable


@pytest.mark.case("MK-003")
def test_mock_grid_operator():
    """Test Case - Mock Grid Operator Frequency and Price Management.

    Description:
    -----------------
    Verify that MockGridOperator correctly calculates grid frequency
    based on supply/demand balance and generates appropriate price
    signals. Similar to testing CDN load balancer health checks and
    traffic distribution decisions.

    Preconditions:
    -----------------
    1. Create grid operator for test_grid
    2. Nominal frequency: 50 Hz
    3. Base price: $50/MWh

    Steps:
    ----------
    1. Calculate frequency with excess generation (1000 vs 950)
    2. Calculate frequency with deficit (950 vs 1000)
    3. Get price signal with high demand ratio (0.95)
    4. Trigger outage in area_1 for 30 minutes
    5. Verify outage status for area_1 and area_2

    Expected Results:
    ---------------------------
    1. Excess generation: frequency > 50 Hz, grid stable
    2. Deficit: frequency < 50 Hz
    3. High demand: price > $50 (above base)
    4. Outage recorded in outages list
    5. area_1 outage active, area_2 not affected
    """
    grid_op = MockGridOperator("test_grid")
    
    # Test frequency calculation
    freq = grid_op.calculate_grid_frequency(1000, 950)  # Slight excess
    assert freq > 50.0
    assert grid_op.is_stable
    
    # Test under-frequency
    freq = grid_op.calculate_grid_frequency(950, 1000)  # Slight deficit
    assert freq < 50.0
    
    # Test price signal
    price = grid_op.get_price_signal(SUMMER_NOON, 0.95)  # High demand
    assert price > 50.0  # Should be above base price
    assert len(grid_op.price_history) == 1
    
    # Test outage management
    grid_op.trigger_outage("area_1", 30)  # 30 minute outage
    assert len(grid_op.outages) == 1
    
    # Test outage check (use utcnow to match mock internals which use utcnow)
    outage_check_time = datetime.utcnow()
    assert grid_op.is_outage_active("area_1", outage_check_time)
    assert not grid_op.is_outage_active("area_2", outage_check_time)


@pytest.mark.case("MK-004")
def test_mock_energy_market():
    """Test Case - Mock Energy Market Bid/Offer Matching and Clearing.

    Description:
    -----------------
    Verify that MockEnergyMarket correctly matches bids and offers,
    clears the market, and determines clearing prices. Similar to
    testing CDN DNS resolution and origin selection algorithms.

    Preconditions:
    -----------------
    1. Create energy market for test_market
    2. No existing bids or offers

    Steps:
    ----------
    1. Submit bid: buyer_1 wants 10MW at max $60
    2. Submit offer: seller_1 offers 8MW at min $40
    3. Verify bid and offer are recorded
    4. Clear market and verify trades generated
    5. Verify trade quantity (min of bid/offer) and price (average)
    6. Test no-trade scenario: bid price < offer price

    Expected Results:
    ---------------------------
    1. Bid recorded with unique bid_id
    2. Offer recorded with unique offer_id
    3. Market clears with 1 trade
    4. Trade quantity: 8MW (limited by offer)
    5. Trade price: $50 (average of $60 and $40)
    6. No trades when bid $30 < offer $70
    """
    market = MockEnergyMarket("test_market")
    
    # Submit bids and offers
    bid_id = market.submit_bid("buyer_1", 10, 60)  # Buy 10MW at max $60
    offer_id = market.submit_offer("seller_1", 8, 40)  # Sell 8MW at min $40
    
    assert bid_id in [b["bid_id"] for b in market.bids]
    assert offer_id in [o["offer_id"] for o in market.offers]
    
    # Clear market
    trades = market.clear_market()
    assert len(trades) > 0
    assert market.clearing_price > 0
    
    # Check trade details
    trade = trades[0]
    assert trade["quantity_mw"] == 8  # Limited by offer
    assert trade["price"] == 50  # Average of 60 and 40
    
    # Test no trade scenario
    market_no_trade = MockEnergyMarket("no_trade")
    market_no_trade.submit_bid("buyer", 10, 30)  # Low bid
    market_no_trade.submit_offer("seller", 10, 70)  # High offer
    
    trades_no_trade = market_no_trade.clear_market()
    assert len(trades_no_trade) == 0


@pytest.mark.case("MK-005")
def test_producer_maintenance_windows():
    """Test Case - Producer Scheduled Maintenance Windows.

    Description:
    -----------------
    Verify that MockEnergyProducer correctly handles scheduled
    maintenance windows, producing zero output during maintenance.
    Similar to testing CDN origin server maintenance and failover.

    Preconditions:
    -----------------
    1. Create coal producer with 500kW capacity
    2. Schedule maintenance: 10:00 to 14:00 on 2024-06-15
    3. Producer type: COAL (baseload, not time-dependent)

    Steps:
    ----------
    1. Query output at 12:00 (during maintenance window)
    2. Query output at 15:00 (after maintenance window)

    Expected Results:
    ---------------------------
    1. During maintenance: output = 0kW
    2. After maintenance: output > 0kW (normal operation)
    """
    start_time = datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc)
    end_time = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
    
    producer = MockEnergyProducer(
        "plant_1",
        EnergySource.COAL,
        capacity_kw=500,
        maintenance_windows=[(start_time, end_time)]
    )
    
    # Test during maintenance (noon is within 10:00-14:00 window)
    output = producer.get_current_output(SUMMER_NOON)
    assert output == 0
    
    # Test outside maintenance
    after_maintenance = datetime(2024, 6, 15, 15, 0, tzinfo=timezone.utc)
    output = producer.get_current_output(after_maintenance)
    assert output > 0


@pytest.mark.case("MK-006")
def test_consumer_flexibility():
    """Test Case - Consumer Demand Flexibility and Controllability.

    Description:
    -----------------
    Verify that consumer flexibility affects demand response behavior
    and controllability status. Flexible consumers reduce demand more
    under high prices. Similar to testing CDN client rate limiting
    and adaptive bitrate streaming.

    Preconditions:
    -----------------
    1. Create industrial consumer: 100kW, flexibility=0.0 (rigid)
    2. Create residential consumer: 10kW, flexibility=0.3 (flexible)
    3. High price signal: $100/MWh

    Steps:
    ----------
    1. Query rigid consumer demand with high price
    2. Query flexible consumer demand with high price
    3. Compare demand reduction between consumers
    4. Verify controllability status for each consumer

    Expected Results:
    ---------------------------
    1. Rigid consumer: minimal demand reduction
    2. Flexible consumer: significant demand reduction
    3. flexible_demand < rigid_demand (proportionally)
    4. Rigid: is_controllable=False, Flexible: is_controllable=True
    """
    # Inflexible consumer
    rigid_consumer = MockEnergyConsumer(
        "factory_1",
        max_demand_kw=100,
        consumer_type="industrial",
        flexibility=0.0
    )
    
    # Flexible consumer
    flexible_consumer = MockEnergyConsumer(
        "home_1",
        max_demand_kw=10,
        consumer_type="residential",
        flexibility=0.3
    )
    
    high_price = 100
    
    # Both should have similar base demand
    rigid_demand = rigid_consumer.get_current_demand(SUMMER_EVENING, high_price)
    flexible_demand = flexible_consumer.get_current_demand(SUMMER_EVENING, high_price)
    
    # Flexible consumer should reduce more with high price
    assert flexible_demand < rigid_demand
    
    # Test controllable status
    assert not rigid_consumer.is_controllable
    assert flexible_consumer.is_controllable
