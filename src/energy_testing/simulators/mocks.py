"""Mock components for energy system testing."""

import asyncio
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import random
import json

from ..models.energy import EnergySource, EnergyReading


class MockEnergyProducer:
    """Mock energy producer that simulates various energy sources."""
    
    def __init__(
        self,
        producer_id: str,
        source_type: EnergySource,
        capacity_kw: float,
        location: str = "default",
        failure_rate: float = 0.001,
        maintenance_windows: Optional[List[tuple]] = None,
    ):
        self.producer_id = producer_id
        self.source_type = source_type
        self.capacity_kw = capacity_kw
        self.location = location
        self.failure_rate = failure_rate
        self.maintenance_windows = maintenance_windows or []
        self.is_online = True
        self.current_output = 0.0
        self.readings_history: List[EnergyReading] = []
        
    def get_current_output(self, timestamp: datetime) -> float:
        """Get current power output based on time and conditions."""
        if not self.is_online:
            return 0.0
            
        # Check if in maintenance window
        for start, end in self.maintenance_windows:
            if start <= timestamp <= end:
                return 0.0
        
        # Simulate random failures
        if random.random() < self.failure_rate:
            self.is_online = False
            return 0.0
        
        # Calculate output based on source type
        if self.source_type == EnergySource.SOLAR:
            output = self._solar_output(timestamp)
        elif self.source_type == EnergySource.WIND:
            output = self._wind_output(timestamp)
        elif self.source_type in [EnergySource.COAL, EnergySource.GAS, EnergySource.NUCLEAR]:
            output = self._baseload_output(timestamp)
        elif self.source_type == EnergySource.HYDRO:
            output = self._hydro_output(timestamp)
        else:
            output = self.capacity_kw * random.uniform(0.8, 1.0)
        
        self.current_output = min(output, self.capacity_kw)
        
        # Record reading
        reading = EnergyReading(
            timestamp=timestamp,
            value_kw=self.current_output,
            source=self.source_type,
            location=self.location
        )
        self.readings_history.append(reading)
        
        return self.current_output
    
    def _solar_output(self, timestamp: datetime) -> float:
        """Calculate solar output based on time of day."""
        hour = timestamp.hour
        if 6 <= hour < 18:
            # Sinusoidal curve peaking at noon
            solar_factor = max(0, (hour - 6) * 3.14159 / 12)
            return self.capacity_kw * math.sin(solar_factor) * random.uniform(0.8, 1.0)
        return 0.0
    
    def _wind_output(self, timestamp: datetime) -> float:
        """Calculate wind output with variability."""
        wind_speed = random.uniform(3, 25)  # m/s
        # Simplified wind power curve
        if wind_speed < 3:
            return 0.0
        elif wind_speed > 25:
            return 0.0  # Cut-out speed
        else:
            return self.capacity_kw * (wind_speed / 15) ** 3 * random.uniform(0.9, 1.0)
    
    def _baseload_output(self, timestamp: datetime) -> float:
        """Calculate baseload plant output (stable)."""
        return self.capacity_kw * random.uniform(0.85, 1.0)
    
    def _hydro_output(self, timestamp: datetime) -> float:
        """Calculate hydro output (stable with seasonal variation)."""
        season_factor = 1.0
        month = timestamp.month
        if 6 <= month <= 8:  # Summer - lower water
            season_factor = 0.8
        elif 3 <= month <= 5 or 9 <= month <= 11:  # Spring/Fall - normal
            season_factor = 1.0
        else:  # Winter - higher water
            season_factor = 1.1
        
        return self.capacity_kw * season_factor * random.uniform(0.9, 1.0)
    
    def force_offline(self) -> None:
        """Force the producer offline for testing."""
        self.is_online = False
    
    def force_online(self) -> None:
        """Force the producer online for testing."""
        self.is_online = True


class MockEnergyConsumer:
    """Mock energy consumer that simulates various load types."""
    
    def __init__(
        self,
        consumer_id: str,
        max_demand_kw: float,
        consumer_type: str = "residential",
        location: str = "default",
        flexibility: float = 0.0,
        priority: int = 1,
    ):
        self.consumer_id = consumer_id
        self.max_demand_kw = max_demand_kw
        self.consumer_type = consumer_type
        self.location = location
        self.flexibility = flexibility
        self.priority = priority
        self.current_demand = 0.0
        self.readings_history: List[EnergyReading] = []
        self.is_controllable = flexibility > 0
        
    def get_current_demand(self, timestamp: datetime, price_signal: float = 0.0) -> float:
        """Get current power demand based on time and price."""
        base_demand = self._base_demand_pattern(timestamp)
        
        # Apply price response if controllable
        if self.is_controllable and price_signal > 0:
            price_response = self._price_response(price_signal)
            base_demand *= (1 - price_response)
        
        self.current_demand = min(base_demand, self.max_demand_kw)
        
        # Record reading
        reading = EnergyReading(
            timestamp=timestamp,
            value_kw=self.current_demand,
            source=EnergySource.GAS,  # Using GAS as placeholder for demand
            location=self.location
        )
        self.readings_history.append(reading)
        
        return self.current_demand
    
    def _base_demand_pattern(self, timestamp: datetime) -> float:
        """Calculate base demand pattern based on consumer type."""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        if self.consumer_type == "residential":
            if 7 <= hour < 9 or 17 <= hour < 21:  # Peak hours
                return self.max_demand_kw * random.uniform(0.7, 1.0)
            elif 23 <= hour or hour < 5:  # Night
                return self.max_demand_kw * random.uniform(0.1, 0.3)
            else:  # Daytime
                return self.max_demand_kw * random.uniform(0.3, 0.6)
                
        elif self.consumer_type == "commercial":
            if 8 <= hour < 18 and day_of_week < 5:  # Weekday business hours
                return self.max_demand_kw * random.uniform(0.8, 1.0)
            else:
                return self.max_demand_kw * random.uniform(0.1, 0.3)
                
        elif self.consumer_type == "industrial":
            # Industrial is more constant
            return self.max_demand_kw * random.uniform(0.7, 1.0)
        
        return self.max_demand_kw * random.uniform(0.5, 1.0)
    
    def _price_response(self, price_signal: float) -> float:
        """Calculate demand response to price signal."""
        # Simple linear response
        max_reduction = self.flexibility
        return min(max_reduction, price_signal / 100)  # Normalize price signal


class MockGridOperator:
    """Mock grid operator that manages grid operations."""
    
    def __init__(self, grid_id: str = "test_grid"):
        self.grid_id = grid_id
        self.frequency = 50.0
        self.voltage = 230.0
        self.is_stable = True
        self.outages: List[Dict] = []
        self.price_history: List[Dict] = []
        
    def calculate_grid_frequency(
        self, 
        total_generation: float, 
        total_demand: float
    ) -> float:
        """Calculate grid frequency based on supply-demand balance."""
        if total_demand == 0:
            return 50.0
        
        imbalance = (total_generation - total_demand) / total_demand
        # Simple frequency response model
        # Excess generation increases frequency, deficit decreases it
        # Using 5x multiplier so small imbalances stay within stable bounds
        self.frequency = 50.0 + imbalance * 5  # Simplified response
        
        # Check stability
        self.is_stable = 49.5 <= self.frequency <= 50.5
        
        return self.frequency
    
    def get_price_signal(self, timestamp: datetime, demand_level: float) -> float:
        """Generate price signal based on demand level."""
        # Base price
        base_price = 50.0  # $/MWh
        
        # Price increases with demand
        if demand_level > 0.9:  # High demand
            price_multiplier = random.uniform(2.0, 5.0)
        elif demand_level > 0.8:  # Medium-high demand
            price_multiplier = random.uniform(1.5, 2.0)
        elif demand_level > 0.7:  # Medium demand
            price_multiplier = random.uniform(1.0, 1.5)
        else:  # Low demand
            price_multiplier = random.uniform(0.5, 1.0)
        
        price = base_price * price_multiplier
        
        # Record price
        self.price_history.append({
            "timestamp": timestamp,
            "price": price,
            "demand_level": demand_level
        })
        
        return price
    
    def trigger_outage(self, area: str, duration_minutes: int) -> None:
        """Trigger a mock outage for testing."""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        outage = {
            "area": area,
            "start_time": start_time,
            "end_time": end_time,
            "affected_consumers": [],
            "affected_producers": []
        }
        self.outages.append(outage)
    
    def is_outage_active(self, area: str, timestamp: datetime) -> bool:
        """Check if there's an active outage in the given area."""
        for outage in self.outages:
            if (outage["area"] == area and 
                outage["start_time"] <= timestamp <= outage["end_time"]):
                return True
        return False


class MockEnergyMarket:
    """Mock energy market for trading simulations."""
    
    def __init__(self, market_id: str = "test_market"):
        self.market_id = market_id
        self.bids: List[Dict] = []
        self.offers: List[Dict] = []
        self.trades: List[Dict] = []
        self.clearing_price = 50.0
        
    def submit_bid(self, bidder_id: str, quantity_mw: float, price_limit: float) -> str:
        """Submit a bid to buy energy."""
        bid = {
            "bid_id": f"bid_{len(self.bids)}",
            "bidder_id": bidder_id,
            "quantity_mw": quantity_mw,
            "price_limit": price_limit,
            "timestamp": datetime.utcnow(),
            "status": "pending"
        }
        self.bids.append(bid)
        return bid["bid_id"]
    
    def submit_offer(self, offerer_id: str, quantity_mw: float, price_min: float) -> str:
        """Submit an offer to sell energy."""
        offer = {
            "offer_id": f"offer_{len(self.offers)}",
            "offerer_id": offerer_id,
            "quantity_mw": quantity_mw,
            "price_min": price_min,
            "timestamp": datetime.utcnow(),
            "status": "pending"
        }
        self.offers.append(offer)
        return offer["offer_id"]
    
    def clear_market(self) -> List[Dict]:
        """Clear the market and match bids with offers."""
        # Sort bids by price limit (descending)
        sorted_bids = sorted(self.bids, key=lambda x: x["price_limit"], reverse=True)
        # Sort offers by price minimum (ascending)
        sorted_offers = sorted(self.offers, key=lambda x: x["price_min"])
        
        trades = []
        bid_idx = 0
        offer_idx = 0
        
        while (bid_idx < len(sorted_bids) and 
               offer_idx < len(sorted_offers)):
            bid = sorted_bids[bid_idx]
            offer = sorted_offers[offer_idx]
            
            if bid["price_limit"] >= offer["price_min"]:
                # Trade can happen
                trade_quantity = min(bid["quantity_mw"], offer["quantity_mw"])
                trade_price = (bid["price_limit"] + offer["price_min"]) / 2
                
                trade = {
                    "trade_id": f"trade_{len(self.trades)}",
                    "bid_id": bid["bid_id"],
                    "offer_id": offer["offer_id"],
                    "quantity_mw": trade_quantity,
                    "price": trade_price,
                    "timestamp": datetime.utcnow()
                }
                trades.append(trade)
                self.trades.append(trade)
                
                # Update quantities
                bid["quantity_mw"] -= trade_quantity
                offer["quantity_mw"] -= trade_quantity
                
                if bid["quantity_mw"] <= 0:
                    bid_idx += 1
                if offer["quantity_mw"] <= 0:
                    offer_idx += 1
            else:
                break  # No more trades possible
        
        # Set clearing price
        if trades:
            self.clearing_price = sum(t["price"] for t in trades) / len(trades)
        
        return trades
