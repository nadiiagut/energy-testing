"""Pytest configuration and fixtures for energy-testing."""

import pytest
from datetime import datetime, timedelta, timezone

from harness import (
    TimeController,
    MeterIngestionStub,
    PaymentProviderStub,
    NotificationSink,
    ArrearsEngine,
)
from simulators.grid import EnergyGridSimulator
from simulators.mocks import (
    MockEnergyProducer,
    MockEnergyConsumer,
    MockGridOperator,
    MockEnergyMarket,
)
from models.energy import EnergySource

# Import shared constants
from tests.test_constants import (
    SUMMER_NOON,
    SUMMER_MORNING,
    SUMMER_EVENING,
    SUMMER_NIGHT,
    CHRISTMAS,
)


# =============================================================================
# Pytest configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "prio1: Priority 1 (critical) tests")
    config.addinivalue_line("markers", "prio2: Priority 2 (important) tests")
    config.addinivalue_line("markers", "prio3: Priority 3 (nice to have) tests")
    config.addinivalue_line("markers", "case(id): Test case identifier (e.g., SM-AR-001)")
    config.addinivalue_line("markers", "linux_only: Tests that require Linux (LD_PRELOAD)")


# =============================================================================
# Timestamp fixtures
# =============================================================================

@pytest.fixture
def sample_timestamp():
    """Provide a sample timestamp for testing (summer noon UTC)."""
    return SUMMER_NOON


@pytest.fixture
def morning_timestamp():
    """Provide a morning timestamp for testing (summer morning UTC)."""
    return SUMMER_MORNING


@pytest.fixture
def evening_timestamp():
    """Provide an evening timestamp for testing (summer evening UTC)."""
    return SUMMER_EVENING


@pytest.fixture
def night_timestamp():
    """Provide a night timestamp for testing (summer night UTC)."""
    return SUMMER_NIGHT


# =============================================================================
# Grid simulation fixtures
# =============================================================================

@pytest.fixture
def sample_grid():
    """Create a sample grid with various producers and consumers."""
    grid = EnergyGridSimulator(time_step_seconds=300)
    grid.add_producer("solar_large", EnergySource.SOLAR, capacity_kw=2000)
    grid.add_producer("wind_coastal", EnergySource.WIND, capacity_kw=1500)
    grid.add_producer("coal_baseload", EnergySource.COAL, capacity_kw=1000)
    grid.add_producer("gas_peaker", EnergySource.GAS, capacity_kw=500)
    grid.add_consumer("residential_suburbs", max_demand_kw=800, consumer_type="residential")
    grid.add_consumer("commercial_downtown", max_demand_kw=1200, consumer_type="commercial")
    grid.add_consumer("industrial_zone", max_demand_kw=1500, consumer_type="industrial")
    return grid


@pytest.fixture
def mock_producers():
    """Create a set of mock producers for testing."""
    return {
        "solar": MockEnergyProducer("solar_1", EnergySource.SOLAR, 1000),
        "wind": MockEnergyProducer("wind_1", EnergySource.WIND, 800),
        "coal": MockEnergyProducer("coal_1", EnergySource.COAL, 1200),
    }


@pytest.fixture
def mock_consumers():
    """Create a set of mock consumers for testing."""
    return {
        "residential": MockEnergyConsumer("home_1", 50, "residential", flexibility=0.3),
        "commercial": MockEnergyConsumer("office_1", 200, "commercial", flexibility=0.1),
        "industrial": MockEnergyConsumer("factory_1", 500, "industrial", flexibility=0.05),
    }


@pytest.fixture
def mock_grid_operator():
    """Create a mock grid operator for testing."""
    return MockGridOperator("test_grid")


@pytest.fixture
def mock_energy_market():
    """Create a mock energy market for testing."""
    return MockEnergyMarket("test_market")


# =============================================================================
# Arrears harness fixtures
# =============================================================================

@pytest.fixture
def time_ctl():
    """TimeController for deterministic time manipulation."""
    return TimeController()


@pytest.fixture
def ledger():
    """Shared ledger state dictionary."""
    return {}


@pytest.fixture
def stubs(ledger):
    """Service stubs (ingestion, payments, notification sink)."""
    return {
        "ingestion": MeterIngestionStub(ledger),
        "payments": PaymentProviderStub(ledger),
        "sink": NotificationSink(),
    }


@pytest.fixture
def engine():
    """ArrearsEngine with default config."""
    return ArrearsEngine()
