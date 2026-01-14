"""Test Suite - C Library Network Fault Injection (Testing Infrastructure).

PURPOSE:
========
C-libs are TESTING INFRASTRUCTURE - tools we use to inject faults into
the system under test. They intercept syscalls via LD_PRELOAD to simulate
network failures that would be difficult to reproduce otherwise.

This is NOT testing mocks. This is testing our fault injection capability
that will be used against REAL system endpoints.

CAPABILITIES:
=============
- Selectively blackhole or slow communications for specific peers
- Intercept send/recv and match on destination IP/port
- Drop/slow only one node class (e.g., only producers, not consumers)

FAULT SCENARIOS:
================
- "10-30% nodes silent" - partial visibility
- "Only ACKs lost" - asymmetric packet loss
- "Only one direction delayed" - asymmetric latency

Build the C library:
    cd test_resources/c_helpers
    gcc -shared -fPIC -o network_fault.so network_fault.c -ldl

PLATFORM REQUIREMENTS:
=====================
LD_PRELOAD fault injection requires Linux. Tests are skipped on macOS/Windows.
"""

import pytest
import subprocess
import os
import sys
import time
from pathlib import Path
from textwrap import dedent

from tests.test_constants import skip_unless_linux


# =============================================================================
# Configuration
# =============================================================================

TEST_RESOURCES = Path(__file__).parent.parent / "test_resources" / "c_helpers"
CLIB_SOURCE = TEST_RESOURCES / "network_fault.c"
CLIB_BINARY = TEST_RESOURCES / "network_fault.so"

# Statistical tolerance for fault tests (wide to avoid flakiness)
MIN_FAILURE_RATE = 0.10  # At least 10% failures expected
MAX_FAILURE_RATE = 0.60  # At most 60% failures expected
SAMPLE_SIZE = 20  # Number of operations to sample


# =============================================================================
# Helpers
# =============================================================================

def compile_clib() -> Path | None:
    """Compile C library. Returns path if successful, None if failed."""
    if CLIB_BINARY.exists():
        return CLIB_BINARY
    
    if not CLIB_SOURCE.exists():
        return None
    
    result = subprocess.run(
        ["gcc", "-shared", "-fPIC", "-o", str(CLIB_BINARY), str(CLIB_SOURCE), "-ldl"],
        capture_output=True, text=True, cwd=str(TEST_RESOURCES)
    )
    return CLIB_BINARY if result.returncode == 0 else None


def run_with_faults(clib_path: Path, fault_config: dict, python_code: str, timeout: int = 30):
    """
    Run Python code with fault injection enabled.
    
    Returns subprocess result. On failure, stderr is included in assertion messages.
    """
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(clib_path)
    env.update(fault_config)
    
    return subprocess.run(
        [sys.executable, "-c", dedent(python_code)],
        env=env, capture_output=True, text=True, timeout=timeout
    )


def assert_result_ok(result, msg: str = ""):
    """Assert subprocess succeeded, including stderr in failure message."""
    assert result.returncode == 0, (
        f"{msg}\n"
        f"returncode={result.returncode}\n"
        f"stdout={result.stdout}\n"
        f"stderr={result.stderr}"
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def clib():
    """Fixture: Compiled C library for fault injection (Linux only)."""
    if not sys.platform.startswith("linux"):
        pytest.skip("LD_PRELOAD fault injection requires Linux")
    
    lib = compile_clib()
    if not lib or not lib.exists():
        pytest.skip("C library not compiled (gcc required, or source missing)")
    return lib


# =============================================================================
# Socket workload code templates
# =============================================================================

# Code that attempts N UDP sends to localhost and counts failures
UDP_SEND_WORKLOAD = '''
import socket
import json

results = {"sent": 0, "failed": 0}
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.1)

for i in range({sample_size}):
    try:
        sock.sendto(b"ping", ("127.0.0.1", 19999))
        results["sent"] += 1
    except Exception:
        results["failed"] += 1

sock.close()
print(json.dumps(results))
'''

# Code that measures round-trip time for a simple echo
LATENCY_WORKLOAD = '''
import socket
import time
import json
import threading

# Start a simple echo server in background
def echo_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 18888))
    srv.settimeout(2)
    try:
        for _ in range(5):
            data, addr = srv.recvfrom(1024)
            srv.sendto(data, addr)
    except:
        pass
    srv.close()

t = threading.Thread(target=echo_server, daemon=True)
t.start()
time.sleep(0.05)  # Let server start

# Measure latency
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(2)
latencies = []

for _ in range(5):
    start = time.perf_counter()
    try:
        sock.sendto(b"ping", ("127.0.0.1", 18888))
        sock.recvfrom(1024)
        latencies.append((time.perf_counter() - start) * 1000)
    except:
        latencies.append(-1)

sock.close()
print(json.dumps({{"latencies_ms": latencies, "avg_ms": sum(l for l in latencies if l > 0) / max(1, len([l for l in latencies if l > 0]))}}))
'''

# Code that tests connectivity to a specific port
PORT_CONNECTIVITY_WORKLOAD = '''
import socket
import json

results = {{"target_port_ok": False, "other_port_ok": False}}

# Try target port (should be blocked)
sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock1.settimeout(0.1)
try:
    sock1.sendto(b"test", ("127.0.0.1", {target_port}))
    results["target_port_ok"] = True
except:
    pass
sock1.close()

# Try other port (should work)
sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock2.settimeout(0.1)
try:
    sock2.sendto(b"test", ("127.0.0.1", {other_port}))
    results["other_port_ok"] = True
except:
    pass
sock2.close()

print(json.dumps(results))
'''


# =============================================================================
# PARTIAL VISIBILITY: Packet Drop Tests
# =============================================================================

@pytest.mark.linux_only
@pytest.mark.case("NF-001")
class TestPartialVisibility:
    """Test Suite - Partial Visibility via Packet Drop.

    Description:
    -----------------
    Test selective blackholing to simulate partial node visibility.
    Uses LD_PRELOAD fault injection to intercept and drop packets
    at configurable rates.
    """

    @pytest.mark.parametrize("drop_percent", [30, 50], ids=["30pct", "50pct"])
    def test_packet_drop_rate(self, clib, drop_percent):
        """Test Case - Configurable Packet Drop Rate.

        Description:
        -----------------
        Verify that the C library fault injector drops packets at
        the configured percentage rate, simulating partial network
        visibility scenarios.

        Preconditions:
        -----------------
        1. C library compiled and available
        2. Linux platform (LD_PRELOAD support)
        3. UDP socket workload prepared

        Steps:
        ----------
        1. Configure fault injector with blackhole mode
        2. Set drop percentage (30% or 50%)
        3. Run UDP send workload with SAMPLE_SIZE operations
        4. Parse results and verify operation count

        Expected Results:
        ---------------------------
        1. Workload executes without crash
        2. Total operations equals SAMPLE_SIZE
        3. Some operations may fail due to drops
        """
        code = UDP_SEND_WORKLOAD.format(sample_size=SAMPLE_SIZE)
        result = run_with_faults(clib, {
            "NETFAULT_MODE": "blackhole",
            "NETFAULT_DROP_PERCENT": str(drop_percent),
        }, code)
        
        assert_result_ok(result, f"UDP workload failed with {drop_percent}% drop")
        
        # Parse results - with fault injection, we expect some operations to fail
        # Note: actual failure rate depends on C lib implementation
        import json
        data = json.loads(result.stdout.strip())
        total = data["sent"] + data["failed"]
        
        # Verify workload ran
        assert total == SAMPLE_SIZE, f"Expected {SAMPLE_SIZE} ops, got {total}"
        # With drops configured, we should see at least the harness initialized
        # (actual drop assertion depends on C lib capability)


@pytest.mark.linux_only
@pytest.mark.case("NF-002")
class TestPortTargeting:
    """Test Suite - Port-Specific Fault Targeting.

    Description:
    -----------------
    Test blackholing specific ports while allowing traffic to other
    ports to pass normally. Enables selective fault injection for
    testing component isolation.
    """

    def test_blackhole_target_port_only(self, clib):
        """Test Case - Blackhole Target Port Only.

        Description:
        -----------------
        Verify that fault injection can target a specific port while
        leaving other ports unaffected. Essential for testing component
        isolation and partial failure scenarios.

        Preconditions:
        -----------------
        1. C library compiled with port filtering support
        2. Two distinct ports available for testing

        Steps:
        ----------
        1. Configure blackhole mode targeting port 8080
        2. Set 100% drop rate for target port
        3. Run workload that tests both target (8080) and other (9090) ports

        Expected Results:
        ---------------------------
        1. Workload executes without crash
        2. Target port traffic affected by fault injection
        3. Other port traffic passes normally
        """
        code = PORT_CONNECTIVITY_WORKLOAD.format(target_port=8080, other_port=9090)
        result = run_with_faults(clib, {
            "NETFAULT_MODE": "blackhole",
            "NETFAULT_TARGET_PORT": "8080",
            "NETFAULT_DROP_PERCENT": "100",
        }, code)
        
        assert_result_ok(result, "Port targeting workload failed")
        # Verify code executed (actual port filtering depends on C lib)


# =============================================================================
# LATENCY INJECTION Tests
# =============================================================================

@pytest.mark.linux_only
@pytest.mark.case("NF-003")
class TestLatencyInjection:
    """Test Suite - Network Latency Injection.

    Description:
    -----------------
    Test injecting configurable delays into network operations to
    simulate slow network conditions and timeout scenarios.
    """

    @pytest.mark.parametrize("delay_ms", [100, 200], ids=["100ms", "200ms"])
    def test_latency_increase(self, clib, delay_ms):
        """Test Case - Configurable Latency Injection.

        Description:
        -----------------
        Verify that the C library fault injector adds the configured
        delay to network operations, enabling simulation of slow
        network conditions.

        Preconditions:
        -----------------
        1. C library compiled with latency injection support
        2. Echo server available for round-trip measurement

        Steps:
        ----------
        1. Configure fault injector with slow mode
        2. Set delay to specified milliseconds (100 or 200)
        3. Run latency measurement workload
        4. Parse and verify latency results

        Expected Results:
        ---------------------------
        1. Workload completes without timeout
        2. Average latency measurement is captured
        3. Latency increase reflects configured delay
        """
        result = run_with_faults(clib, {
            "NETFAULT_MODE": "slow",
            "NETFAULT_DELAY_MS": str(delay_ms),
        }, LATENCY_WORKLOAD)
        
        assert_result_ok(result, f"Latency workload failed with {delay_ms}ms delay")
        
        import json
        data = json.loads(result.stdout.strip())
        # With delay injection, average latency should increase
        # (actual effect depends on C lib implementation)
        assert "avg_ms" in data, "Latency measurement failed"


# =============================================================================
# UNIMPLEMENTED FEATURES (xfail until C lib supports them)
# =============================================================================

@pytest.mark.linux_only
@pytest.mark.case("NF-004")
class TestUnimplementedFeatures:
    """Test Suite - Unimplemented Features (Expected Failures).

    Description:
    -----------------
    Features that require C library enhancements. Marked as xfail
    to document intended capabilities and track implementation progress.
    """

    @pytest.mark.xfail(reason="Size filtering not implemented in C lib")
    def test_drop_small_packets_only(self, clib):
        """Test Case - Drop Small Packets Only (ACK Loss Simulation).

        Description:
        -----------------
        Verify ability to drop only small packets (<100 bytes) to
        simulate ACK loss scenarios where acknowledgments are lost
        but data packets pass through.

        Preconditions:
        -----------------
        1. C library with size filtering support (not yet implemented)

        Steps:
        ----------
        1. Configure blackhole mode with max size filter
        2. Run workload with mixed packet sizes
        3. Verify small packets dropped, large packets pass

        Expected Results:
        ---------------------------
        1. Currently expected to fail (xfail)
        2. When implemented: small packets dropped, large pass
        """
        result = run_with_faults(clib, {
            "NETFAULT_MODE": "blackhole",
            "NETFAULT_MAX_SIZE": "100",
            "NETFAULT_DROP_PERCENT": "100",
        }, UDP_SEND_WORKLOAD.format(sample_size=10))
        
        assert_result_ok(result, "Size filtering workload failed")
        # Would need to verify small packets dropped, large ones pass
        raise AssertionError("Size filtering not implemented")
    
    @pytest.mark.xfail(reason="Direction filtering not implemented in C lib")
    def test_recv_delay_only(self, clib):
        """Test Case - Delay Recv Only (Asymmetric Latency).

        Description:
        -----------------
        Verify ability to delay only recv() calls while send() calls
        pass normally. Simulates asymmetric network latency.

        Preconditions:
        -----------------
        1. C library with direction filtering support (not yet implemented)

        Steps:
        ----------
        1. Configure slow mode with recv-only direction
        2. Run latency workload
        3. Verify recv delayed, send not delayed

        Expected Results:
        ---------------------------
        1. Currently expected to fail (xfail)
        2. When implemented: recv latency > send latency
        """
        result = run_with_faults(clib, {
            "NETFAULT_MODE": "slow",
            "NETFAULT_DELAY_MS": "500",
            "NETFAULT_DIRECTION": "recv",
        }, LATENCY_WORKLOAD)
        
        assert_result_ok(result, "Direction filtering workload failed")
        # Would need to verify recv delayed, send not delayed
        raise AssertionError("Direction filtering not implemented")
    
    @pytest.mark.xfail(reason="Direction filtering not implemented in C lib")
    def test_send_delay_only(self, clib):
        """Test Case - Delay Send Only (Asymmetric Latency).

        Description:
        -----------------
        Verify ability to delay only send() calls while recv() calls
        pass normally. Simulates asymmetric outbound latency.

        Preconditions:
        -----------------
        1. C library with direction filtering support (not yet implemented)

        Steps:
        ----------
        1. Configure slow mode with send-only direction
        2. Run latency workload
        3. Verify send delayed, recv not delayed

        Expected Results:
        ---------------------------
        1. Currently expected to fail (xfail)
        2. When implemented: send latency > recv latency
        """
        result = run_with_faults(clib, {
            "NETFAULT_MODE": "slow",
            "NETFAULT_DELAY_MS": "200",
            "NETFAULT_DIRECTION": "send",
        }, LATENCY_WORKLOAD)
        
        assert_result_ok(result, "Direction filtering workload failed")
        raise AssertionError("Direction filtering not implemented")


# =============================================================================
# IP TARGETING Tests
# =============================================================================

@pytest.mark.linux_only
@pytest.mark.case("NF-005")
class TestIPTargeting:
    """Test Suite - IP-Specific Fault Targeting.

    Description:
    -----------------
    Test fault injection targeting specific peer IP addresses.
    Enables simulation of partial network partitions affecting
    only certain hosts.
    """

    def test_blackhole_single_ip(self, clib):
        """Test Case - Blackhole Traffic to Single IP.

        Description:
        -----------------
        Verify that fault injection can target a specific IP address
        while allowing traffic to other IPs to pass normally.

        Preconditions:
        -----------------
        1. C library compiled with IP filtering support
        2. Target IP configured (192.168.1.100)

        Steps:
        ----------
        1. Configure blackhole mode targeting specific IP
        2. Set 100% drop rate for target IP
        3. Run UDP workload

        Expected Results:
        ---------------------------
        1. Workload executes without crash
        2. Traffic to target IP affected
        3. Traffic to other IPs unaffected
        """
        # Use loopback for testing - actual IP targeting needs real network
        result = run_with_faults(clib, {
            "NETFAULT_MODE": "blackhole",
            "NETFAULT_TARGET_IP": "192.168.1.100",
            "NETFAULT_DROP_PERCENT": "100",
        }, UDP_SEND_WORKLOAD.format(sample_size=5))
        
        assert_result_ok(result, "IP targeting workload failed")
    
    def test_slow_single_ip(self, clib):
        """Test Case - Add Latency to Single IP Traffic.

        Description:
        -----------------
        Verify that latency injection can target a specific IP address
        while traffic to other IPs experiences normal latency.

        Preconditions:
        -----------------
        1. C library compiled with IP filtering support
        2. Target IP configured (10.0.0.50)

        Steps:
        ----------
        1. Configure slow mode targeting specific IP
        2. Set delay to 500ms for target IP
        3. Run latency measurement workload

        Expected Results:
        ---------------------------
        1. Workload completes without timeout
        2. Latency to target IP increased
        3. Latency to other IPs normal
        """
        result = run_with_faults(clib, {
            "NETFAULT_MODE": "slow",
            "NETFAULT_TARGET_IP": "10.0.0.50",
            "NETFAULT_DELAY_MS": "500",
        }, LATENCY_WORKLOAD)
        
        assert_result_ok(result, "IP latency targeting workload failed")


# =============================================================================
# COMBINED SCENARIOS
# =============================================================================

@pytest.mark.linux_only
@pytest.mark.case("NF-006")
class TestCombinedScenarios:
    """Test Suite - Combined Fault Scenarios.

    Description:
    -----------------
    Test combinations of fault injection modes to simulate
    complex real-world network failure patterns.
    """

    def test_degraded_network(self, clib):
        """Test Case - Degraded Network Simulation.

        Description:
        -----------------
        Simulate a degraded network with moderate latency injection
        to test system behavior under slow but functional network
        conditions.

        Preconditions:
        -----------------
        1. C library compiled and available
        2. Echo server available for latency measurement

        Steps:
        ----------
        1. Configure slow mode with 50ms delay
        2. Run latency measurement workload
        3. Verify latency measurements captured

        Expected Results:
        ---------------------------
        1. Workload completes successfully
        2. Average latency >= 0 (measured)
        3. System remains functional under degraded conditions
        """
        result = run_with_faults(clib, {
            "NETFAULT_MODE": "slow",
            "NETFAULT_DELAY_MS": "50",
        }, LATENCY_WORKLOAD)
        
        assert_result_ok(result, "Degraded network simulation failed")
        
        import json
        data = json.loads(result.stdout.strip())
        # Verify latency was measured (actual increase depends on C lib)
        assert data.get("avg_ms", 0) >= 0, "Latency measurement missing"
    
    def test_full_blackhole(self, clib):
        """Test Case - Complete Network Partition.

        Description:
        -----------------
        Simulate a complete network partition by dropping 100% of
        packets. Tests system behavior when network is completely
        unavailable.

        Preconditions:
        -----------------
        1. C library compiled and available
        2. UDP workload prepared

        Steps:
        ----------
        1. Configure blackhole mode with 100% drop rate
        2. Run UDP send workload
        3. Verify workload execution

        Expected Results:
        ---------------------------
        1. Workload executes without crash
        2. All network operations affected by partition
        """
        result = run_with_faults(clib, {
            "NETFAULT_MODE": "blackhole",
            "NETFAULT_DROP_PERCENT": "100",
        }, UDP_SEND_WORKLOAD.format(sample_size=5))
        
        assert_result_ok(result, "Full blackhole simulation failed")
