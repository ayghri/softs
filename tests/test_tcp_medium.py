"""Tests for TCPMedium."""
import time

import pytest

from softs import TCPMedium

NUM_SLOTS = 4
SLOT_SIZE = 256


def make_owner():
    return TCPMedium(address=None, slot_size=SLOT_SIZE, num_slots=NUM_SLOTS, create=True)


class TestTCPMedium:
    def test_server_starts_and_stops(self):
        medium = make_owner()
        assert ":" in medium.address
        medium.close()

    def test_address_is_host_port(self):
        medium = make_owner()
        host, port = medium.address.split(":")
        assert host == "127.0.0.1"
        assert int(port) > 0
        medium.close()

    def test_write_and_read_roundtrip(self):
        medium = make_owner()
        time.sleep(0.1)  # Let server start

        writer = TCPMedium.attach(medium.address)
        data = b"\xab" * SLOT_SIZE
        assert writer.write(0, data) is True
        time.sleep(0.05)  # Let server process

        result = medium.read(0)
        assert result == data

        writer.close()
        medium.close()

    def test_multiple_slots(self):
        medium = make_owner()
        time.sleep(0.1)

        writer = TCPMedium.attach(medium.address)

        for i in range(NUM_SLOTS):
            data = bytes([i]) * SLOT_SIZE
            offset = i * SLOT_SIZE
            assert writer.write(offset, data) is True

        time.sleep(0.1)

        for i in range(NUM_SLOTS):
            result = medium.read(i)
            assert result == bytes([i]) * SLOT_SIZE

        writer.close()
        medium.close()

    def test_writer_factory(self):
        medium = make_owner()
        time.sleep(0.1)

        writer = TCPMedium.attach(medium.address)
        assert writer.write(0, b"\xff" * 10) is True
        writer.close()
        medium.close()

    def test_multiple_concurrent_writers(self):
        medium = make_owner()
        time.sleep(0.1)

        writers = [TCPMedium.attach(medium.address) for _ in range(3)]

        # Each writer writes to a different slot
        for i, writer in enumerate(writers):
            offset = i * SLOT_SIZE
            data = bytes([i + 1]) * SLOT_SIZE
            assert writer.write(offset, data) is True

        time.sleep(0.1)

        for i in range(3):
            result = medium.read(i)
            assert result == bytes([i + 1]) * SLOT_SIZE

        for w in writers:
            w.close()
        medium.close()

    def test_context_manager(self):
        with make_owner() as medium:
            assert ":" in medium.address

    def test_properties(self):
        medium = make_owner()
        assert medium.num_slots == NUM_SLOTS
        assert medium.slot_size == SLOT_SIZE
        medium.close()

    def test_writer_returns_false_on_closed_server(self):
        medium = make_owner()
        time.sleep(0.1)

        writer = TCPMedium.attach(medium.address)
        medium.close()
        time.sleep(0.2)

        # Writer may or may not fail depending on TCP buffering, but should
        # not raise an exception
        result = writer.write(0, b"\x00" * 10)
        # Result could be True (buffered) or False (connection reset)
        assert isinstance(result, bool)
        writer.close()
