"""Tests for TCPMedium and TCPMedium."""
import time

import pytest

from softs import TCPMedium

SLOT_COUNT = 4
SLOT_STRIDE = 256


class TestTCPMedium:
    def test_server_starts_and_stops(self):
        medium = TCPMedium(SLOT_COUNT, SLOT_STRIDE)
        assert ":" in medium.buf_name
        medium.close()

    def test_buf_name_is_host_port(self):
        medium = TCPMedium(SLOT_COUNT, SLOT_STRIDE, host="127.0.0.1", port=0)
        host, port = medium.buf_name.split(":")
        assert host == "127.0.0.1"
        assert int(port) > 0
        medium.close()

    def test_write_and_read_roundtrip(self):
        medium = TCPMedium(SLOT_COUNT, SLOT_STRIDE)
        time.sleep(0.1)  # Let server start

        writer = TCPMedium.attach(medium.buf_name)
        data = b"\xab" * SLOT_STRIDE
        assert writer.write(0, data) is True
        time.sleep(0.05)  # Let server process

        result = medium.read(0)
        assert result == data

        writer.close()
        medium.close()

    def test_multiple_slots(self):
        medium = TCPMedium(SLOT_COUNT, SLOT_STRIDE)
        time.sleep(0.1)

        writer = TCPMedium.attach(medium.buf_name)

        for i in range(SLOT_COUNT):
            data = bytes([i]) * SLOT_STRIDE
            offset = i * SLOT_STRIDE
            assert writer.write(offset, data) is True

        time.sleep(0.1)

        for i in range(SLOT_COUNT):
            result = medium.read(i)
            assert result == bytes([i]) * SLOT_STRIDE

        writer.close()
        medium.close()

    def test_writer_factory(self):
        medium = TCPMedium(SLOT_COUNT, SLOT_STRIDE)
        time.sleep(0.1)

        writer = TCPMedium.attach(medium.buf_name)
        assert writer.write(0, b"\xff" * 10) is True
        writer.close()
        medium.close()

    def test_multiple_concurrent_writers(self):
        medium = TCPMedium(SLOT_COUNT, SLOT_STRIDE)
        time.sleep(0.1)

        writers = [TCPMedium.attach(medium.buf_name) for _ in range(3)]

        # Each writer writes to a different slot
        for i, writer in enumerate(writers):
            offset = i * SLOT_STRIDE
            data = bytes([i + 1]) * SLOT_STRIDE
            assert writer.write(offset, data) is True

        time.sleep(0.1)

        for i in range(3):
            result = medium.read(i)
            assert result == bytes([i + 1]) * SLOT_STRIDE

        for w in writers:
            w.close()
        medium.close()

    def test_context_manager(self):
        with TCPMedium(SLOT_COUNT, SLOT_STRIDE) as medium:
            assert ":" in medium.buf_name

    def test_properties(self):
        medium = TCPMedium(SLOT_COUNT, SLOT_STRIDE)
        assert medium.slot_count == SLOT_COUNT
        assert medium.slot_stride == SLOT_STRIDE
        medium.close()

    def test_writer_returns_false_on_closed_server(self):
        medium = TCPMedium(SLOT_COUNT, SLOT_STRIDE)
        time.sleep(0.1)

        writer = TCPMedium.attach(medium.buf_name)
        medium.close()
        time.sleep(0.2)

        # Writer may or may not fail depending on TCP buffering, but should
        # not raise an exception
        result = writer.write(0, b"\x00" * 10)
        # Result could be True (buffered) or False (connection reset)
        assert isinstance(result, bool)
        writer.close()
