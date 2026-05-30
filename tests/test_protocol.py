"""Tests for softs.market.protocol module."""

from softs.market.protocol import (
    ClientCmd, SupplierCmd, decode_payload, encode_payload,
    make_reply, make_request, parse_request,
)


class TestPayloadEncoding:
    def test_roundtrip(self):
        data = {"key": "value", "num": 42, "nested": {"a": 1}}
        assert decode_payload(encode_payload(data)) == data

    def test_empty(self):
        assert decode_payload(encode_payload({})) == {}


class TestRequestReply:
    def test_make_request(self):
        frames = make_request(ClientCmd.HELLO, {"pid": 1234})
        assert frames[0] == ClientCmd.HELLO
        assert len(frames) == 2

    def test_parse_request(self):
        frames = make_request(ClientCmd.ORDER, {"order_id": "abc", "product_id": "v1"})
        cmd, payload = parse_request(frames)
        assert cmd == ClientCmd.ORDER
        assert payload["order_id"] == "abc"

    def test_make_reply_ok(self):
        reply = decode_payload(make_reply(True, count=5))
        assert reply["ok"] is True
        assert reply["count"] == 5

    def test_make_reply_error(self):
        reply = decode_payload(make_reply(False, error="bad"))
        assert reply["ok"] is False
        assert reply["error"] == "bad"

    def test_supplier_cmd_distinct(self):
        assert SupplierCmd.WORK == b"WORK"
        assert SupplierCmd.DONE == b"DONE"
        assert SupplierCmd.READY == b"READY"
