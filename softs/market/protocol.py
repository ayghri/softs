"""Wire protocol for broker ↔ client/supplier communication."""

from dataclasses import dataclass, field

import msgpack
import yaml


@dataclass
class EndpointConfig:
    frontend: str = "ipc:///tmp/softlabels_frontend.sock"
    backend: str = "ipc:///tmp/softlabels_backend.sock"

    @classmethod
    def from_yaml(cls, path: str) -> "EndpointConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


class ClientCmd:
    """Commands on the client ↔ broker channel (frontend)."""

    HELLO = b"HELLO"
    ORDER = b"ORDER"
    CANCEL = b"CANCEL"
    DISCARD = b"DISCARD"
    STATS = b"STATS"
    FULFILLED = b"FULFILLED"  # broker → client


class SupplierCmd:
    """Commands on the supplier ↔ broker channel (backend)."""

    HELLO = b"HELLO"
    GOODBYE = b"GOODBYE"
    READY = b"READY"
    DONE = b"DONE"
    WORK = b"WORK"  # broker → supplier
    TERMINATE = b"TERMINATE"  # broker → supplier


class OrderState:
    QUEUED = "queued"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# -- Peer tracking --


@dataclass
class ClientInfo:
    peer_id: int
    last_seen: float = 0.0


@dataclass
class SupplierInfo:
    peer_id: int
    product_ids: list[str] = field(default_factory=list)
    last_seen: float = 0.0


# -- Message payloads (validated via dataclass construction) --


@dataclass
class OrderRequest:
    """Client places an order for a product at a specific address/offset."""

    order_id: str
    product_id: str
    address: str
    offset: int


@dataclass
class OrderCancel:
    order_id: str


@dataclass
class OrderWork:
    """Broker assigns work to a supplier."""

    order_id: str
    product_id: str
    address: str
    offset: int


@dataclass
class OrderDone:
    """Supplier reports completion."""

    order_id: str
    success: bool = True


@dataclass
class Order:
    order_id: str
    client_id: bytes
    product_id: str
    address: str
    offset: int
    state: str = OrderState.QUEUED
    supplier_id: bytes = b""
    dispatched_at: float = 0.0


@dataclass
class BrokerStats:
    pending_orders: int
    available_suppliers: int
    connected_clients: int
    connected_suppliers: int
    total_completed: int
    product_ids: list[str]


# -- Wire encoding --


def encode_payload(data: dict) -> bytes:
    return msgpack.packb(data, use_bin_type=True)


def decode_payload(data: bytes) -> dict:
    return msgpack.unpackb(data, raw=False)


def make_request(cmd: bytes, payload: dict) -> list[bytes]:
    return [cmd, encode_payload(payload)]


def parse_request(frames: list[bytes]) -> tuple[bytes, dict]:
    if len(frames) < 2:
        raise ValueError(f"Expected at least 2 frames, got {len(frames)}")
    return frames[0], decode_payload(frames[1]) if frames[1] else {}


def make_reply(ok: bool, error: str | None = None, **data) -> bytes:
    return encode_payload({"ok": ok, "error": error, **data})
