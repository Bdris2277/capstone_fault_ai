from __future__ import annotations

from struct import pack, unpack
from typing import Sequence


def decode_f32_from_u16_pair(reg_hi: int, reg_lo: int, *, float_endianness: str) -> float:
    if float_endianness == "big":
        raw = pack(">HH", reg_hi, reg_lo)
        return unpack(">f", raw)[0]
    if float_endianness == "little":
        raw = pack("<HH", reg_hi, reg_lo)
        return unpack("<f", raw)[0]
    raise ValueError("float_endianness must be 'big' or 'little'")


def registers_to_floats(registers: Sequence[int], *, float_endianness: str) -> list[float]:
    pairs = [(registers[i], registers[i + 1]) for i in range(0, len(registers), 2)]
    return [decode_f32_from_u16_pair(hi, lo, float_endianness=float_endianness) for hi, lo in pairs]


def poll_holding_registers(
    client,
    *,
    ip: str,
    unit_id: int,
    address: int,
    count: int,
    float_endianness: str,
) -> list[float] | None:
    result = client.read_holding_registers(address=address, count=count, slave=unit_id)
    if result.isError():
        print(f"[{ip}] Modbus read error: {result}")
        return None
    return registers_to_floats(result.registers, float_endianness=float_endianness)

