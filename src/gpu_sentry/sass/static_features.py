from __future__ import annotations

BITWISE_OPS = {
    "AND",
    "BFE",
    "BFI",
    "BREV",
    "FLO",
    "IADD3",
    "IMAD",
    "ISCADD",
    "LOP",
    "LOP3",
    "POPC",
    "PRMT",
    "SHF",
    "SHL",
    "SHR",
    "VABSDIFF",
    "XOR",
}
INTEGER_OPS = {
    "I2F",
    "I2I",
    "IABS",
    "IADD",
    "IADD3",
    "ICMP",
    "IDP",
    "IMAD",
    "IMNMX",
    "IMUL",
    "ISCADD",
    "ISETP",
    "LEA",
    "LOP",
    "LOP3",
    "POPC",
    "SHF",
    "SHL",
    "SHR",
    "VADD",
}
def opcode_from_normalized(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped.endswith(":") or stripped == "KERNEL_BOUNDARY":
        return ""
    token = stripped.split(maxsplit=1)[0]
    token = token.split(".", 1)[0]
    token = token.lstrip("@!PT0123456789")
    return token.upper()


def is_bitwise_integer_op(opcode: str) -> bool:
    return opcode in BITWISE_OPS or opcode in INTEGER_OPS


def bitwise_integer_instruction_ratio(lines: list[str]) -> float:
    ops = [opcode_from_normalized(line) for line in lines]
    ops = [op for op in ops if op]
    if not ops:
        return 0.0
    return sum(1 for op in ops if is_bitwise_integer_op(op)) / len(ops)
