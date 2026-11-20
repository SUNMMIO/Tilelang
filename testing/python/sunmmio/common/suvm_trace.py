"""Scalar/SCF trace of generated SUVM for compiler tests.

Descriptor construction and signal observations are modeled; communication
completion, DMA execution, and device timing are outside this test helper.
"""

import re


def trace_suvm_control_flow(result, source, *, signal_values=None, max_loop_iterations=100, trace_memory=False, **arguments):
    """Trace SSA/control flow and descriptor values, without simulating completion.

    Memory views in these fixtures are one-dimensional row-major. Unknown ops
    fail the trace so unsupported semantics cannot silently hide a regression.
    """
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    function = next(i for i, line in enumerate(lines) if line.startswith("func.func"))

    def parse(pos):
        block = []
        while not lines[pos].startswith("}"):
            line = lines[pos]
            if any(op in line for op in ("scf.if ", "scf.for ", "scf.while ")):
                body, pos = parse(pos + 1)
                other = []
                if lines[pos].startswith(("} else {", "} do {")):
                    other, pos = parse(pos + 1)
                block.append((line, body, other))
            else:
                block.append((line, [], []))
            pos += 1
        return block, pos

    program, _ = parse(function + 1)
    func = next(iter(result.device_mod.functions.values()))
    env = {}
    rank_index = int(func.attrs.get("tl.dist.rank_id_param_index", -1))
    for i, param in enumerate(func.params):
        name = param.name
        if i == rank_index:
            value = arguments["rank_id"]
        elif str(param.dtype) == "handle":
            value = arguments.get(name, {"name": name, "data": [0] * 256})
        else:
            assert name in arguments, f"missing final device ABI argument {name}"
            value = arguments[name]
        env[f"%arg{i}"] = value
    batches, events = [], []
    fetch_counts = {}
    value_pattern = r"%[\w-]+(?:#\d+)?"

    def signed(value, width):
        value &= (1 << width) - 1
        return value - (1 << width) if value & (1 << (width - 1)) else value

    def quotient(a, b):
        assert b != 0, "trace division by zero"
        value = abs(a) // abs(b)
        return -value if (a < 0) != (b < 0) else value

    def execute(block, values):
        for line, body, other in block:
            lhs, op = line.split(" = ", 1) if " = " in line and line.startswith("%") else (None, line)
            if op.startswith("^bb"):
                continue
            refs = re.findall(value_pattern, op)
            args = [] if op.startswith(("scf.for", "scf.while")) else [values[name] for name in refs]
            if op.startswith("scf.condition"):
                return ("condition", bool(args[0]), tuple(args[1:]))
            if op.startswith("scf.yield"):
                return tuple(args)
            if op.startswith("return"):
                return ()
            if op.startswith("scf.if"):
                output = execute(body if args[0] else other, values.copy())
            elif op.startswith("scf.for"):
                bindings = (
                    re.findall(r"(%[\w-]+) = (%[\w-]+(?:#\d+)?)", op.split("iter_args(", 1)[1].split(")", 1)[0])
                    if "iter_args(" in op
                    else []
                )
                bounds = re.search(r"scf.for (%[\w-]+) = (%[\w-]+) to (%[\w-]+) step (%[\w-]+)", op)
                carried = tuple(values[initial] for _, initial in bindings)
                assert "unsigned" not in op, "unsupported trace unsigned scf.for"
                lower, upper, step = (signed(values[bounds[i]], 64) for i in (2, 3, 4))
                assert step > 0, "trace scf.for requires a positive step"
                for iv in range(lower, upper, step):
                    local = values.copy()
                    local[bounds[1]] = iv
                    local.update((name, value) for (name, _), value in zip(bindings, carried))
                    carried = execute(body, local)
                    assert iv + step <= (1 << 63) - 1, "trace scf.for index increment overflow"
                output = carried
            elif op.startswith("scf.while"):
                bindings = re.findall(r"(%[\w-]+) = (%[\w-]+(?:#\d+)?)", op.split(" : ", 1)[0])
                carried = tuple(values[initial] for _, initial in bindings)
                after = re.findall(r"(%[\w-]+):", other[0][0])
                for _iteration in range(max_loop_iterations):
                    local = values.copy()
                    local.update((name, value) for (name, _), value in zip(bindings, carried))
                    marker, condition, forwarded = execute(body, local)
                    assert marker == "condition"
                    if not condition:
                        output = forwarded
                        break
                    local = values.copy()
                    local.update(zip(after, forwarded))
                    carried = execute(other, local)
                else:
                    raise AssertionError("trace loop did not terminate")
            elif op.startswith("arith.constant"):
                token = op.split()[1]
                output = token == "true" if token in ("true", "false") else int(token)
                integer_type = re.search(r": i(\d+)\b", op)
                if integer_type:
                    output &= (1 << int(integer_type[1])) - 1
            elif op.startswith(("arith.index_cast", "arith.extui", "arith.extsi")):
                cast = re.search(r": (i\d+|index) to (i\d+|index)", op)
                source_type, target_type = cast.groups()
                source_width = 64 if source_type == "index" else int(source_type[1:])
                value = args[0] & ((1 << source_width) - 1)
                if not op.startswith("arith.extui") and value & (1 << (source_width - 1)):
                    value -= 1 << source_width
                target_width = 64 if target_type == "index" else int(target_type[1:])
                output = value & ((1 << target_width) - 1)
            elif op.startswith("arith.trunci"):
                width = int(re.search(r"to i(\d+)", op)[1])
                output = args[0] & ((1 << width) - 1)
            elif op.startswith("arith.cmpi"):
                pred = op.split()[1].rstrip(",")
                integer_type = re.search(r": i(\d+)\b", op)
                width = int(integer_type[1]) if integer_type else 64
                operands = [signed(value, width) if pred.startswith("s") else value & ((1 << width) - 1) for value in args]
                comparison = pred[1:] if pred.startswith(("s", "u")) else pred
                output = {
                    "eq": lambda a, b: a == b,
                    "ne": lambda a, b: a != b,
                    "gt": lambda a, b: a > b,
                    "lt": lambda a, b: a < b,
                    "ge": lambda a, b: a >= b,
                    "le": lambda a, b: a <= b,
                }[comparison](*operands)
            elif op.startswith("arith.select"):
                output = args[1] if args[0] else args[2]
            elif op.startswith("arith."):
                operator = op.split()[0].split(".")[1]
                integer_type = re.search(r": i(\d+)\s*$", op)
                width = int(integer_type[1]) if integer_type else 64
                if operator in ("divsi", "remsi", "floordivsi"):
                    args = [signed(value, width) for value in args]
                    assert args[1] != 0, "trace division by zero"
                    assert operator == "remsi" or not (args[0] == -(1 << (width - 1)) and args[1] == -1), "trace signed division overflow"
                elif operator in ("divui", "remui"):
                    args = [value & ((1 << width) - 1) for value in args]
                    assert args[1] != 0, "trace division by zero"
                elif operator in ("maxsi", "minsi"):
                    args = [signed(value, width) for value in args]
                output = {
                    "addi": lambda a, b: a + b,
                    "subi": lambda a, b: a - b,
                    "muli": lambda a, b: a * b,
                    "maxsi": max,
                    "minsi": min,
                    "remsi": lambda a, b: a - quotient(a, b) * b,
                    "remui": lambda a, b: a % b,
                    "divsi": quotient,
                    "divui": lambda a, b: a // b,
                    "floordivsi": lambda a, b: a // b,
                    "andi": lambda a, b: a & b,
                    "ori": lambda a, b: a | b,
                    "xori": lambda a, b: a ^ b,
                    "shli": lambda a, b: a << b,
                    "shrsi": lambda a, b, width=width: signed(a, width) >> b,
                    "shrui": lambda a, b, width=width: (a & ((1 << width) - 1)) >> b,
                }[operator](*args)
            elif op.startswith("suvm.get_core_id"):
                output = 0
            elif op.startswith("suvm.dist.signal :"):
                kind, space = re.search(r"kind = (\w+), space = (\w+)", op).groups()
                output = (kind, space)
            elif op.startswith("suvm.dist.signal_fetch"):
                (kind, space), index = args
                key = (kind, space, index)
                observations = (signal_values or {}).get(key, (signal_values or {}).get(index))
                assert observations is not None, f"unexpected signal fetch {key}"
                if isinstance(observations, int):
                    observations = [observations]
                number = fetch_counts.get(key, 0)
                output = observations[min(number, len(observations) - 1)]
                fetch_counts[key] = number + 1
                assert 0 <= output < (1 << (64 if kind == "memory" else 8))
                events.append(("fetch", kind, space, index, output))
            elif op.startswith("suvm.dist.batch_begin"):
                output = []
            elif op.startswith("suvm.alloc"):
                logical = int(re.search(r"memtensor<(\d+)x", op)[1])
                covered = int(re.search(r"#suvm.layout<\((\d+)\), \(1\)>", op)[1])
                output = {"name": lhs, "data": [0] * max(logical, covered)}
            elif op.startswith("suvm.get_partitioned_tile_view"):
                width = int(re.search(r"-> !suvm.tile_view<(\d+)x", op)[1])
                output = (args[0], args[1] * width, width)
            elif op.startswith("suvm.tile.load"):
                memory, begin, width = args[0]
                assert begin >= 0 and begin + width <= len(memory["data"]), "unexpected address evaluation"
                output = memory["data"][begin : begin + width].copy()
            elif op.startswith("suvm.tile.fill"):
                width = int(re.search(r"-> !suvm.tile<(\d+)x", op)[1])
                output = [args[0]] * width
            elif op.startswith("suvm.tile.pick"):
                lane = args[1] if len(args) == 2 else int(re.search(r"\[(-?\d+)\]", op)[1])
                assert 0 <= lane < len(args[0]), "unexpected tile lane"
                output = args[0][lane]
                events.append(("pick", output))
            elif op.startswith("suvm.tile.set"):
                output = args[1].copy()
                lane = args[2] if len(args) == 3 else int(re.search(r"\[(-?\d+)\]", op)[1])
                assert 0 <= lane < len(output), "unexpected tile lane"
                output[lane] = args[0]
                if trace_memory:
                    events.append(("set", lane, args[0]))
            elif op.startswith("suvm.tile.store"):
                memory, begin, width = args[1]
                memory["data"][begin : begin + width] = args[0]
                if trace_memory:
                    events.append(("store", memory["name"], begin, width))
                continue
            elif op.startswith("suvm.dist.batch_put"):
                batch, peer, src, dst = args
                item = ("data", peer, src[0]["name"], src[1], dst[0]["name"], dst[1], src[2])
                batch.append(item)
                events.append(("append", item))
                continue
            elif op.startswith("suvm.dist.batch_signal"):
                batch, peer, _, index, *value = args
                item = ("signal", peer, index, value[0] if value else None)
                batch.append(item)
                events.append(("append", item))
                continue
            elif op.startswith("suvm.dist.batch_submit"):
                events.append(("submit", tuple(args[0])))
                if args[0]:
                    batches.append(tuple(args[0]))
                continue
            elif op.startswith("suvm.dist.flush"):
                events.append(("flush",))
                continue
            elif op.startswith("suvm.dist.signal_wait"):
                output = args[-1]
                events.append(("wait", output))
            elif op.startswith("suvm.sync"):
                continue
            elif op.startswith("cf.assert"):
                assert args[0]
                continue
            else:
                raise AssertionError(f"Unsupported trace operation: {op}")
            if op.startswith("arith.") and not op.startswith("arith.cmpi"):
                result_type = re.search(r": (i\d+|index)\s*$", op)
                if result_type and isinstance(output, int):
                    width = 64 if result_type[1] == "index" else int(result_type[1][1:])
                    output &= (1 << width) - 1
            if lhs:
                name, sep, count = lhs.partition(":")
                if sep:
                    for i, value in enumerate(output):
                        values[f"{name}#{i}"] = value
                    assert len(output) == int(count)
                else:
                    values[name] = output[0] if isinstance(output, tuple) and op.startswith(("scf.if", "scf.for", "scf.while")) else output
        return ()

    execute(program, env)
    return batches, events
