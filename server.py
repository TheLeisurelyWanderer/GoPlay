"""
KataGo WebSocket Bridge Server

Bridges a browser-based Go game (WebSocket JSON) to KataGo (GTP stdin/stdout).
Supports dual engines for AI vs AI with different KataGo versions/models.
Supports game analysis via kata-analyze.

Configuration is via config.json (in same directory as server.py):

    {
      "server": { "host": "localhost", "port": 9001 },
      "engine1": {
        "name": "KataGo v1.16.0 OpenCL",
        "command": "path/to/katago.exe gtp -model path/to/model.bin.gz -config path/to/config.cfg"
      },
      "engine2": {
        "name": "KataGo v1.13.1 TRT",
        "command": "path/to/katago2.exe gtp -model path/to/model2.bin.gz -config path/to/config2.cfg"
      }
    }

Usage:
    python server.py                                    # Use config.json
    python server.py --command "katago gtp -model ..."  # Override engine 1
    python server.py --port 9002                        # Override port
"""

import argparse
import asyncio
import json
import logging

import re
import sys
from pathlib import Path

import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("katago-bridge")

# ── Coordinate conversion ──────────────────────────────────────────────

GTP_COLS = "ABCDEFGHJKLMNOPQRST"  # Standard Go: skip 'I'


def board_to_gtp(row: int, col: int, board_size: int) -> str:
    """Convert 0-indexed (row, col) where row 0 = top to GTP vertex like 'D16'."""
    letter = GTP_COLS[col]
    number = board_size - row
    return f"{letter}{number}"


def gtp_to_board(vertex: str, board_size: int):
    """Convert GTP vertex like 'D16' to (row, col). Returns 'pass' or 'resign' as strings."""
    vertex = vertex.strip().upper()
    if vertex == "PASS":
        return "pass"
    if vertex == "RESIGN":
        return "resign"
    letter = vertex[0]
    number = int(vertex[1:])
    col = GTP_COLS.index(letter)
    row = board_size - number
    return (row, col)


# ── KataGo GTP Engine ──────────────────────────────────────────────────


class KataGoError(Exception):
    pass


class KataGoEngine:
    """Manages a KataGo subprocess communicating via GTP protocol."""

    def __init__(self, command: str, name: str = "primary", display_name: str = ""):
        self.command = command  # full command string, e.g. "katago.exe gtp -model ... -config ..."
        self.name = name
        self.display_name = display_name  # from config "name" field
        self.process = None
        self._lock = asyncio.Lock()
        self.version_info = ""

    async def start(self):
        """Spawn the KataGo process and wait for it to be ready."""
        log.info(f"[{self.name}] Starting: {self.command}")

        self.process = await asyncio.create_subprocess_shell(
            self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1024 * 1024,  # 1MB buffer for kata-analyze output
        )

        # Drain stderr in background (KataGo logs extensively to stderr)
        asyncio.create_task(self._drain_stderr())

        # Wait for KataGo to be ready
        kg_name = await self.send_command("name")
        version = await self.send_command("version")
        # Extract base version (strip abbreviated model after '+')
        base_version = version.split("+")[0] if "+" in version else version
        self.version_info = f"{kg_name} v{base_version}"
        log.info(f"[{self.name}] KataGo ready: {self.version_info}")

    async def send_command(self, command: str) -> str:
        """
        Send a GTP command and return the response body.

        GTP responses:
          = <body>\\n\\n   (success)
          ? <body>\\n\\n   (error)
        """
        if self.process is None or self.process.returncode is not None:
            raise KataGoError(f"[{self.name}] KataGo process is not running")

        async with self._lock:
            log.info(f"[{self.name}] >> {command}")
            self.process.stdin.write((command + "\n").encode())
            await self.process.stdin.drain()

            # Read response lines until we get the terminating empty line
            response_lines = []
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    raise KataGoError(f"[{self.name}] KataGo process closed stdout unexpectedly")
                decoded = line.decode().strip()
                if decoded == "" and not response_lines:
                    continue
                if decoded == "" and response_lines:
                    break
                response_lines.append(decoded)

            full = "\n".join(response_lines)
            log.info(f"[{self.name}] << {full[:200]}")

            if full.startswith("= "):
                return full[2:].strip()
            elif full.startswith("="):
                return full[1:].strip()
            elif full.startswith("? "):
                raise KataGoError(full[2:].strip())
            elif full.startswith("?"):
                raise KataGoError(full[1:].strip())
            return full

    async def run_analysis(self, color: str, max_visits: int = 200, analysis_duration: int = 3) -> str:
        """
        Run kata-analyze and collect the final analysis output.

        GTP kata-analyze syntax:
            kata-analyze [player] [interval] KEYVALUEPAIR ...
        Each output line is a complete batch containing multiple 'info move ...'
        records concatenated on a single line. Lines are NOT separated by
        blank lines — each line replaces the previous one with updated stats.

        We read lines until enough time has elapsed, then interrupt.
        The LAST info line is the most accurate result.
        """
        if self.process is None or self.process.returncode is not None:
            raise KataGoError(f"[{self.name}] KataGo process is not running")

        async with self._lock:
            # kata-analyze [player] interval CENTISECONDS maxmoves N
            cmd = f"kata-analyze {color} interval 10 maxmoves 15 ownership true"
            log.info(f"[{self.name}] >> {cmd}")
            self.process.stdin.write((cmd + "\n").encode())
            await self.process.stdin.drain()

            last_line = ""
            got_first_info = False
            update_count = 0

            first_data_timeout = 120  # seconds waiting for first info line
            # analysis_duration passed as parameter (default 3s)

            import time
            start_time = time.monotonic()

            try:
                while True:
                    timeout = first_data_timeout if not got_first_info else 5
                    try:
                        raw_line = await asyncio.wait_for(
                            self.process.stdout.readline(),
                            timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        if got_first_info:
                            log.info(f"[{self.name}] readline timeout, stopping")
                        else:
                            log.warning(
                                f"[{self.name}] kata-analyze: no info line"
                                f" after {first_data_timeout}s"
                            )
                        break

                    if not raw_line:
                        raise KataGoError(
                            f"[{self.name}] KataGo closed stdout"
                        )

                    line = raw_line.decode(errors="replace").rstrip("\r\n")

                    # Skip empty lines
                    if line == "":
                        continue

                    # Check for GTP error response
                    if line.startswith("?"):
                        log.error(
                            f"[{self.name}] kata-analyze error: {line[:200]}"
                        )
                        raise KataGoError(line)

                    # Check for '=' terminator
                    if line.startswith("="):
                        if got_first_info:
                            log.info(
                                f"[{self.name}] got '=' after"
                                f" {update_count} updates,"
                                f" {len(last_line)} chars"
                            )
                            return last_line
                        else:
                            # Stale '=' from a previous command — skip
                            log.info(
                                f"[{self.name}] skipping stale '=':"
                                f" {line[:50]}"
                            )
                            continue

                    # Collect info lines — each line is a complete batch
                    if line.startswith("info"):
                        got_first_info = True
                        last_line = line
                        update_count += 1

                        # Check if we've run long enough
                        elapsed = time.monotonic() - start_time
                        if elapsed >= analysis_duration:
                            log.info(
                                f"[{self.name}] {elapsed:.1f}s elapsed,"
                                f" {update_count} updates,"
                                f" {len(last_line)} chars"
                            )
                            break
                    else:
                        log.info(
                            f"[{self.name}] skipping non-info line:"
                            f" {line[:100]}"
                        )

            except KataGoError:
                raise
            except Exception as e:
                log.error(f"[{self.name}] analysis read error: {e}")

            # Phase 2: Interrupt kata-analyze by sending a GTP command.
            # Sending any valid command stops the running analysis.
            # We use 'name' because it produces a known response '= KataGo'.
            log.info(f"[{self.name}] interrupting kata-analyze")
            self.process.stdin.write(b"name\n")
            await self.process.stdin.drain()

            # Phase 3: Drain trailing info lines until we get the '='
            # response from the 'name' command.
            try:
                while True:
                    try:
                        drain_line = await asyncio.wait_for(
                            self.process.stdout.readline(),
                            timeout=2
                        )
                    except asyncio.TimeoutError:
                        log.warning(f"[{self.name}] drain timeout")
                        break
                    if not drain_line:
                        break
                    decoded = drain_line.decode(errors="replace").strip()
                    if decoded.startswith("="):
                        log.info(f"[{self.name}] drained '=' ok")
                        break
                    # Consume trailing info lines or blank lines
            except Exception as e:
                log.error(f"[{self.name}] drain error: {e}")

            log.info(
                f"[{self.name}] analysis done"
                f" ({len(last_line)} chars,"
                f" {update_count} updates)"
            )
            return last_line

    async def _drain_stderr(self):
        """Read stderr to prevent buffer blocking. Log startup messages."""
        while True:
            line = await self.process.stderr.readline()
            if not line:
                break
            text = line.decode().rstrip()
            if text:
                log.info(f"[{self.name}] {text}")

    async def stop(self):
        """Gracefully shut down KataGo."""
        if self.process and self.process.returncode is None:
            try:
                await asyncio.wait_for(self.send_command("quit"), timeout=5)
            except Exception:
                pass
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except Exception:
                self.process.kill()
            log.info(f"[{self.name}] KataGo stopped")


# ── Analysis parsing ──────────────────────────────────────────────────


def parse_kata_analyze(response: str, board_size: int):
    """Parse kata-analyze output into structured data.

    kata-analyze outputs all candidate moves on a SINGLE line:
    info move D4 visits 100 winrate 0.55 ... pv D4 Q16 info move Q16 visits 80 ...

    Each 'info' keyword starts a new candidate. We split on 'info' boundaries.
    """
    moves = []

    # Split on 'info ' boundaries to get individual move records.
    # The response may have multiple lines OR all on one line.
    # Use regex to split: each record starts with 'info move' or 'info pass'.
    records = re.split(r'(?=\binfo\b)', response)
    for record in records:
        record = record.strip()
        if not record.startswith("info"):
            continue
        # Parse key-value pairs
        tokens = record.split()
        data = {}
        i = 1  # skip "info"
        while i < len(tokens):
            key = tokens[i]
            if key == "pv":
                # PV runs until the next known keyword or end of record
                pv_moves = []
                for t in tokens[i + 1:]:
                    if t in ("whiteOwnership", "ownership"):
                        break
                    pv_moves.append(t)
                data["pv"] = pv_moves
                break
            elif i + 1 < len(tokens):
                data[key] = tokens[i + 1]
                i += 2
            else:
                i += 1

        if "move" in data:
            move_info = {"move": data["move"]}
            vertex = data["move"]
            result = gtp_to_board(vertex, board_size)
            if isinstance(result, tuple):
                move_info["row"], move_info["col"] = result
            elif result == "pass":
                move_info["pass"] = True

            if "visits" in data:
                move_info["visits"] = int(data["visits"])
            if "winrate" in data:
                move_info["winrate"] = float(data["winrate"])
            if "scoreLead" in data:
                move_info["scoreLead"] = float(data["scoreLead"])
            if "scoreSelfplay" in data:
                move_info["scoreSelfplay"] = float(data["scoreSelfplay"])
            if "prior" in data:
                move_info["prior"] = float(data["prior"])
            if "utility" in data:
                move_info["utility"] = float(data["utility"])
            if "order" in data:
                move_info["order"] = int(data["order"])
            if "pv" in data:
                # Return PV as both GTP notation and board coords (up to 10 moves)
                pv_gtp = data["pv"][:10]
                pv_moves = []
                for v in pv_gtp:
                    r = gtp_to_board(v, board_size)
                    if isinstance(r, tuple):
                        pv_moves.append({"row": r[0], "col": r[1], "gtp": v})
                    elif r == "pass":
                        pv_moves.append({"pass": True, "gtp": "pass"})
                move_info["pv"] = pv_moves
                move_info["pvText"] = " ".join(pv_gtp)

            moves.append(move_info)

    # Sort by order
    moves.sort(key=lambda m: m.get("order", 999))
    return moves


def parse_ownership(response: str, board_size: int):
    """Extract ownership array from kata-analyze response.

    When 'ownership true' is passed to kata-analyze, KataGo appends:
        ownership <float> <float> ... <float>
    with board_size*board_size floats in row-major order (top-left to
    bottom-right).  Values range from -1 (black) to +1 (white).
    """
    idx = response.rfind(" ownership ")
    if idx < 0:
        return None
    ownership_str = response[idx + len(" ownership "):]
    tokens = ownership_str.split()
    expected = board_size * board_size
    if len(tokens) < expected:
        log.warning(
            f"ownership: expected {expected} floats, got {len(tokens)}"
        )
        return None
    try:
        ownership = []
        for r in range(board_size):
            row = []
            for c in range(board_size):
                row.append(round(float(tokens[r * board_size + c]), 4))
            ownership.append(row)
        return ownership
    except (ValueError, IndexError) as e:
        log.warning(f"ownership parse error: {e}")
        return None


# ── WebSocket Handler ──────────────────────────────────────────────────


async def setup_engine_for_game(engine, board_size, komi, msg):
    """Send board setup commands to a KataGo engine."""
    await engine.send_command(f"boardsize {board_size}")
    await engine.send_command("clear_board")
    await engine.send_command(f"komi {komi}")
    try:
        await engine.send_command("kata-set-rules chinese")
    except KataGoError:
        pass
    main_time = msg.get("main_time", 0)
    byo_time = msg.get("byo_time", 5)
    byo_stones = msg.get("byo_stones", 1)
    try:
        await engine.send_command(
            f"time_settings {main_time} {byo_time} {byo_stones}"
        )
    except KataGoError:
        pass


async def ensure_engine_started(eng: KataGoEngine, websocket=None):
    """Start a KataGo engine if not already running. Returns the engine."""
    if eng.process is None or eng.process.returncode is not None:
        log.info(f"[{eng.name}] Lazy-starting engine on first use...")
        await eng.start()
        # Notify client of newly available engine info
        if websocket:
            info_msg = {"type": "engine_started", "name": eng.name, "version": eng.display_name or eng.version_info}
            await websocket.send(json.dumps(info_msg))
    return eng


async def handle_client(websocket, engines: dict):
    """Handle one browser WebSocket connection."""
    board_size = 19
    default_key = next(iter(engines))  # first engine key
    active_engine = engines[default_key]
    # For AI vs AI: black_engine and white_engine
    black_engine = None
    white_engine = None
    use_dual = False
    log.info(f"Client connected: {websocket.remote_address}")

    def _engine_label(eng):
        """Return configured display name, or version info if started, or command name."""
        if eng.display_name:
            return eng.display_name
        if eng.process is not None and eng.process.returncode is None:
            return eng.version_info
        return eng.command.split()[0]

    # Report all available engines
    engine_info = {"type": "ready", "engines": {k: _engine_label(v) for k, v in engines.items()}}
    await websocket.send(json.dumps(engine_info))

    try:
        async for raw_message in websocket:
            try:
                msg = json.loads(raw_message)
                msg_type = msg.get("type")
                log.info(f"Received: {msg_type} {msg}")

                if msg_type == "new_game":
                    board_size = msg.get("size", 19)
                    komi = msg.get("komi", 6.5)
                    # Engine selection: use_engine key (e.g. "engine1") for HvAI
                    selected_key = msg.get("use_engine", default_key)
                    active_engine = engines.get(selected_key, engines[default_key])
                    await ensure_engine_started(active_engine, websocket)
                    await setup_engine_for_game(active_engine, board_size, komi, msg)

                    # AI vs AI: black_engine and white_engine keys
                    bk = msg.get("black_engine")
                    wk = msg.get("white_engine")
                    use_dual = bool(bk and wk and bk != wk)
                    if use_dual:
                        black_engine = engines.get(bk, engines[default_key])
                        white_engine = engines.get(wk, engines[default_key])
                        for eng in (black_engine, white_engine):
                            await ensure_engine_started(eng, websocket)
                            if eng is not active_engine:
                                await setup_engine_for_game(eng, board_size, komi, msg)

                    await websocket.send(json.dumps({
                        "type": "new_game_ack",
                        "dual_engine": use_dual,
                    }))

                elif msg_type == "play":
                    color = msg["color"]  # "B" or "W"
                    row, col = msg["row"], msg["col"]
                    vertex = board_to_gtp(row, col, board_size)
                    if use_dual and black_engine and white_engine:
                        for eng in {black_engine, white_engine}:
                            await eng.send_command(f"play {color} {vertex}")
                    else:
                        await active_engine.send_command(f"play {color} {vertex}")

                elif msg_type == "genmove":
                    color = msg["color"]  # "B" or "W"
                    if use_dual and black_engine and white_engine:
                        gen_engine = black_engine if color == "B" else white_engine
                        other_engine = white_engine if color == "B" else black_engine
                    else:
                        gen_engine = active_engine
                        other_engine = None

                    response = await gen_engine.send_command(f"genmove {color}")
                    result = gtp_to_board(response, board_size)

                    # Sync the move to the other engine
                    if other_engine and result != "resign":
                        if result == "pass":
                            await other_engine.send_command(f"play {color} pass")
                        else:
                            r, c = result
                            v = board_to_gtp(r, c, board_size)
                            await other_engine.send_command(f"play {color} {v}")

                    if result == "pass":
                        await websocket.send(json.dumps({
                            "type": "pass", "color": color
                        }))
                    elif result == "resign":
                        await websocket.send(json.dumps({
                            "type": "resign", "color": color
                        }))
                    else:
                        row, col = result
                        await websocket.send(json.dumps({
                            "type": "move", "color": color,
                            "row": row, "col": col
                        }))

                elif msg_type == "play_pass":
                    color = msg["color"]
                    if use_dual and black_engine and white_engine:
                        for eng in {black_engine, white_engine}:
                            await eng.send_command(f"play {color} pass")
                    else:
                        await active_engine.send_command(f"play {color} pass")

                elif msg_type == "undo":
                    if use_dual and black_engine and white_engine:
                        for eng in {black_engine, white_engine}:
                            await eng.send_command("undo")
                    else:
                        await active_engine.send_command("undo")
                    await websocket.send(json.dumps({"type": "undo_ack"}))

                elif msg_type == "set_difficulty":
                    visits = msg.get("visits", 200)
                    target = msg.get("target", "all")  # engine key or "all"
                    for key, eng in engines.items():
                        if eng.process and eng.process.returncode is None:
                            if target == "all" or target == key:
                                try:
                                    await eng.send_command(
                                        f"kata-set-param maxVisits {visits}"
                                    )
                                except KataGoError:
                                    log.warning(f"{key}: kata-set-param maxVisits not supported")
                    await websocket.send(json.dumps({"type": "difficulty_ack"}))

                elif msg_type == "analyze":
                    # Analysis: replay moves on engine, then run kata-analyze
                    color = msg.get("color", "B")  # Whose perspective
                    moves = msg.get("moves", [])  # Move list to replay
                    analyze_visits = msg.get("visits", 100)
                    analyze_duration = msg.get("duration", 3)

                    # Select which engine to use for analysis
                    analyze_key = msg.get("use_engine", default_key)
                    analyze_engine = engines.get(analyze_key, engines[default_key])
                    await ensure_engine_started(analyze_engine, websocket)
                    log.info(f"Analysis using {analyze_key}")

                    # Setup fresh board
                    a_size = msg.get("size", board_size)
                    a_komi = msg.get("komi", 6.5)
                    await analyze_engine.send_command(f"boardsize {a_size}")
                    await analyze_engine.send_command("clear_board")
                    await analyze_engine.send_command(f"komi {a_komi}")
                    try:
                        await analyze_engine.send_command("kata-set-rules chinese")
                    except KataGoError:
                        pass

                    # Replay moves up to the analysis point
                    for m in moves:
                        mc = m["color"]
                        if m.get("pass"):
                            await analyze_engine.send_command(f"play {mc} pass")
                        else:
                            v = board_to_gtp(m["row"], m["col"], a_size)
                            await analyze_engine.send_command(f"play {mc} {v}")

                    # Run analysis
                    try:
                        response = await analyze_engine.run_analysis(
                            color, max_visits=analyze_visits,
                            analysis_duration=analyze_duration
                        )
                        analysis = parse_kata_analyze(response, a_size)
                        ownership = parse_ownership(response, a_size)
                        if not analysis:
                            log.warning(f"NO candidates parsed! response[:500]: {response[:500]}")
                        # kata-analyze with default SIDETOMOVE perspective
                        # returns values from the current player's POV.
                        # Normalize winrate/score to Black's perspective,
                        # and ownership to absolute (neg=black, pos=white).
                        if color == "W":
                            for m in analysis:
                                if "winrate" in m:
                                    m["winrate"] = 1.0 - m["winrate"]
                                if "scoreLead" in m:
                                    m["scoreLead"] = -m["scoreLead"]
                                if "scoreSelfplay" in m:
                                    m["scoreSelfplay"] = -m["scoreSelfplay"]
                                if "utility" in m:
                                    m["utility"] = -m["utility"]
                            # Ownership when White to move: pos=White,
                            # neg=Black — already absolute convention.
                        if color == "B" and ownership:
                            # Ownership when Black to move: pos=Black,
                            # neg=White — negate to get absolute convention
                            # (neg=black, pos=white).
                            ownership = [
                                [-v for v in row] for row in ownership
                            ]
                        msg_out = {
                            "type": "analysis",
                            "moves": analysis,
                            "moveNumber": len(moves),
                        }
                        if ownership:
                            msg_out["ownership"] = ownership
                        await websocket.send(json.dumps(msg_out))
                    except KataGoError as e:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"Analysis error: {e}"
                        }))

                else:
                    log.warning(f"Unknown message type: {msg_type}")

            except KataGoError as e:
                log.error(f"KataGo error: {e}")
                await websocket.send(json.dumps({
                    "type": "error", "message": str(e)
                }))
            except json.JSONDecodeError:
                log.error(f"Invalid JSON: {raw_message}")
            except Exception as e:
                log.error(f"Handler error: {e}", exc_info=True)
                await websocket.send(json.dumps({
                    "type": "error", "message": str(e)
                }))

    except websockets.ConnectionClosed:
        log.info(f"Client disconnected: {websocket.remote_address}")


# ── Config File Loading ────────────────────────────────────────────────


def _load_config_file():
    """Load settings from config.json if it exists in the same directory as server.py."""
    config_paths = [
        Path(__file__).parent / "config.json",
        Path.cwd() / "config.json",
    ]
    for p in config_paths:
        if p.exists():
            try:
                with open(p, "r") as f:
                    cfg = json.load(f)
                log.info(f"Loaded config from: {p}")
                return cfg
            except (json.JSONDecodeError, OSError) as e:
                log.warning(f"Failed to load {p}: {e}")
    return {}


_CONFIG = _load_config_file()


# ── Main ───────────────────────────────────────────────────────────────


def parse_args():
    cfg_server = _CONFIG.get("server", {})
    cfg_defaults = _CONFIG.get("game_defaults", {})

    # Collect all engine* entries from config.json
    engines_cfg = {k: v for k, v in sorted(_CONFIG.items())
                   if k.startswith("engine") and isinstance(v, dict)}

    parser = argparse.ArgumentParser(
        description="KataGo WebSocket Bridge (supports multiple engines)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Configure engines in config.json (see module docstring for format).\n"
            "Each engine entry needs 'name' and 'command' fields.\n"
        ),
    )
    parser.add_argument(
        "--port", type=int,
        default=cfg_server.get("port", 9001),
        help="WebSocket port",
    )
    parser.add_argument(
        "--host",
        default=cfg_server.get("host", "localhost"),
        help="Bind host",
    )

    args = parser.parse_args()
    args.engines_cfg = engines_cfg
    args.game_defaults = cfg_defaults
    return args


async def main():
    args = parse_args()

    if not args.engines_cfg:
        log.error("No engines configured. Add engine entries to config.json.")
        sys.exit(1)

    # Create all engines (lazy-start on first use)
    engines = {}
    for key, cfg in args.engines_cfg.items():
        command = cfg.get("command")
        if not command:
            log.warning(f"{key}: no command, skipping")
            continue
        display_name = cfg.get("name", key)
        engines[key] = KataGoEngine(command, name=key, display_name=display_name)
        log.info(f"{key}: {display_name} — {command}")

    if not engines:
        log.error("No valid engines configured.")
        sys.exit(1)

    log.info(f"Configured {len(engines)} engine(s)")
    log.info(f"WebSocket server starting on ws://{args.host}:{args.port}")

    stop_event = asyncio.Event()

    async with websockets.serve(
        lambda ws: handle_client(ws, engines),
        args.host,
        args.port,
    ):
        log.info("Server ready. Waiting for browser connections...")
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass

    for eng in engines.values():
        await eng.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutting down...")
