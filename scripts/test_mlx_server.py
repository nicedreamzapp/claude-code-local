#!/usr/bin/env python3
"""
Test harness for MLX Native Server — simulates Claude Code's Anthropic API requests.
Tests multi-step tool-use tasks to verify the model can reliably emit tool calls.
"""

import json
import sys
import time
import requests

SERVER = "http://localhost:4000"
PASS = 0
FAIL = 0
RESULTS = []

# ─── Minimal tool definitions (what Claude Code sends) ──────────────────────

TOOLS = [
    {
        "name": "Bash",
        "description": "Execute a bash command and return output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"},
                "description": {"type": "string", "description": "Description of what this command does"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read a file from the filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Edit",
        "description": "Edit a file by replacing text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file"},
                "old_string": {"type": "string", "description": "Text to find"},
                "new_string": {"type": "string", "description": "Replacement text"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "Write",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Grep",
        "description": "Search file contents with regex.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search"},
                "path": {"type": "string", "description": "Path to search in"},
            },
            "required": ["pattern"],
        },
    },
]


def send_message(messages, system="You are a helpful coding assistant. When asked to perform tasks, you MUST use the available tools. Always call tools using <tool_call> JSON format.", max_tokens=4096):
    """Send a request to the MLX server and return the parsed response."""
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
        "tools": TOOLS,
    }
    t0 = time.time()
    resp = requests.post(f"{SERVER}/v1/messages", json=body, timeout=120)
    elapsed = time.time() - t0
    data = resp.json()
    return data, elapsed


def extract_tool_calls(response):
    """Extract tool_use blocks from an Anthropic response."""
    calls = []
    for block in response.get("content", []):
        if block.get("type") == "tool_use":
            calls.append(block)
    return calls


def extract_text(response):
    """Extract text blocks from an Anthropic response."""
    parts = []
    for block in response.get("content", []):
        if block.get("type") == "text" and block.get("text", "").strip():
            parts.append(block["text"].strip())
    return "\n".join(parts)


def run_test(name, messages, expect_tool=None, expect_any_tool=True, system=None):
    """Run a single test case."""
    global PASS, FAIL
    kwargs = {"messages": messages}
    if system:
        kwargs["system"] = system

    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")

    try:
        resp, elapsed = send_message(**kwargs)
    except Exception as e:
        print(f"  FAIL — request error: {e}")
        FAIL += 1
        RESULTS.append((name, "FAIL", str(e)))
        return None

    tool_calls = extract_tool_calls(resp)
    text = extract_text(resp)
    tokens_in = resp.get("usage", {}).get("input_tokens", 0)
    tokens_out = resp.get("usage", {}).get("output_tokens", 0)

    print(f"  Time: {elapsed:.1f}s | Tokens: {tokens_in} in, {tokens_out} out")
    if text:
        print(f"  Text: {text[:150]}...")
    if tool_calls:
        for tc in tool_calls:
            args_preview = json.dumps(tc.get("input", {}))[:120]
            print(f"  Tool: {tc['name']}({args_preview})")
    else:
        print(f"  Tool calls: NONE")

    # Check expectations
    passed = True
    reason = ""

    if expect_any_tool and not tool_calls:
        passed = False
        reason = "Expected a tool call but got none"
    elif expect_tool and not any(tc["name"] == expect_tool for tc in tool_calls):
        passed = False
        reason = f"Expected tool '{expect_tool}' but got: {[tc['name'] for tc in tool_calls]}"
    elif tool_calls:
        # Verify tool calls have non-empty arguments
        for tc in tool_calls:
            inp = tc.get("input", {})
            if not inp:
                passed = False
                reason = f"Tool '{tc['name']}' has empty arguments"
                break

    if passed:
        print(f"  ✓ PASS")
        PASS += 1
        RESULTS.append((name, "PASS", ""))
    else:
        print(f"  ✗ FAIL — {reason}")
        FAIL += 1
        RESULTS.append((name, "FAIL", reason))

    return resp


# ─── Test Cases ─────────────────────────────────────────────────────────────

def test_simple_bash():
    """Test: simple single bash command."""
    return run_test(
        "Simple Bash command",
        [{"role": "user", "content": "List the files in /tmp"}],
        expect_tool="Bash",
    )


def test_mkdir():
    """Test: create a directory."""
    return run_test(
        "Create directory with mkdir",
        [{"role": "user", "content": "Create a folder called /tmp/test_mlx_calendar"}],
        expect_tool="Bash",
    )


def test_multi_step_calendar():
    """Test: the exact calendar scenario that failed — create folders then delete some."""
    # Turn 1: create the structure
    print("\n" + "#"*70)
    print("# MULTI-STEP TEST: Calendar folder creation + deletion")
    print("#"*70)

    resp1 = run_test(
        "Calendar Step 1: Create month folders",
        [{"role": "user", "content": "Create a folder /tmp/test_calendar and inside it create a folder for each month of the year: January through December."}],
        expect_tool="Bash",
    )
    if not resp1:
        return

    # Simulate the tool result coming back
    tool_calls = extract_tool_calls(resp1)
    if not tool_calls:
        return

    tool_result_msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_calls[0]["id"],
                "content": "Done. Directories created."
            }
        ]
    }

    # Turn 2: now delete all months except September
    resp2 = run_test(
        "Calendar Step 2: Delete all months except September",
        [
            {"role": "user", "content": "Create a folder /tmp/test_calendar and inside it create a folder for each month of the year: January through December."},
            {"role": "assistant", "content": resp1.get("content", [])},
            tool_result_msg,
            {"role": "user", "content": "Now delete all of the month folders except for September."},
        ],
        expect_tool="Bash",
    )
    if not resp2:
        return

    tool_calls2 = extract_tool_calls(resp2)
    if not tool_calls2:
        return

    tool_result_msg2 = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_calls2[0]["id"],
                "content": "Done. Only September remains."
            }
        ]
    }

    # Turn 3: verify with ls
    run_test(
        "Calendar Step 3: Verify remaining folders",
        [
            {"role": "user", "content": "Create a folder /tmp/test_calendar and inside it create a folder for each month of the year."},
            {"role": "assistant", "content": resp1.get("content", [])},
            tool_result_msg,
            {"role": "user", "content": "Delete all of the month folders except for September."},
            {"role": "assistant", "content": resp2.get("content", [])},
            tool_result_msg2,
            {"role": "user", "content": "Now list the contents of /tmp/test_calendar to verify only September is left."},
        ],
        expect_tool="Bash",
    )


def test_file_read():
    """Test: read a file."""
    return run_test(
        "Read a file",
        [{"role": "user", "content": "Read the file /etc/hosts"}],
        expect_tool="Read",
    )


def test_multi_tool_sequence():
    """Test: a task requiring multiple different tools in sequence."""
    return run_test(
        "Multi-tool: find then read",
        [{"role": "user", "content": "Find all .py files in /tmp and then read the first one you find. Start by searching for them."}],
        expect_any_tool=True,
    )


def test_edit_file():
    """Test: edit a file (requires structured arguments)."""
    return run_test(
        "Edit a file",
        [
            {"role": "user", "content": "I have a file at /tmp/test_edit.txt with the content 'hello world'. Change 'hello' to 'goodbye'."},
        ],
        expect_any_tool=True,
    )


def test_complex_bash():
    """Test: complex bash command with pipes and logic."""
    return run_test(
        "Complex Bash with pipes",
        [{"role": "user", "content": "Count how many lines are in /etc/hosts and show me the result."}],
        expect_tool="Bash",
    )


def test_back_to_back_tools():
    """Simulate 5 rapid tool calls in sequence to test consistency."""
    print("\n" + "#"*70)
    print("# RAPID-FIRE TEST: 5 sequential Bash commands")
    print("#"*70)

    prompts = [
        "Run: echo 'test 1'",
        "Run: echo 'test 2'",
        "Run: pwd",
        "Run: date",
        "Run: whoami",
    ]
    for i, prompt in enumerate(prompts):
        run_test(
            f"Rapid-fire #{i+1}: {prompt}",
            [{"role": "user", "content": prompt}],
            expect_tool="Bash",
        )


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("MLX Server Test Harness")
    print(f"Server: {SERVER}")
    print(f"Testing tool-call reliability for multi-step tasks\n")

    # Check server health
    try:
        r = requests.get(f"{SERVER}/health", timeout=5)
        print(f"Server health: {r.json()}")
    except Exception as e:
        print(f"Server not reachable: {e}")
        sys.exit(1)

    # Run all tests
    test_simple_bash()
    test_mkdir()
    test_file_read()
    test_complex_bash()
    test_edit_file()
    test_multi_tool_sequence()
    test_back_to_back_tools()
    test_multi_step_calendar()

    # Summary
    print("\n" + "="*70)
    print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
    print("="*70)
    for name, status, reason in RESULTS:
        icon = "✓" if status == "PASS" else "✗"
        suffix = f" — {reason}" if reason else ""
        print(f"  {icon} {name}{suffix}")

    if FAIL > 0:
        print(f"\n{FAIL} test(s) FAILED — server needs more work")
        sys.exit(1)
    else:
        print(f"\nAll {PASS} tests PASSED!")
        sys.exit(0)
