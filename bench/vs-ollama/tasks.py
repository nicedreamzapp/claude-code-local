"""Ten small agentic coding tasks, each with an automatic pass/fail check.

Every task gets a fresh empty directory. `setup(d)` writes the starting files,
`prompt` is what Claude Code is told, `check(d)` returns (passed, reason).
The tasks were written and frozen before any run; they are not tuned to
either server. Checks run the code with the machine's python3 and only use the
standard library (unittest, not pytest), so nothing outside the task dir is
needed.
"""
import hashlib
import json
import os
import subprocess

PY = "/usr/bin/python3"


def _w(d, rel, text):
    p = os.path.join(d, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


def _r(d, rel):
    try:
        with open(os.path.join(d, rel)) as f:
            return f.read()
    except OSError:
        return None


def _run(d, *args, timeout=30):
    try:
        p = subprocess.run(list(args), cwd=d, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:  # noqa: BLE001
        return -1, "", str(e)


def _sha(text):
    return hashlib.sha256((text or "").encode()).hexdigest()


# ── 1. write a file ─────────────────────────────────────────────────────────
def t1_setup(d):
    pass


def t1_check(d):
    if _r(d, "hello.py") is None:
        return False, "hello.py missing"
    rc, out, err = _run(d, PY, "hello.py")
    return (rc == 0 and out.strip() == "Hello, world!"), f"rc={rc} out={out.strip()[:60]!r}"


# ── 2. edit one value, keep the rest ────────────────────────────────────────
CONFIG = 'HOST = "127.0.0.1"\nPORT = 8080\nTIMEOUT = 30\nRETRIES = 3\n'


def t2_setup(d):
    _w(d, "config.py", CONFIG)


def t2_check(d):
    src = _r(d, "config.py")
    if src is None:
        return False, "config.py missing"
    want = CONFIG.replace("TIMEOUT = 30", "TIMEOUT = 60")
    return src.strip() == want.strip(), "content " + ("ok" if src.strip() == want.strip() else repr(src[:120]))


# ── 3. read a failing test, fix the code (not the test) ─────────────────────
CALC = '''def add(a, b):
    return a - b


def mul(a, b):
    return a * b
'''
TEST_CALC = '''import unittest
from calc import add, mul


class T(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)

    def test_mul(self):
        self.assertEqual(mul(4, 5), 20)


if __name__ == "__main__":
    unittest.main()
'''


def t3_setup(d):
    _w(d, "calc.py", CALC)
    _w(d, "test_calc.py", TEST_CALC)


def t3_check(d):
    if _sha(_r(d, "test_calc.py")) != _sha(TEST_CALC):
        return False, "test file was modified"
    rc, out, err = _run(d, PY, "-m", "unittest", "-q", "test_calc")
    return rc == 0, f"unittest rc={rc} {err.strip()[-80:]!r}"


# ── 4. run a script with Bash, record its output ────────────────────────────
STATS = '''import csv
rows = list(csv.DictReader(open("sales.csv")))
print(sum(int(r["qty"]) * int(r["price"]) for r in rows))
'''
SALES = "item,qty,price\napple,3,7\npear,5,11\nplum,2,13\nfig,9,4\n"  # 21+55+26+36 = 138


def t4_setup(d):
    _w(d, "stats.py", STATS)
    _w(d, "sales.csv", SALES)


def t4_check(d):
    a = _r(d, "answer.txt")
    if a is None:
        return False, "answer.txt missing"
    return a.strip() == "138", f"answer={a.strip()[:40]!r}"


# ── 5. create then run (output file produced by running it) ─────────────────
def t5_setup(d):
    pass


def t5_check(d):
    exp = []
    for i in range(1, 16):
        exp.append("FizzBuzz" if i % 15 == 0 else "Fizz" if i % 3 == 0 else "Buzz" if i % 5 == 0 else str(i))
    if _r(d, "fizzbuzz.py") is None:
        return False, "fizzbuzz.py missing"
    out = _r(d, "out.txt")
    if out is None:
        return False, "out.txt missing"
    got = [x.strip() for x in out.strip().splitlines()]
    return got == exp, "out.txt " + ("ok" if got == exp else repr(got[:6]))


# ── 6. rename a function across two files ───────────────────────────────────
UTILS = '''def get_data():
    return [1, 2, 3]
'''
MAIN = '''from utils import get_data

print(sum(get_data()))
'''


def t6_setup(d):
    _w(d, "utils.py", UTILS)
    _w(d, "main.py", MAIN)


def t6_check(d):
    u, m = _r(d, "utils.py") or "", _r(d, "main.py") or ""
    if "def fetch_data" not in u or "get_data" in u or "get_data" in m:
        return False, "rename incomplete"
    rc, out, err = _run(d, PY, "main.py")
    return (rc == 0 and out.strip() == "6"), f"rc={rc} out={out.strip()!r}"


# ── 7. edit JSON, preserve existing keys ────────────────────────────────────
SETTINGS = {"name": "demo", "version": 2, "features": ["a", "b"]}


def t7_setup(d):
    _w(d, "settings.json", json.dumps(SETTINGS, indent=2) + "\n")


def t7_check(d):
    try:
        got = json.loads(_r(d, "settings.json") or "")
    except ValueError as e:
        return False, f"invalid json: {e}"
    want = dict(SETTINGS, debug=True)
    return got == want, "json " + ("ok" if got == want else repr(got)[:100])


# ── 8. search several files, count, write the number ────────────────────────
SRC_FILES = {
    "src/a.py": "x = 1  # TODO: rename\n# TODO tidy\ny = 2\n",
    "src/b.py": "def f():\n    pass  # TODO implement\n",
    "src/c.py": "print('done')\n",
    "src/d.py": "# TODO one\n# TODO two\n# todo lowercase does not count\n",
}  # 5 lines containing uppercase TODO


def t8_setup(d):
    for k, v in SRC_FILES.items():
        _w(d, k, v)


def t8_check(d):
    a = _r(d, "todo_count.txt")
    if a is None:
        return False, "todo_count.txt missing"
    return a.strip() == "5", f"count={a.strip()[:20]!r}"


# ── 9. fix a syntax error so the script runs ────────────────────────────────
BROKEN = '''def greet(name)
    return "Hi " + name

print(greet("Ada"))
'''


def t9_setup(d):
    _w(d, "broken.py", BROKEN)


def t9_check(d):
    rc, out, err = _run(d, PY, "broken.py")
    return (rc == 0 and out.strip() == "Hi Ada"), f"rc={rc} out={out.strip()!r} {err.strip()[-60:]!r}"


# ── 10. multi-step: package + test + run the test ───────────────────────────
def t10_setup(d):
    pass


def t10_check(d):
    if _r(d, "mathx/__init__.py") is None:
        return False, "mathx/__init__.py missing"
    if _r(d, "test_square.py") is None:
        return False, "test_square.py missing"
    rc, out, err = _run(d, PY, "-c", "from mathx import square; assert square(7) == 49; assert square(-3) == 9")
    if rc != 0:
        return False, f"square wrong: {err.strip()[-80:]!r}"
    rc, out, err = _run(d, PY, "-m", "unittest", "-v", "test_square")
    ran = "Ran 0 tests" not in err and " ok" in err
    return (rc == 0 and ran), f"unittest rc={rc} ran={ran}"


TASKS = [
    {"id": "t01_write_file", "setup": t1_setup, "check": t1_check,
     "prompt": 'Create a file named hello.py in the current directory that prints exactly "Hello, world!" when run with python3.'},
    {"id": "t02_edit_value", "setup": t2_setup, "check": t2_check,
     "prompt": "In config.py, change TIMEOUT from 30 to 60. Do not change anything else in the file."},
    {"id": "t03_fix_failing_test", "setup": t3_setup, "check": t3_check,
     "prompt": "The tests in test_calc.py are failing. Run them with `python3 -m unittest test_calc`, find the bug in calc.py and fix it. Do not modify test_calc.py."},
    {"id": "t04_run_and_record", "setup": t4_setup, "check": t4_check,
     "prompt": "Run `python3 stats.py` and write the number it prints (and nothing else) into a new file named answer.txt."},
    {"id": "t05_create_and_run", "setup": t5_setup, "check": t5_check,
     "prompt": "Write fizzbuzz.py that prints FizzBuzz for the numbers 1 to 15, one per line (Fizz for multiples of 3, Buzz for multiples of 5, FizzBuzz for both, otherwise the number). Then run it with python3 and save its output to out.txt."},
    {"id": "t06_rename_across_files", "setup": t6_setup, "check": t6_check,
     "prompt": "Rename the function get_data to fetch_data in utils.py and update every place that uses it, so that `python3 main.py` still works."},
    {"id": "t07_edit_json", "setup": t7_setup, "check": t7_check,
     "prompt": 'Add a key "debug" with the boolean value true to settings.json. Keep every existing key and value unchanged.'},
    {"id": "t08_search_and_count", "setup": t8_setup, "check": t8_check,
     "prompt": "Count how many lines in the .py files under src/ contain the exact uppercase text TODO. Write just that number into todo_count.txt."},
    {"id": "t09_fix_syntax_error", "setup": t9_setup, "check": t9_check,
     "prompt": "broken.py fails with a syntax error. Fix it so `python3 broken.py` runs and prints its greeting."},
    {"id": "t10_package_and_test", "setup": t10_setup, "check": t10_check,
     "prompt": "Create a Python package mathx (a mathx/ folder with __init__.py) that defines square(x) returning x*x. Then write test_square.py with a unittest test for it, and run it with `python3 -m unittest test_square` to confirm it passes."},
]
