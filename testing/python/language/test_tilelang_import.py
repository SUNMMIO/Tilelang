import subprocess
import sys


def test_tilelang_package_imports_in_clean_interpreter():
    result = subprocess.run(
        [sys.executable, "-c", "import tilelang; import tilelang.language"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
