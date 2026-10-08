#!/usr/bin/env python3
"""Compile the live distribution logger in isolation; never launch a worker."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainingDistributionLoggingTests(unittest.TestCase):
    def test_live_format_output_and_reset(self):
        source = (ROOT / "LSTM/LSTM.cpp").read_text()
        start = source.index("void PrintAndResetDistribution()\n{")
        end = source.index("\n}\n", start) + 3
        function = source[start:end]
        fixture = r'''
#include <cassert>
#include <cstddef>
#include <cstdio>
using std::size_t;
constexpr size_t direction_output_size = 3;
size_t epoch_total = 4, epoch_correct = 3;
size_t epoch_actual[3] = {1, 2, 1}, epoch_pred[3] = {2, 1, 1};
size_t epoch_conf[3][3] = {{1, 0, 0}, {1, 1, 0}, {0, 0, 1}};
''' + function + r'''
int main() {
    PrintAndResetDistribution();
    assert(epoch_total == 0 && epoch_correct == 0);
    for (size_t i = 0; i < 3; ++i) {
        assert(epoch_actual[i] == 0 && epoch_pred[i] == 0);
        for (size_t j = 0; j < 3; ++j) assert(epoch_conf[i][j] == 0);
    }
    PrintAndResetDistribution(); // Empty epochs must emit nothing.
}
'''
        with tempfile.TemporaryDirectory(prefix="ea_distribution_logging_") as temp:
            cpp, binary = Path(temp) / "logger.cpp", Path(temp) / "logger"
            cpp.write_text(fixture)
            argv = [os.environ.get("CXX", "/usr/bin/clang++"), "-std=c++20",
                    "-Wall", "-Wextra", "-Werror", "-Wformat=2",
                    str(cpp), "-o", str(binary)]
            subprocess.run(argv, check=True, capture_output=True, text=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout, "\n".join([
                "EPOCH_3CLASS_ACTUAL_DISTRIBUTION total=4 down=0.2500 neutral=0.5000 up=0.2500",
                "EPOCH_3CLASS_PRED_DISTRIBUTION total=4 down=0.5000 neutral=0.2500 up=0.2500",
                "EPOCH_3CLASS_CONFUSION_MATRIX rows=actual cols=predicted",
                "row0: 1 0 0", "row1: 1 1 0", "row2: 0 0 1",
                "EPOCH_3CLASS_ACCURACY correct=3 total=4 acc=0.7500", ""]))
            # Prove the compiler oracle catches the confirmed original defect.
            self.assertIn('"row%zu:', fixture)
            cpp.write_text(fixture.replace('"row%zu:', '"row%d:', 1))
            negative = subprocess.run(argv, capture_output=True, text=True)
            self.assertNotEqual(negative.returncode, 0)
            self.assertIn("format specifies type", negative.stderr)


if __name__ == "__main__":
    unittest.main()
