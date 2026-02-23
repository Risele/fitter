import unittest

from codegen.c_float import generate_c_float
from codegen.formatting import emit_wrapped_array_items
from core.types import FitResult


class CodegenFormattingTests(unittest.TestCase):
    def test_emit_wrapped_array_items_respects_line_length(self):
        lines = emit_wrapped_array_items(["1.0f", "2.0f", "3.0f", "4.0f"], indent="  ", max_line_len=14)
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLessEqual(len(line), 14)

    def test_generate_c_float_wraps_arrays(self):
        res = FitResult(
            name="t",
            xmin=0.0,
            xmax=1.0,
            n=6,
            dx=0.2,
            inv_dx=5.0,
            y=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            k=None,
            max_abs_err=0.0,
            mean_signed_err=0.0,
            verify_points=0,
        )
        code = generate_c_float(
            res=res,
            func_name="f",
            array_name="tbl",
            store_slopes=False,
            test_count=0,
            precision="float",
            max_line_len=30,
        )
        self.assertIn("static const float tbl_y[6]", code)
        self.assertIn("\n  0.0f,", code)


if __name__ == "__main__":
    unittest.main()
