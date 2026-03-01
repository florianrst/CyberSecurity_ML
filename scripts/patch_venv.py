"""
Post-install patches for third-party library bugs.
Run once after: pip install -r requirements.txt
"""

import importlib
import inspect
import pathlib
import sys


def patch_prince_famd():
    """
    Fix prince.FAMD.fit: sparse arithmetic changes the fill_value to nan,
    which floods the dense matrix with NaN when to_dense() is called.
    row_coordinates() already uses .fillna(0) — fit() was missing it.
    """
    try:
        import prince.famd
    except ImportError:
        print("[SKIP] prince not installed — skipping FAMD patch")
        return

    path = pathlib.Path(inspect.getfile(prince.famd))
    src = path.read_text()

    buggy = "X_cat_oh_norm = X_cat_oh_norm.sparse.to_dense()"
    fixed = "X_cat_oh_norm = X_cat_oh_norm.sparse.to_dense().fillna(0)"

    if fixed in src:
        print("[OK]   prince FAMD patch already applied")
        return

    if buggy not in src:
        print("[SKIP] prince FAMD source changed — patch may not be needed anymore, check manually")
        return

    path.write_text(src.replace(buggy, fixed))
    print(f"[PATCHED] prince FAMD — {path}")


if __name__ == "__main__":
    patch_prince_famd()
    print("Done.")
