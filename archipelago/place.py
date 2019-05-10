import os
import subprocess
import pythunder


def place(packed_filename: str, layout_filename: str, placement_filename: str,
          fixed: bool = False):
    path = os.path.abspath(os.path.dirname(pythunder.__file__))
    placer_binary = os.path.join(path, "placer")
    assert os.path.isfile(placer_binary), placer_binary + " not found"
    if fixed:
        subprocess.check_call([placer_binary, "-f", layout_filename,
                               packed_filename,
                               placement_filename])
    else:
        subprocess.check_call([placer_binary, layout_filename,
                               packed_filename, placement_filename])
