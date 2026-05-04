from pathlib import Path


def write_sw4_input(path, grid_line, tmax, fileio_path, supergrid_gp,
                    material_lines, source_lines, receiver_lines,
                    topography_line=None):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# SW4 input generated from a ShakerMaker model\n")
        f.write("# Source time functions are read from sources/\n\n")
        f.write(grid_line + "\n")
        f.write(f"time t={float(tmax):.16g}\n")
        f.write(f"fileio path={fileio_path}\n\n")
        f.write(f"supergrid gp={int(supergrid_gp)}\n")
        if topography_line:
            f.write(topography_line + "\n")
        f.write("\n# Material model\n")
        f.write("\n".join(material_lines) + "\n")
        f.write("\n# Sources\n")
        f.write("\n".join(source_lines) + "\n")
        f.write("\n# Receivers\n")
        f.write("\n".join(receiver_lines) + "\n")
