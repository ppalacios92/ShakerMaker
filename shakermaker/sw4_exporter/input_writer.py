from pathlib import Path


def write_sw4_input(path, grid_line, tmax, fileio_path, supergrid_gp,
                    material_lines, source_lines, receiver_lines,
                    topography_line=None):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(sw4_input_text(
            grid_line, tmax, fileio_path, supergrid_gp,
            material_lines, source_lines, receiver_lines, topography_line))


def sw4_input_text(grid_line, tmax, fileio_path, supergrid_gp,
                   material_lines, source_lines, receiver_lines,
                   topography_line=None):
    lines = [
        "# SW4 input generated from a ShakerMaker model",
        "# Source time functions are read from sources/",
        "",
        grid_line,
        f"time t={float(tmax):.16g}",
        f"fileio path={fileio_path}",
        "",
        f"supergrid gp={int(supergrid_gp)}",
    ]
    if topography_line:
        lines.append(topography_line)
    lines += [
        "",
        "# Material model",
        *material_lines,
        "",
        "# Sources",
        *source_lines,
        "",
        "# Receivers",
    ]
    if _is_receiver_blocks(receiver_lines):
        for title, block_lines in receiver_lines:
            lines += ["", f"# {title}", *block_lines]
    else:
        lines += receiver_lines
    return "\n".join(lines) + "\n"


def _is_receiver_blocks(receiver_lines):
    if not receiver_lines:
        return False
    first = receiver_lines[0]
    return isinstance(first, tuple) and len(first) == 2
