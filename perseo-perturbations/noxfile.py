# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Automating python testing, PERSEO Perturbations Package"""

import re
import sys
from pathlib import Path

import nox

sys.path.append("..")
import nox_common  # type: ignore # noqa: F401


@nox.session()
def version_consistency_check(session: nox.Session) -> None:
    """Checking version consistency between __init__.py and meson.build files"""

    # checking that __init__.py __version__ is equal to the meson.build version
    project_version_match = re.search(
        r"__version__\s*=\s*(.*)", Path("src/perseo_perturbations/__init__.py").read_text(encoding="UTF-8")
    )
    if project_version_match is None:
        session.error("Could not find Python Package version inside __init__.py file")
    project_version = project_version_match.group(1).replace('"', "")

    meson_version_match = re.search(r"version:\s*(.*)", Path("meson.build").read_text(encoding="UTF-8"))
    if meson_version_match is None:
        session.error("Could not find Meson version inside meson.build file")
    meson_version = meson_version_match.group(1).replace("'", "").replace(",", "")
    if project_version != meson_version:
        session.error(f"__init__.py ({project_version}) and meson.build ({meson_version}) versions are different!")
    else:
        session.log(f"__init__.py ({project_version}) and meson.build ({meson_version}) versions are the same!")
