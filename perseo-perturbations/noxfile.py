# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Automating python testing, PERSEO Perturbations Package"""

import re
import sys
from pathlib import Path

import nox

sys.path.append("..")
import nox_common  # type: ignore # noqa: F401


def pytest_executor(session: nox.Session, project: str) -> None:
    """Executor of pytest from nox session.

    Parameters
    ----------
    session : nox.Session
        nox session
    project : str
        project name, with "-" as separator for namespace sub-packages
    """
    Path("_build").mkdir(exist_ok=True)

    project = project.replace("-", "_")

    session.install("-e", ".[test,graphs]", silent=True)

    session.run(
        "python",
        "-m",
        "pytest",
        "--cov",
        project,
        "--cov-report",
        "term-missing",
        "--cov-report",
        f"xml:_build/pytest-coverage-{nox_common.PLATFORM}-py{session.python}.xml",
        "--junitxml",
        f"_build/pytest-report-{nox_common.PLATFORM}-py{session.python}.xml",
    )


@nox.session(python=nox_common.PY_VERSIONS)
def unittest(session: nox.Session):
    """Module testing with pytest"""
    cwd = Path.cwd()
    pytest_executor(session, project=cwd.name)


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


@nox.session(python=nox_common.PY_VERSIONS)
def build_python_package(session: nox.Session):
    """Building python wheel package for distribution"""
    session.install("build")
    session.install("numpy>2")
    session.run("python", "-m", "build", "--wheel", silent=False)

    if nox_common.PLATFORM == "linux":
        auditwheel_plat = session.env.get("AUDITWHEEL_PLAT")
        if auditwheel_plat is not None:
            session.install("auditwheel")
            for wheel in Path("dist").glob("*.whl"):
                session.run(
                    "auditwheel",
                    "repair",
                    "-w",
                    "dist",
                    "--plat",
                    auditwheel_plat,
                    str(wheel),
                )
                wheel.unlink()
