# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Automating python testing, PERSEO common NOX sessions"""

from __future__ import annotations

import glob
import shutil
import sys
from pathlib import Path

import nox

nox.options.error_on_missing_interpreters = True

_LICENSE_HEADER = """# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""


PY_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]
WIN32 = sys.platform == "win32"
PLATFORM = "win" if WIN32 else "linux"


def sdist_builder(session: nox.Session) -> None:
    """Function to build sdist"""
    session.install("build")
    session.run("python", "-m", "build", "--sdist", silent=True)


def wheel_builder(session: nox.Session) -> None:
    """Function to build a wheel"""
    session.install("build")
    session.run("python", "-m", "build", "--wheel", silent=True)


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
    session.install("-e", ".[test]", silent=True)
    # Run pytest with coverage and JUnit XML output
    session.run(
        "python",
        "-m",
        "pytest",
        f"--junitxml=_build/pytest-report-{PLATFORM}-py{session.python}.xml",
        f"--cov-report=xml:_build/pytest-coverage-{PLATFORM}-py{session.python}.xml",
    )


@nox.session()
def fix_format(session: nox.Session):
    """Fix formatting current project folder"""
    session.install("ruff")
    session.run("ruff", "format")
    session.run("ruff", "check", "--select", "I", "--fix-only")


@nox.session()
def check_format(session: nox.Session):
    """Check proper formatting with ruff. Check presence of license header"""
    session.install("ruff")
    session.run("ruff", "format", "--check")
    session.run("ruff", "check")

    def wrong_license_header(file: str) -> bool:
        with open(file, "r", encoding="UTF-8") as input_file:
            header = (
                input_file.readline() + input_file.readline() + input_file.readline()
            )
            return header != _LICENSE_HEADER

    source_files = glob.glob("src/**/*.py", recursive=True)
    no_licensed_files = list(filter(wrong_license_header, source_files))

    if len(no_licensed_files) > 0:
        for file in no_licensed_files:
            session.warn(f"{file} has no license header")
        session.error()


@nox.session()
def pylint(session: nox.Session):
    """Linting with pylint"""
    session.install("pylint")
    session.run("python", "-m", "pylint", "src")


@nox.session(python=PY_VERSIONS)
def pytest(session: nox.Session) -> None:
    """Module testing with pytest"""
    cwd = Path.cwd()
    pytest_executor(session, project=cwd.name)


@nox.session()
def build_sdist(session: nox.Session):
    """Building source distribution"""
    sdist_builder(session)


@nox.session()
def build_wheel(session: nox.Session):
    """Building wheel for distribution"""
    wheel_builder(session)
