# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Automating python testing, PERSEO Core Package"""

import sys
from pathlib import Path

import nox  # noqa: F401

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
    session.install("-e", ".[test]", silent=True)
    # Run pytest with coverage and JUnit XML output
    session.run(
        "python",
        "-m",
        "pytest",
        f"--junitxml=_build/pytest-report-{nox_common.PLATFORM}-py{session.python}.xml",
        f"--cov-report=xml:_build/pytest-coverage-{nox_common.PLATFORM}-py{session.python}.xml",
    )


@nox.session(python=nox_common.PY_VERSIONS)
def pytest(session: nox.Session):
    """Module testing with pytest"""
    cwd = Path.cwd()
    pytest_executor(session, project=cwd.name)
