# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Automating python testing, PERSEO Quality Package"""

import sys
from pathlib import Path

import nox  # noqa: F401

sys.path.append("..")
import nox_common  # type: ignore # noqa: F401


def unittest_executor(session: nox.Session, project: str) -> None:
    """Executor of unittest from nox session.

    Parameters
    ----------
    session : nox.Session
        nox session
    project_source : str
        project name, with "-" as separator for namespace sub-packages
    """
    Path("_build").mkdir(exist_ok=True)

    project = project.replace("-", "_")

    session.install("-e", ".[test,graphs]", silent=True)

    session.run(
        "python",
        "-m",
        "coverage",
        "run",
        f"--source={project}",
        "-m",
        "xmlrunner",
        "--output-file",
        f"_build/unittest-report-{nox_common.PLATFORM}-py{session.python}.xml",
        "discover",
    )
    session.run("python", "-m", "coverage", "report", "-m")
    session.run(
        "python",
        "-m",
        "coverage",
        "xml",
        "-o",
        f"_build/unittest-coverage-{nox_common.PLATFORM}-py{session.python}.xml",
    )


@nox.session(python=nox_common.PY_VERSIONS)
def unittest(session: nox.Session):
    """Module testing with unittest"""
    cwd = Path.cwd()
    unittest_executor(session, project=cwd.name)
