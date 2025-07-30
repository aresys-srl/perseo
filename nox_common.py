# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Automating python testing, PERSEO common NOX sessions"""

from __future__ import annotations

import glob
import os
import shutil
import sys
from pathlib import Path

import nox

nox.options.error_on_missing_interpreters = True

_LICENSE_HEADER = """# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""


PY_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]
WIN32 = sys.platform == "win32"
PLATFORM = "win" if WIN32 else "linux"


def _get_only_file_matching_in_dir(directory: Path, pattern: str) -> Path:
    """Retrieving only a single file matching the input pattern in the selected folder.

    Parameters
    ----------
    directory : Path
        directory where to search for files
    pattern : str
        pattern to search for

    Returns
    -------
    Path
        Path to the searched file
    """
    matching_dir_content = list(directory.glob(pattern))
    assert len(matching_dir_content) == 1, (
        f"multiple files found: {matching_dir_content}"
    )
    return matching_dir_content[0]


def conda_recipe_builder(session: nox.Session, project_name: str) -> None:
    """Build a conda recipe using grayskull for the selected project.

    Parameters
    ----------
    session : nox.Session
        nox session
    project_name : str
        name of the project
    """
    dist_folder = Path("dist")
    if dist_folder.exists() and len(list(Path("dist").iterdir())) > 0:
        raise RuntimeError("dist folder is not empty, please remove it before building")

    sdist_builder(session)

    sdist_file = _get_only_file_matching_in_dir(dist_folder, "*.tar.gz").absolute()

    session.conda_install(
        "conda-build",
        "conda-verify",
        "grayskull",
        channel="conda-forge",
    )

    recipe_maintainer = "Aresys srl"
    session.run("grayskull", "pypi", str(sdist_file), "-m", recipe_maintainer)
    yaml_file = Path(project_name, "meta.yaml")
    assert yaml_file.exists()

    import_name = project_name.replace("-", ".")

    yaml_content = yaml_file.read_text(encoding="utf-8")
    yaml_content = yaml_content.replace(
        " " + project_name.replace("-", "_"), " " + import_name
    )
    yaml_file.write_text(yaml_content, encoding="utf-8")


def conda_package_builder(session: nox.Session, project: str) -> None:
    """Creating a conda package from a conda recipe.

    Parameters
    ----------
    session : nox.Session
        nox session
    project : str
        project name, with "-" as separator for namespace sub-packages
    """
    conda_build_dir = Path("conda_build_dir")
    session.run("conda", "build", project, "--output-folder", str(conda_build_dir))

    package = _get_only_file_matching_in_dir(
        conda_build_dir.joinpath("noarch"), "*.conda"
    ).absolute()
    shutil.copy(str(package), "dist")


def sdist_builder(session: nox.Session) -> None:
    """Function to build sdist"""
    session.install("build")
    session.run("python", "-m", "build", "--sdist", silent=True)


def wheel_builder(session: nox.Session) -> None:
    """Function to build a wheel"""
    session.install("build")
    session.run("python", "-m", "build", "--wheel", silent=True)


def doc_builder(session: nox.Session, project: str) -> None:
    """Documentation building function.

    Parameters
    ----------
    session : nox.Session
        nox session
    project_source : str
        project name, with "-" as separator for namespace sub-packages
    """
    if tag := os.getenv("CI_COMMIT_TAG"):
        build_dir = f"docs/build/{project}-{tag}-html-doc"
    elif sha := os.getenv("CI_COMMIT_SHORT_SHA"):
        build_dir = f"docs/build/{project}-{sha}-html-doc"
    else:
        build_dir = "docs/build/"

    session.install("-e", ".[docs]")

    session.run("python", "-m", "sphinx", "-M", "clean", "docs/source", build_dir)
    session.run("python", "-m", "sphinx", "-b", "html", "docs/source", build_dir)

    if os.getenv("CI") == "true":
        session.log(f"compressing '{build_dir}'")
        root_dir, base_dir = Path(build_dir).parent, Path(build_dir).name
        shutil.make_archive(build_dir, "zip", root_dir=root_dir, base_dir=base_dir)


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

    session.install("-e", ".[test]", silent=True)

    session.run(
        "python",
        "-m",
        "coverage",
        "run",
        f"--source={project}",
        "-m",
        "xmlrunner",
        "--output-file",
        f"_build/unittest-report-{PLATFORM}-py{session.python}.xml",
        "discover",
    )
    session.run("python", "-m", "coverage", "report", "-m")
    session.run(
        "python",
        "-m",
        "coverage",
        "xml",
        "-o",
        f"_build/unittest-coverage-{PLATFORM}-py{session.python}.xml",
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


@nox.session()
def build_sdist(session: nox.Session):
    """Building source distribution"""
    sdist_builder(session)


@nox.session()
def build_wheel(session: nox.Session):
    """Building wheel for distribution"""
    wheel_builder(session)


@nox.session(venv_backend="conda", python="3.11")
def build_conda_pkg(session: nox.Session):
    """Build a conda package from conda recipe"""
    cwd = Path.cwd()
    conda_recipe_builder(
        session=session,
        project_name=cwd.name,
    )
    conda_package_builder(session, project=cwd.name)


@nox.session(python=PY_VERSIONS)
def unittest(session: nox.Session):
    """Module testing with unittest"""
    cwd = Path.cwd()
    unittest_executor(session, project=cwd.name)


@nox.session()
def build_doc(session: nox.Session):
    """Building documentation"""
    cwd = Path.cwd()
    doc_builder(session, project=cwd.name)
