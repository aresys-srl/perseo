# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Automating python testing, PERSEO common NOX sessions"""

from __future__ import annotations

from datetime import datetime
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


PY_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]
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


@nox.session()
def build_doc(session: nox.Session):
    """Building documentation"""

    if Path("site").exists():
        shutil.rmtree("site")

    tag = os.getenv("CI_COMMIT_TAG", "dev")
    sha = os.getenv("CI_COMMIT_SHORT_SHA")
    date = datetime.now().strftime("%Y-%m-%d")

    if sha is None:
        try:
            sha = session.run(
                "git", "rev-parse", "--short", "HEAD", external=True, silent=True
            ).strip()
        except Exception:
            sha = ""

    doc_name = f"{date}-{tag}-{sha}-html-doc"

    session.log("Adding build info to documentation section")
    build_info_file = Path("docs/about/build.template.md")
    build_info = build_info_file.read_text()
    build_info = (
        build_info.replace("__SHA__", sha)
        .replace("__TAG__", tag)
        .replace("__DATE__", date)
    )
    build_info_file.parent.joinpath("build.md").write_text(build_info)
    build_info_file.unlink()

    session.log(f"Current dir: {Path.cwd()}")
    session.log(f"Building documentation: {doc_name}")

    session.install("zensical", "mkdocstrings-python")
    session.run("pip", "list")
    session.run("zensical", "build", "-f", str(Path.cwd() / "zensical.toml"))

    if os.getenv("CI") == "true":
        session.log("compressing documentation")
        shutil.make_archive(
            f"documentation-{doc_name}", "zip", root_dir=".", base_dir="site"
        )
