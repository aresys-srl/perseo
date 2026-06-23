import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import nox


@contextmanager
def _hidden(path: Path):
    """Temporarily move *path* to a temp folder and restore it on exit (even on failure)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir) / path.name
        shutil.move(str(path), str(tmp))
        try:
            yield
        finally:
            shutil.move(str(tmp), str(path))


def _get_sha(session: nox.Session) -> str:
    sha = os.getenv("CI_COMMIT_SHORT_SHA")
    if sha is None:
        try:
            sha = session.run(
                "git", "rev-parse", "--short", "HEAD", external=True, silent=True
            )
            assert sha is not None
            sha = sha.strip()
        except Exception:
            sha = ""
    return sha


@nox.session()
def build_doc(session: nox.Session):
    """Building documentation"""

    if Path.cwd() != Path(__file__).parent:
        session.error("This session must be run from the root of the repository")

    site_dir = Path("site")
    build_info_file = Path("docs/about/build.template.md")
    zensical_toml = Path("zensical.toml")

    if site_dir.exists():
        shutil.rmtree(site_dir)

    tag = os.getenv("CI_COMMIT_TAG") or os.getenv("GITHUB_REF_NAME", "dev")
    sha = _get_sha(session)
    date = datetime.now().strftime("%Y-%m-%d")
    session.log(f"Building documentation for tag {tag} and sha {sha} on {date}")

    doc_name = f"{date}-{tag}-{sha}-html-doc"
    session.log(
        f"Documentation will be built in {site_dir} and archived as {doc_name}.zip"
    )

    session.log("Adding build info to documentation section")
    build_info = build_info_file.read_text()
    build_info = (
        build_info.replace("__SHA__", sha)
        .replace("__TAG__", tag)
        .replace("__DATE__", date)
    )
    build_info_file.parent.joinpath("build.md").write_text(build_info)

    session.log(f"Building documentation: {doc_name}")

    session.install("zensical", "mkdocstrings-python")
    session.run("pip", "list")
    with _hidden(build_info_file):
        session.run("zensical", "build", "-f", str(zensical_toml))

    if os.getenv("CI") == "true":
        session.log("compressing documentation")
        shutil.make_archive(
            f"documentation-{doc_name}", "zip", root_dir=".", base_dir=site_dir
        )
