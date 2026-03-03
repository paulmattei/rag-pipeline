import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def clone_or_pull(repo_url: str, local_path: Path):
    try:
        # if local_path.exists():
        #     logger.info(f"Repo already exists at {local_path}, pulling latest...")
        #     subprocess.run(["git", "-C", str(local_path), "pull"], check=True)
        # else:
        #     logger.info(f"Cloning {repo_url} into {local_path}...")
        #     subprocess.run(
        #         ["git", "clone", "--depth", "1", repo_url, str(local_path)], check=True
        #     )
        if not local_path.exists():
            logger.info(f"Cloning {repo_url} into {local_path}...")
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(local_path)], check=True
            )
        else:
            pass
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed with exit code {e.returncode}")
        raise
    except FileNotFoundError:
        logger.error("Git is not installed or not found in PATH")
        raise
