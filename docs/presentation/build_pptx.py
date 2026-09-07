"""Run the validated PowerPoint builder with the Codex workspace runtime."""

import os
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    builder = Path(__file__).with_name("build_pptx.mjs")
    runtime_root = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
    node = runtime_root / "node/bin/node"
    runtime_python = runtime_root / "python/bin/python3"
    artifact_tool = runtime_root / "node/node_modules/@oai/artifact-tool"
    skill_root = Path.home() / ".codex/plugins/cache/openai-primary-runtime/presentations"
    skills = sorted(skill_root.glob("*/skills/presentations"))
    if not node.exists() or not runtime_python.exists() or not artifact_tool.exists() or not skills:
        raise SystemExit(
            "The editable slide builder requires the bundled Codex presentation runtime. "
            "Open this project in Codex Desktop and rerun this command."
        )

    env = os.environ.copy()
    env.update(
        {
            "ARTIFACT_TOOL_DIR": str(artifact_tool),
            "RUNTIME_NODE_MODULES": str(runtime_root / "node/node_modules"),
            "WORKSPACE_DIR": str(root),
            "SKILL_DIR": str(skills[-1]),
            "RUNTIME_PYTHON": str(runtime_python),
        }
    )
    subprocess.run([str(node), str(builder)], check=True, cwd=root, env=env)


if __name__ == "__main__":
    main()
