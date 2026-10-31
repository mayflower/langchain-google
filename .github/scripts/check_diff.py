import json
import sys

LANGCHAIN_DIRS = {
    "libs/genai",
    "libs/vertexai",
    "libs/community",
    "libs/agent-sandbox",
}

if __name__ == "__main__":
    files = sys.argv[1:]
    dirs_to_run = set()

    if len(files) == 300:
        # max diff length is 300 files - there are likely files missing
        raise ValueError("Max diff reached. Please manually run CI on changed libs.")

    for file in files:
        if any(
            file.startswith(dir_)
            for dir_ in (
                ".github/workflows",
                ".github/tools",
                ".github/actions",
                ".github/scripts/check_diff.py",
            )
        ):
            dirs_to_run.update(LANGCHAIN_DIRS)
        elif "libs/genai" in file:
            dirs_to_run.update({"libs/genai"})
        elif "libs/vertexai" in file:
            dirs_to_run.update({"libs/vertexai"})
        elif "libs/community" in file:
            dirs_to_run.update({"libs/community"})
        elif "libs/agent-sandbox" in file:
            dirs_to_run.update({"libs/agent-sandbox"})
        else:
            pass
    json_output = json.dumps(list(dirs_to_run))
    print(f"dirs-to-run={json_output}")
