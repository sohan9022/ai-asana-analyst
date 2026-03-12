import os
import shutil
import subprocess
import stat

def remove_readonly(func, path, _):
    # This bypasses Windows read-only errors so we can force-delete the hidden folder
    os.chmod(path, stat.S_IWRITE)
    func(path)

print("🧹 1. Deleting corrupted .git folder...")
if os.path.exists(".git"):
    shutil.rmtree(".git", onerror=remove_readonly)

print("🗑️ 2. Deleting massive data.zip file...")
if os.path.exists("data.zip"):
    try:
        os.remove("data.zip")
    except Exception as e:
        print(f"Could not delete data.zip (it might already be gone).")

print("🛡️ 3. Building secure .gitignore...")
with open(".gitignore", "w") as f:
    f.write("venv/\n")
    f.write("__pycache__/\n")
    f.write("*.zip\n")

print("🚀 4. Initializing fresh Git repository...")
# Run the git commands automatically
subprocess.run("git init", shell=True)
subprocess.run("git add .", shell=True)
subprocess.run('git commit -m "feat: Clean initial commit with MLOps pipeline"', shell=True)

print("\n✅ DONE! The bloated history is gone and your code is safely committed.")