"""
SAM3D TouchDesigner Client

Drop this into a Script DAT or Text DAT and call generate_glb() / generate_obj()
from a button CHOP, Python callback, etc.

Requirements (install in TD's Python or use system Python):
    pip install requests Pillow

Server must be running:  start_server.bat
"""

import io
import zipfile
import tempfile
import os
import requests
from PIL import Image

SERVER_URL = "http://localhost:8000"


# ── helpers ──────────────────────────────────────────────────────────────

def _image_to_png_bytes(image_path_or_top: str) -> bytes:
    """Load an image file and return PNG bytes."""
    img = Image.open(image_path_or_top).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mask_to_png_bytes(mask_path_or_top: str) -> bytes:
    """Load a mask image (any format) and return grayscale PNG bytes."""
    mask = Image.open(mask_path_or_top).convert("L")
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return buf.getvalue()


def check_server() -> bool:
    """Returns True if the server is up and the model is loaded."""
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        return r.json().get("status") == "ready"
    except Exception:
        return False


# ── main API ─────────────────────────────────────────────────────────────

def generate(
    image_path: str,
    mask_path: str,
    format: str = "glb",
    output_path: str = "output.glb",
    seed: int = 42,
    timeout: int = 300,
) -> str | list[str]:
    """
    Send image + mask to SAM3D server and save the result.

    Args:
        image_path:  Path to the input RGB image (PNG/JPG).
        mask_path:   Path to the binary mask image (grayscale PNG).
        format:      "glb" (default) or "obj".
                       glb → saves a single .glb file, returns its path.
                       obj → extracts a .zip into the same dir as output_path,
                             returns list of extracted file paths.
        output_path: Destination file path (glb) or base dir (obj).
        seed:        Random seed for reproducibility.
        timeout:     Request timeout in seconds (inference takes ~1-3 min).

    Returns:
        str  — absolute path to the saved GLB file (format="glb")
        list — absolute paths of extracted files    (format="obj")
    """
    fmt = format.lower()
    if fmt not in ("glb", "obj"):
        raise ValueError(f"Unsupported format '{format}'. Use 'glb' or 'obj'.")

    print(f"[SAM3D] Requesting {fmt.upper()} from {SERVER_URL}/generate ...")

    with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
        response = requests.post(
            f"{SERVER_URL}/generate",
            files={
                "image": ("image.png", img_f, "image/png"),
                "mask":  ("mask.png",  mask_f, "image/png"),
            },
            data={"seed": seed, "format": fmt},
            timeout=timeout,
        )

    response.raise_for_status()

    if fmt == "glb":
        out = os.path.abspath(output_path)
        with open(out, "wb") as f:
            f.write(response.content)
        print(f"[SAM3D] GLB saved to: {out}")
        return out

    # fmt == "obj" — unzip into the same directory as output_path
    output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(output_dir, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for name in zf.namelist():
            zf.extract(name, output_dir)
            extracted.append(os.path.abspath(os.path.join(output_dir, name)))
            print(f"[SAM3D] Extracted: {name}")
    return extracted


# Convenience aliases
def generate_glb(image_path, mask_path, output_path="output.glb", seed=42, timeout=300):
    return generate(image_path, mask_path, "glb", output_path, seed, timeout)


def generate_obj(image_path, mask_path, output_path="output.obj", seed=42, timeout=300):
    return generate(image_path, mask_path, "obj", output_path, seed, timeout)


# ── TouchDesigner integration example ────────────────────────────────────
#
# In TouchDesigner, use a Script CHOP or Button CHOP callback like this:
#
#   import sys
#   sys.path.insert(0, "C:/Users/Zach/Code/Lab & Tools/Meta_AI/SAM3D")
#   import touchdesigner_client as sam3d
#
#   def onValueChange(channel, sampleIndex, val, prev):
#       if val == 1:
#           if not sam3d.check_server():
#               print("SAM3D server is not running. Start start_server.bat first.")
#               return
#
#           # GLB (recommended — single file, textures included)
#           path = sam3d.generate(
#               image_path  = "C:/path/to/image.png",
#               mask_path   = "C:/path/to/mask.png",
#               format      = "glb",            # or "obj"
#               output_path = "C:/path/to/output.glb",
#               seed        = 42,
#           )
#           op("fileIn1").par.file = path
#
#           # OBJ (returns list of extracted files: .obj, .mtl, textures)
#           files = sam3d.generate(
#               image_path  = "C:/path/to/image.png",
#               mask_path   = "C:/path/to/mask.png",
#               format      = "obj",
#               output_path = "C:/path/to/output/output.obj",
#               seed        = 42,
#           )
#           obj_file = next(f for f in files if f.endswith(".obj"))
#           op("fileIn1").par.file = obj_file
#
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test — make sure the server is running first
    if not check_server():
        print("Server is not ready. Start start_server.bat and wait for 'Model ready.'")
        exit(1)

    IMAGE = "notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"
    MASK  = "notebook/images/shutterstock_stylish_kidsroom_1640806567/mask_014.png"

    # GLB
    glb = generate(IMAGE, MASK, format="glb", output_path="output.glb")
    print(f"GLB: {glb}")

    # OBJ
    files = generate(IMAGE, MASK, format="obj", output_path="output_obj/output.obj")
    print(f"OBJ files: {files}")
