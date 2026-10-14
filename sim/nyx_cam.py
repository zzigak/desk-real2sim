"""Nyx path-traced camera for the desk (after ~/projects/4dcodebench-genesis/nyx_render.py): a Genesis sensor that
renders the simulated geometry itself, so what is drawn is exactly what was simulated.
Needs VK_DRIVER_FILES pointing at the NVIDIA ICD; set here so a bare run works."""
import os
import subprocess

os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
import numpy as np


def desk_lights(center, scale=1.2):
    """key / fill / rim point lights around `center`; intensities grow with distance^2 (the tone mapper clips hard:
    ~12 at 1 m for the key, see genesis-nyx-rendering memory)."""
    c = np.asarray(center, float)
    return [dict(type="point", pos=tuple(c + np.array((-0.9, -1.2, 1.5)) * scale), color=(1.0, 0.96, 0.92), intensity=12.0 * scale ** 2, shadow=True),
            dict(type="point", pos=tuple(c + np.array((1.4, -0.8, 0.9)) * scale), color=(0.88, 0.92, 1.0), intensity=4.0 * scale ** 2, shadow=False),
            dict(type="point", pos=tuple(c + np.array((0.3, 1.4, 1.4)) * scale), color=(1.0, 1.0, 1.0), intensity=5.0 * scale ** 2, shadow=False)]


def add_nyx(scene, pos, lookat, res=(960, 720), fov=40.0, spp=64, lights=None):
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    return scene.add_sensor(NyxCameraOptions(res=tuple(res), pos=tuple(pos), lookat=tuple(lookat), fov=fov, spp=spp,
                                             denoise=True, lights=lights if lights is not None else desk_lights(lookat)))


class Frames:
    """PNG per captured frame, muxed to mp4 at the end."""

    def __init__(self, outdir, fps=30):
        os.makedirs(outdir, exist_ok=True)
        self.outdir, self.fps, self.count = outdir, fps, 0

    def capture(self, cam):
        from PIL import Image
        from genesis.utils.misc import tensor_to_array
        rgb = tensor_to_array(cam.read().rgb)
        if rgb.ndim == 4:
            rgb = rgb[0]
        Image.fromarray(rgb).save(os.path.join(self.outdir, f"frame_{self.count:05d}.png"))
        self.count += 1

    def encode(self, mp4):
        if self.count:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(self.fps), "-i",
                            os.path.join(self.outdir, "frame_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "18", mp4], check=True)
        return mp4
