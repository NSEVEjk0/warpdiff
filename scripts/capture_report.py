"""Check the viewer in Chrome and capture the published-report view."""

import argparse
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import URLError
from urllib.request import Request, urlopen


def run(report: Path, screenshot: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "site"
    with tempfile.TemporaryDirectory(prefix="warpdiff-preview-") as directory:
        root = Path(directory) / "site"
        shutil.copytree(source, root)
        shutil.copyfile(report, root / "report.json")

        handler = partial(SimpleHTTPRequestHandler, directory=root)
        with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            Thread(target=server.serve_forever, daemon=True).start()
            try:
                capture(server.server_port, screenshot)
            finally:
                server.shutdown()


def capture(port: int, screenshot: Path) -> None:
    installed = os.environ.get("CHROMEWEBDRIVER")
    driver_binary = shutil.which("chromedriver") or (str(Path(installed) / "chromedriver") if installed else "chromedriver")
    with subprocess.Popen([driver_binary, "--port=9515", "--log-level=WARNING"]) as driver:
        try:
            for attempt in range(30):
                try:
                    webdriver("/status")
                    break
                except URLError:
                    if attempt == 29 or driver.poll() is not None:
                        raise
                    time.sleep(0.2)
            session = webdriver("/session", {
                "capabilities": {"alwaysMatch": {"browserName": "chrome", "goog:chromeOptions": {
                    "args": ["--headless", "--disable-dev-shm-usage", "--no-sandbox"],
                }}},
            })
            path = f"/session/{session['sessionId']}"
            webdriver(path + "/window/rect", {"width": 1440, "height": 1080})
            webdriver(path + "/url", {"url": f"http://127.0.0.1:{port}/"})
            script = """
                const done = arguments[0];
                (async () => {
                    const report = document.querySelector('#report');
                    for (let attempt = 0; report.hidden && attempt < 40; attempt++) {
                        await new Promise((resolve) => setTimeout(resolve, 100));
                    }
                    const cards = document.querySelectorAll('.gpu-card');
                    if (report.hidden || !cards.length || document.querySelector('#evidence-link').hidden) {
                        return 'The published report did not render';
                    }
                    const controls = document.querySelectorAll('#shape-controls button');
                    if (controls.length > 1) {
                        const original = document.querySelector('#shape-controls [aria-pressed=true]');
                        const alternate = [...controls].find((button) => button !== original);
                        alternate.click();
                        if (alternate.getAttribute('aria-pressed') !== 'true') return 'Shape selection failed';
                        original.click();
                    }
                    await document.fonts.ready;
                    return null;
                })().then(done, (error) => done(String(error)));
            """
            error = webdriver(path + "/execute/async", {"script": script, "args": []})
            if error:
                raise RuntimeError(error)
            screenshot.write_bytes(base64.b64decode(webdriver(path + "/screenshot")))
            print(f"Verified the report viewer in Chrome. Screenshot saved to {screenshot}")
        finally:
            if driver.poll() is None:
                driver.terminate()


def webdriver(path: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(f"http://127.0.0.1:9515{path}", data, {"Content-Type": "application/json"})
    with urlopen(request, timeout=45) as response:
        return json.load(response)["value"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("screenshot", type=Path)
    args = parser.parse_args()
    run(args.report, args.screenshot)
