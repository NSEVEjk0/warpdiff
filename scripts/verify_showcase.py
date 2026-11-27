"""Exercise all three viewer reports in Chrome and capture the showcase screenshot."""
import base64
import json
import os
import shutil
import subprocess
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import URLError

from scripts.capture_report import webdriver

FILES = {"softmax": "report.json", "rmsnorm": "rmsnorm.json", "swiglu": "swiglu.json"}
ARTIFACTS = Path("artifacts/browser")
SITE = Path("site")
DATA = {key: json.loads((SITE / file).read_text()) for key, file in FILES.items()}

VIEW = """const key = arguments[0], done = arguments[arguments.length - 1];
(async () => {
    const tab = document.querySelector('[data-benchmark="' + key + '"]');
    for (let attempt = 0; attempt < 40; attempt++) {
        if (tab?.getAttribute('aria-current') === 'page' && !document.querySelector('#report').hidden) break;
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    const chart = document.querySelector('.comparison').getBoundingClientRect();
    const panel = document.querySelector('.run-panel').getBoundingClientRect();
    return {
        selected: tab?.getAttribute('aria-current') === 'page',
        cards: document.querySelectorAll('.gpu-card').length,
        rows: document.querySelectorAll('#results-table tr').length,
        tabs: document.querySelectorAll('#benchmark-tabs a').length,
        correctness: [...document.querySelectorAll('.measurement-note')]
            .every(node => node.textContent.includes('Both outputs matched the reference')),
        check: document.querySelector('#pr-link').href,
        workflow: document.querySelector('#run-links a')?.href,
        json: document.querySelector('#evidence-link').href,
        repo: [...document.querySelectorAll('#revisions a')].map(node => node.href),
        viewport: window.innerWidth,
        width: document.documentElement.clientWidth,
        documentWidth: document.documentElement.scrollWidth,
        stacked: panel.top >= chart.bottom,
    };
})().then(done, error => done({error: String(error)}));
"""

INTERACT = """
const options = document.querySelectorAll('#shape-controls button');
options[1].click();
if (!document.querySelector('#results-table tr.selected-case') || options[1].getAttribute('aria-pressed') !== 'true') return false;
document.querySelector('[data-view=time]').click();
if (!document.querySelector('.time-bar')) return false;
document.querySelector('[data-view=change]').click();
options[0].click();
return Boolean(document.querySelector('.confidence-range'));
"""


def verify(session, origin):
    for width, height in ((1440, 1130), (768, 1024), (390, 960), (320, 960)):
        webdriver(session + "/goog/cdp/execute", {
            "cmd": "Emulation.setDeviceMetricsOverride",
            "params": {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": width < 1000},
        })
        webdriver(session + "/url", {"url": origin + "/"})
        for key, filename in FILES.items():
            if key != "softmax":
                tab = webdriver(session + "/element", {
                    "using": "css selector", "value": f"[data-benchmark={key}]",
                })["element-6066-11e4-a52e-4f735466cecf"]
                webdriver(session + f"/element/{tab}/click", {})
            result = webdriver(session + "/execute/async", {"script": VIEW, "args": [key]})
            report = DATA[key]
            repo = f"https://github.com/{report['repository']}"
            check = report.get("check_url", f"{repo}/pull/{report['pull_request']}/checks")
            assert result.get("selected") and result.get("tabs") == 3, (key, result)
            assert result["cards"] == len(report["cards"]) and result["correctness"], (key, result)
            assert result["rows"] == sum(len(card["cases"]) for card in report["cards"]), (key, result)
            assert repo in result["repo"] and result["check"] == check, (key, result)
            assert result["json"] == origin + "/" + filename and result["workflow"].startswith(repo + "/actions/runs/"), (key, result)
            assert result["viewport"] == width and result["documentWidth"] <= result["width"], (key, result)
            assert result["stacked"] == (width <= 840), (key, result)
            assert webdriver(session + "/execute/sync", {"script": INTERACT, "args": []}), key
            if width == 1440 or (width == 390 and key == "swiglu"):
                webdriver(session + "/execute/async", {
                    "script": "const done = arguments[0]; document.fonts.ready.then(() => done(true))", "args": [],
                })
                file = ARTIFACTS / f"{key}-{width}.png"
                file.write_bytes(base64.b64decode(webdriver(session + "/screenshot")))
            print(f"{key}: report, charts, links, and layout verified at {width}px")
    imported = webdriver(session + "/execute/async", {
        "args": [(SITE / FILES["swiglu"]).read_text()],
        "script": """const done = arguments[1];
        const input = document.querySelector('#report-input');
        const transfer = new DataTransfer();
        transfer.items.add(new File([arguments[0]], 'example.json', {type: 'application/json'}));
        input.files = transfer.files;
        input.dispatchEvent(new Event('change'));
        (async () => {
            const start = document.querySelector('#run-source');
            for (let i = 0; i < 30; i++) {
                if (start.textContent.startsWith('Imported report')) {
                    return document.querySelector('#evidence-link').href.startsWith('blob:') &&
                        document.querySelector('#pr-link').href.includes('warpdiff-triton-example');
                }
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            return false;
        })().then(done, () => done(false));
        """,
    })
    assert imported, "SwiGLU JSON import failed"
    print("Imported the example-repo JSON and verified its check and download links")


ARTIFACTS.mkdir(parents=True, exist_ok=True)
driver_dir = os.environ.get("CHROMEWEBDRIVER")
binary = shutil.which("chromedriver") or (str(Path(driver_dir) / "chromedriver") if driver_dir else "chromedriver")
with ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=SITE)) as server:
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with subprocess.Popen([binary, "--port=9515", "--log-level=WARNING"]) as driver:
        try:
            for attempt in range(30):
                try:
                    webdriver("/status")
                    break
                except URLError:
                    if attempt == 29 or driver.poll() is not None:
                        raise
                    time.sleep(.2)
            session = webdriver("/session", {"capabilities": {"alwaysMatch": {
                "browserName": "chrome", "goog:chromeOptions": {
                    "args": ["--headless", "--no-sandbox", "--disable-dev-shm-usage"],
                },
            }}})
            path = f"/session/{session['sessionId']}"
            webdriver(path + "/window/rect", {"width": 1440, "height": 1210})
            verify(path, f"http://127.0.0.1:{server.server_port}")
        finally:
            server.shutdown()
            if driver.poll() is None:
                driver.terminate()
