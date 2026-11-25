"""Test offline episode replay in an isolated Chromium instance."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chromium")
    args=p.parse_args()
    root=Path("results")
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,executable_path=args.chromium)
        page=browser.new_page(viewport={"width":1110,"height":920})
        errors=[]
        page.on("pageerror",lambda error: errors.append(str(error)))
        page.goto((root/"defense-replay.html").resolve().as_uri())
        assert page.locator("#scenario option").count()==3
        assert page.locator(".service").count()==6
        page.locator("#scenario").select_option("shifted")
        page.locator("#policy").select_option("inspect_rule")
        page.locator("#timeline").press("End")
        assert page.locator("#tick").inner_text()=="64"
        page.locator("#timeline").press("Home")
        page.locator("#play").click()
        page.wait_for_function("Number(document.getElementById('tick').textContent) > 2")
        page.locator("#play").click()
        assert page.locator("#play").inner_text()=="Play"
        page.locator("#timeline").press("End")
        page.locator("#timeline").press("ArrowLeft")
        page.screenshot(path=root/"defense-replay.png")
        page.set_viewport_size({"width":390,"height":844})
        assert page.locator("#chart").bounding_box()["width"]<390
        assert not errors,errors
        browser.close()
    print("Episode playback and mobile layout passed.")


if __name__=="__main__":
    main()
