import csv
import json
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

def scrape_slt_packages():
    url = "https://www.slt.lk/en/broadband/packages"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_timeout(10000)  # wait for JS to render

        soup = BeautifulSoup(page.content(), "html.parser")
        browser.close()

    packages = []

    # Each package card
    for card in soup.select("div.card-wrapp.any-time, div.color-card"):
        title_tag = card.select_one("h4.color-card-title")
        if title_tag:
            # Get only the text outside <strong>
            strong_tag = title_tag.select_one("strong")
            if strong_tag:
                strong_tag.extract()  # remove <strong> from h4
            title = title_tag.get_text(strip=True)
        else:
            title = None

        data_amount = card.select_one("span.data-val-amount")
        data_amount = data_amount.get_text(strip=True) if data_amount else None

        bundle_name = card.select_one("h5.bundle-name")
        bundle_name = bundle_name.get_text(strip=True) if bundle_name else None

        price = card.select_one("div.color-card-price")
        price = price.get_text(strip=True) if price else None

        if title or data_amount or bundle_name or price:
            packages.append({
                "title": title,
                "data_amount": data_amount,
                "bundle_name": bundle_name,
                "price": price
            })

    return packages


if __name__ == "__main__":
    results = scrape_slt_packages()

    # Save to CSV
    with open("slt_packages.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "data_amount", "bundle_name", "price"])
        writer.writeheader()
        writer.writerows(results)

    # Save to JSON
    with open("slt_packages.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"✅ Saved {len(results)} packages to slt_packages.csv and slt_packages.json")