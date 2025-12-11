#!/usr/bin/env python3
"""
TFN Flexible Web Scraper - ✅ TRUE LAST PAGE DETECTION (NO LOOPING!)

Usage: python scripts/scraper.py --scrape-all
"""

import os
import json
import requests
import time
import argparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# ==================== CLEAN BASE CONFIGURATION ====================
SCRAPER_CONFIGS = {
    "alumni": {
        "base_url": "https://www.teachfornepal.org/tfn/alumni/",
        "page_param": "page",
        "container_selector": ".listingRow",
        "pagination_selector": "ul.pagination",
        "fields": {
            "name": ".nameSection a.name",
            "school": ".nameSection li a[href*='/school/']",
            "bio": ".textDespHolder p",
            "profile_url": ".nameSection a.name, .viewProfileBtn"
        },
        "output_key": "alumni"
    },
    
    "fellows": {
        "base_url": "https://www.teachfornepal.org/tfn/fellow/",
        "page_param": "page",
        "container_selector": ".listingRow",
        "pagination_selector": "ul.pagination",
        "fields": {
            "name": ".nameSection a.name",
            "bio": ".textDespHolder p",
            "profile_url": ".nameSection a.name, .viewProfileBtn"
        },
        "output_key": "fellows"
    },
    
    "schools": {
        "base_url_template": "https://www.teachfornepal.org/tfn/school/district/{district}/",
        "districts": ["dang", "tanahun", "parsa", "dhanusha", "sindhupalchowk"],
        "page_param": "page",
        "container_selector": ".listingRow",
        "pagination_selector": "ul.pagination",
        "fields": {
            "name": ".nameSection a.name",
            "location": ".nameSection li:nth-child(2)", 
            "bio": ".textDespHolder p",
            "profile_url": ".nameSection a.name"
        },
        "output_key_template": "schools_{district}"
    }
}


def scrape_flexible(sites=None):
    """Main flexible scraper entrypoint."""
    if sites is None:
        sites = ["alumni"]
    
    print(f"🚀 TFN Flexible Web Scraper - Sites: {', '.join(sites)}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    all_results = {}
    
    for site_name in sites:
        if site_name not in SCRAPER_CONFIGS:
            print(f"❌ Config missing for '{site_name}'")
            continue
        
        config = SCRAPER_CONFIGS[site_name]
        
        if site_name == "schools":
            school_results = scrape_schools_all_districts(config, headers)
            all_results.update(school_results)
        else:
            print(f"\n📄 Scraping '{site_name}' → {config['output_key']}")
            data = scrape_site(config, headers)
            all_results[config['output_key']] = data
            print(f"✅ {site_name}: {len(data)} items")
    
    save_results(all_results)
    return all_results


def scrape_schools_all_districts(schools_config, headers):
    """Scrape ALL school districts."""
    print("\n🏫 Scraping ALL School Districts...")
    results = {}
    
    for district in schools_config["districts"]:
        config = schools_config.copy()
        config["base_url"] = schools_config["base_url_template"].format(district=district)
        config["output_key"] = schools_config["output_key_template"].format(district=district)
        
        print(f"\n    📄 schools_{district}")
        data = scrape_site(config, headers)
        results[config["output_key"]] = data
        print(f"    ✅ {len(data)} schools from {district}")
        time.sleep(2)
    
    return results


def scrape_site(config, headers):
    """✅ FIXED: Check pagination BEFORE incrementing page!"""
    data = []
    page = 1
    
    while True:
        page_url = f"{config['base_url']}?{config['page_param']}={page}"
        print(f"\n        🔗 URL: {page_url}")
        
        # Scrape current page
        page_data = scrape_paginated_page(page_url, config, headers)
        current_page_count = len(page_data)
        data.extend(page_data)
        
        print(f"        📊 Page {page}: {current_page_count} items (total: {len(data)})")
        
        if current_page_count == 0:
            print(f"        ⏹️  NO ITEMS → END")
            break
        
        # ✅ CRITICAL FIX: Check pagination BEFORE going to next page
        soup = BeautifulSoup(requests.get(page_url, headers=headers, timeout=15).text, 'html.parser')
        pagination_status = analyze_pagination(soup, config, page)
        print(f"        📄 Pagination: {pagination_status}")
        
        # ✅ STOP if TRUE LAST PAGE
        if is_true_last_page(soup, config, page):
            print(f"        ⏹️  ✅ TRUE LAST PAGE → STOPPING!")
            break
        
        page += 1
        time.sleep(1.5)
    
    print(f"        ✅ Site complete: {len(data)} total items")
    return data


def is_true_last_page(soup, config, current_page):
    """✅ TRUE LAST PAGE: BOTH Prev+Next DISABLED OR no next links."""
    
    # 1. BOTH Prev AND Next are DISABLED → DEFINITE LAST PAGE
    prev_disabled = soup.select_one("li.disabled a.prev")
    next_disabled = soup.select_one("li.disabled a.next")
    
    if prev_disabled and next_disabled:
        return True
    
    # 2. Active page is 1 AND Next is disabled → SINGLE PAGE
    active_page = get_active_page_number(soup)
    if active_page == 1 and next_disabled:
        return True
    
    # 3. No clickable next links at all
    pagination = soup.select_one(config.get("pagination_selector", "ul.pagination"))
    if pagination:
        next_links = pagination.select("a[href*='page=']:not([href='#'])")
        has_next_page = False
        for link in next_links:
            href = link.get('href', '')
            try:
                next_page_num = int(href.split('page=')[-1].split('&')[0])
                if next_page_num > current_page:
                    has_next_page = True
                    break
            except:
                pass
        return not has_next_page
    
    return False


def analyze_pagination(soup, config, current_page):
    """Detailed pagination analysis."""
    status = []
    
    # Check prev/next disabled
    prev_disabled = soup.select_one("li.disabled a.prev")
    next_disabled = soup.select_one("li.disabled a.next")
    status.append(f"P:{'X' if prev_disabled else '-'} N:{'X' if next_disabled else '-'}")
    
    # Active page
    active = get_active_page_number(soup)
    status.append(f"Active={active}")
    
    return " | ".join(status)


def get_active_page_number(soup):
    """Get current active page number."""
    active_link = soup.select_one(".pagination li.active a")
    if active_link:
        href = active_link.get('href', '')
        if 'page=' in href:
            try:
                return int(href.split('page=')[-1].split('&')[0])
            except:
                pass
    return 1


def scrape_paginated_page(url, config, headers):
    """Extract items from page."""
    try:
        print(f"        🌐 Fetching: {url}")
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        containers = soup.select(config["container_selector"])
        
        if not containers:
            print(f"        📭 0 containers found")
            return []
        
        print(f"        📦 {len(containers)} containers found")
        page_data = []
        
        for container in containers:
            item = extract_item(container, config)
            if item:
                page_data.append(item)
        
        print(f"        ✅ {len(page_data)} VALID items extracted")
        return page_data
        
    except Exception as e:
        print(f"        ❌ Error on {url}: {str(e)[:60]}")
        return []


def extract_item(container, config):
    """Extract data from single container."""
    item = {
        "source": "web_scraped", 
        "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    fields = config.get("fields", {})
    for field_name, selector in fields.items():
        selectors = [s.strip() for s in selector.split(',')]
        for sel in selectors:
            elem = container.select_one(sel)
            if elem:
                item[field_name] = elem.get_text().strip()
                break
    
    for field_name in ['profile_url', 'url']:
        if field_name in item and item[field_name] and not item[field_name].startswith('http'):
            item[field_name] = urljoin(config.get("base_url", ""), item[field_name])
    
    if len([v for v in item.values() if v and v not in ["web_scraped"]]) <= 1:
        return None
    
    return item


def save_results(results):
    """Merge all scraped data into JSON."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(project_root, "public", "json", "scraped_data.json")
    
    existing_data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"📂 Loaded existing JSON ({len(existing_data)} sections)")
        except:
            pass
    
    existing_data.update(results)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved to: {json_path}")


# ==================== CLI ====================
def parse_args():
    parser = argparse.ArgumentParser(description="TFN Flexible Web Scraper")
    parser.add_argument("--sites", nargs="*", default=["alumni", "fellows", "schools"])
    parser.add_argument("--test", help="Test single site")
    parser.add_argument("--scrape-all", action="store_true", help="Scrape ALL sites")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.test:
        print(f"🧪 Testing last page detection for {args.test}")
        config = SCRAPER_CONFIGS[args.test]
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        test_url = f"{config['base_url']}?page=1" if args.test != "schools" else config["base_url_template"].format(district=config["districts"][1])
        print(f"🧪 Testing: {test_url}")
        resp = requests.get(test_url, headers=headers)
        soup = BeautifulSoup(resp.text, 'html.parser')
        is_last = is_true_last_page(soup, config, 1)
        pagination_status = analyze_pagination(soup, config, 1)
        print(f"🧪 Status: {pagination_status}")
        print(f"🧪 Is TRUE last page: {'✅ YES → STOPS!' if is_last else '❌ NO → Continues'}")
    elif args.scrape_all:
        results = scrape_flexible(["alumni", "fellows", "schools"])
    else:
        results = scrape_flexible(args.sites)
        
    print("\n📊 Summary:")
    total = sum(len(v) for v in results.values())
    print(f"🎉 Total: {total} items scraped!")
