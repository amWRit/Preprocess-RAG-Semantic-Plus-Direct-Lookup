#!/usr/bin/env python3
"""
TFN Alumni Scraper - Updated with exact HTML selectors from your structure
Saves/merges alumni data into all_structured_data.json

Usage: 
    python scrape_alumni.py
"""

import os
import json
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_alumni_standalone():
    """Scrape alumni using exact HTML structure you provided."""
    print("🚀 TFN Alumni Scraper Starting...")
    
    base_url = "https://www.teachfornepal.org"
    alumni_base_url = f"{base_url}/tfn/alumni/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    all_alumni = []
    page = 1
    
    while page <= 20:
        page_url = f"{alumni_base_url}?page={page}"
        print(f"\n[*] Scraping page {page}: {page_url}")
        
        try:
            resp = requests.get(page_url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # EXACT SELECTOR from your HTML: .listingRow containers
            alumni_rows = soup.select('.listingRow')
            print(f"    Found {len(alumni_rows)} alumni rows")
            
            if not alumni_rows:
                print(f"    [+] No more alumni on page {page}")
                break
            
            new_alumni = 0
            for row in alumni_rows:
                alum_data = extract_alumni_from_row(row, base_url)
                if alum_data:
                    all_alumni.append(alum_data)
                    new_alumni += 1
                    print(f"    [+] {alum_data['name']} - {alum_data['profile_url']}")
            
            if new_alumni == 0:
                print(f"    [+] No new alumni extracted from page {page}")
                break
                
            print(f"    [+] Page {page}: {new_alumni} alumni")
            page += 1
            time.sleep(1.5)
            
        except Exception as e:
            print(f"    [-] Page {page} error: {str(e)}")
            break
    
    print(f"\n🎉 Total alumni scraped: {len(all_alumni)}")
    save_alumni_json(all_alumni)
    return all_alumni

def extract_alumni_from_row(row, base_url):
    """Extract alumni data from single .listingRow using your exact HTML."""
    try:
        # 1. Get name from .nameSection a.name link
        name_link = row.select_one('.nameSection a.name')
        name = name_link.get_text().strip() if name_link else "Unknown"
        profile_url = urljoin(base_url, name_link['href']) if name_link else ""
        
        # 2. Get school from next li in nameSection
        school_link = row.select_one('.nameSection li a[href*="/school/"]')
        school = school_link.get_text().strip() if school_link else ""
        
        # 3. Get bio from .textDespHolder p
        bio_elem = row.select_one('.textDespHolder p')
        bio = bio_elem.get_text().strip()[:2000] if bio_elem else ""
        
        # 4. Double-check View Profile button href matches name link
        view_btn = row.select_one('.viewProfileBtn')
        if view_btn and view_btn['href'] != name_link['href']:
            profile_url = urljoin(base_url, view_btn['href'])
        
        if not profile_url or not name:
            return None
            
        return {
            "name": name,
            "school": school,
            "bio": bio or "Full profile available online.",
            "profile_url": profile_url,
            "source": "tfn_alumni_scraped",
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        print(f"    [-] Row extraction failed: {str(e)}")
        return None

def scrape_full_profile(profile_url, headers):
    """Optional: Scrape full profile page for more details."""
    try:
        resp = requests.get(profile_url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Full bio from profile page
        full_bio_selectors = ['.content p', '.bio', '.description', 'article p']
        full_bio_parts = []
        for selector in full_bio_selectors:
            elems = soup.select(selector)
            for elem in elems:
                text = elem.get_text().strip()
                if len(text) > 50:
                    full_bio_parts.append(text)
        
        return ' '.join(full_bio_parts)[:4000]
    except:
        return None

def save_alumni_json(alumni_data):
    """Merge alumni into existing JSON."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(project_root, "public", "all_structured_data.json")
    
    existing_data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"    [*] Loaded existing JSON ({len(existing_data)} sections)")
        except Exception as e:
            print(f"    [!] Could not load existing JSON: {e}")
    
    existing_data["alumni"] = alumni_data
    
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Alumni saved to: {json_path}")

def test_single_page():
    """Test selectors on page 1."""
    print("\n🧪 Testing selectors on first page...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    url = "https://www.teachfornepal.org/tfn/alumni/?page=1"
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # Test exact selectors
    rows = soup.select('.listingRow')
    print(f"✅ Found {len(rows)} .listingRow containers")
    
    if rows:
        first_row = rows[0]
        name_link = first_row.select_one('.nameSection a.name')
        school_link = first_row.select_one('.nameSection li a[href*="/school/"]')
        bio_elem = first_row.select_one('.textDespHolder p')
        view_btn = first_row.select_one('.viewProfileBtn')
        
        print(f"✅ Name: {name_link.get_text().strip() if name_link else 'NOT FOUND'}")
        print(f"✅ School: {school_link.get_text().strip() if school_link else 'NOT FOUND'}")
        print(f"✅ Bio preview: {bio_elem.get_text()[:100] if bio_elem else 'NOT FOUND'}...")
        print(f"✅ View Profile href: {view_btn['href'] if view_btn else 'NOT FOUND'}")

if __name__ == "__main__":
    test_single_page()  # Test selectors first
    print("\n" + "="*60)
    alumni = scrape_alumni_standalone()  # Full scrape
    
    # Show sample results
    if alumni:
        print("\n📋 First 3 alumni:")
        for alum in alumni[:3]:
            print(f"  • {alum['name']} ({alum['school']})")
            print(f"    Bio: {alum['bio'][:100]}...")
            print(f"    URL: {alum['profile_url']}\n")
