import pandas as pd
import time
from tqdm import tqdm
import airbnb_scraper.pyairbnb as pyairbnb

# ===== Step 0: Load host_ids (from open_guest_reviews_by_hosts.csv) =====
df_reviews = pd.read_csv(r"C:\Users\fraxi\OneDrive\Desktop\code task\Amsterdam\open_guest_reviews_by_hosts.csv")
host_ids = df_reviews["reviewer.id"].dropna().astype(str).unique().tolist()

# ===== Step 1: Set proxy information and initialize API =====
proxy_url = "http://brd-customer-hl_ae81836b-zone-residential_proxy1:jtf8my8ioxry@brd.superproxy.io:33335"
api_key = pyairbnb.get_api_key(proxy_url)
cookies = {}

results = []

# ===== Step 2: Iterate through each host, get the first listing, and crawl its details =====
for host_id in tqdm(host_ids, desc="Scraping host -> listing -> detail"):
    try:
        # Step 1: Get all listings (managedListings) for the host
        host_data = pyairbnb.host_details.get(api_key, cookies, host_id, "en", proxy_url=proxy_url)
        listings = host_data["data"]["presentation"]["userProfileContainer"]["userProfile"].get("managedListings", [])
        if not listings:
            continue

        # Step 2: Take the first listing's ID
        listing_id = listings[0].get("id")
        room_url = f"https://www.airbnb.com/rooms/{listing_id}"

        # Step 3: Get listing detail page
        listing_data, *_ = pyairbnb.details.get(room_url, "en", proxy_url=proxy_url)
        d = listing_data

        result = {
            "host_id": host_id,
            "listing_id": listing_id,
            "title": d.get("title"),
            "sub_description": d.get("sub_description", {}).get("title"),
            "description": d.get("description"),
            "amenities": ", ".join(
                v["title"]
                for cat in d.get("amenities", [])
                for v in cat.get("values", [])
                if v.get("available", False)
            ),
            "location_summary": " | ".join(
                f"{x['title']}: {x['content']}" for x in d.get("location_descriptions", [])
            ),
            "house_rules": " | ".join(
                f"{x['title']}: {', '.join(v['title'] for v in x.get('values', []))}"
                for x in d.get("house_rules", {}).get("general", [])
            ),
            "host_name": d.get("host", {}).get("name"),
            "room_type": d.get("room_type"),
            "person_capacity": d.get("person_capacity"),
            "is_super_host": d.get("is_super_host"),
            "latitude": d.get("coordinates", {}).get("latitude"),
            "longitude": d.get("coordinates", {}).get("longitude"),
        }

        results.append(result)
        time.sleep(2)

    except Exception as e:
        print(f"❌ Error for host_id {host_id}: {e}")
        time.sleep(2)

# ===== Step 4: Save as CSV =====
output_csv = r"C:\Users\fraxi\OneDrive\Desktop\code task\Amsterdam\host_listing_descriptions_chain.csv"
pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"✅ Done! Scraped {len(results)} records, saved to: {output_csv}")
