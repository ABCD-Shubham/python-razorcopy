import json
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from bs4 import BeautifulSoup
from app.seo_audit.schemas import AuditCreate
import os
from app.seo_audit.helpers import (
    get_page_content,
    analyze_meta_tags,
    analyze_headings,
    analyze_images,
    analyze_content,
    analyze_links,
    check_robots_txt,
    check_www_resolve,
    check_redirect_chain,
    check_analytics,
    check_custom_404,
    check_https,
    check_sitemap,
    check_schema_markup,
    check_search_engine_accessibility,
    check_pagespeed,
    get_seo_recommendations,
)

seo_audit_router = APIRouter()


@seo_audit_router.post("/audits")
def trigger_audit(audit: AuditCreate):
    def audit_generator():
        url = audit.url
        html, status, final_url = get_page_content(url)
        if not html:
            yield json.dumps({"error": f"Failed to fetch URL: {status}"})
            return
        base_url = final_url or url
        soup = BeautifulSoup(html, "html.parser")
        yield "Page fetched successfully.\n\n"

        analysis = {}
        robots = None
        meta_tags = None

        subtasks = [
            ("Meta Tags", "Analyzing meta tags...", lambda: analyze_meta_tags(soup)),
            ("Headings", "Analyzing headings...", lambda: analyze_headings(soup)),
            ("Images", "Analyzing images...", lambda: analyze_images(soup, base_url)),
            ("Content", "Analyzing main content...", lambda: analyze_content(soup)),
            ("Links", "Analyzing links...", lambda: analyze_links(soup, base_url)),
            (
                "Robots.txt",
                "Checking robots.txt...",
                lambda: check_robots_txt(base_url),
            ),
            (
                "WWW Resolve",
                "Checking www/non-www redirection...",
                lambda: check_www_resolve(base_url),
            ),
            (
                "Redirect Chain",
                "Checking redirect chains...",
                lambda: check_redirect_chain(base_url),
            ),
            (
                "Analytics",
                "Checking for analytics scripts...",
                lambda: check_analytics(html),
            ),
            (
                "Custom 404",
                "Checking for custom 404 page...",
                lambda: check_custom_404(base_url),
            ),
            ("HTTPS", "Checking HTTPS...", lambda: check_https(base_url)),
            ("Sitemap", "Checking for sitemap.xml...", lambda: check_sitemap(base_url)),
            (
                "Schema Markup",
                "Checking for structured data...",
                lambda: check_schema_markup(html, soup),
            ),
            (
                "Search Engine Accessibility",
                "Checking search engine accessibility...",
                None,
            ),
            (
                "PageSpeed",
                "Checking page speed...",
                lambda: check_pagespeed(base_url, api_key=os.getenv('PAGESPEED_API_KEY'), run_pagespeed=True),
            ),
        ]

        for label, message, func in subtasks:
            yield message + "\n"
            if label == "Meta Tags":
                meta_tags = func()
                analysis["meta_tags"] = meta_tags
            elif label == "Robots.txt":
                robots = func()
                analysis["robots"] = robots
            elif label == "Search Engine Accessibility":
                analysis["accessibility"] = check_search_engine_accessibility(
                    base_url, robots, meta_tags
                )
            else:
                result = func()
                key = (
                    label.lower()
                    .replace(".txt", "_txt")
                    .replace(" ", "_")
                    .replace("-", "_")
                )
                analysis[key] = result

        # --- Inner Pages Meta Audit (Sitemap Bulk Audit) ---
        sitemap = analysis.get("sitemap", {})
        url_list = sitemap.get("url_list", []) if sitemap.get("exists", False) else []
        if url_list:
            from concurrent.futures import ThreadPoolExecutor

            inner_results = []
            inner_link_map = {}

            def fetch_meta(u):
                html_content, status, final_url = get_page_content(u)
                if html_content:
                    soup = BeautifulSoup(html_content, "html.parser")
                    meta = analyze_meta_tags(soup)
                    links = analyze_links(soup, u).get("internal_links", [])
                    inner_link_map[u] = [
                        li.get("url")
                        for li in links
                        if isinstance(li, dict) and li.get("url")
                    ]
                    return {
                        "URL": u,
                        "Title": meta.get("title"),
                        "Title Length": meta.get("title_length"),
                        "Meta Description": meta.get("meta_description"),
                        "Description Length": meta.get("meta_description_length"),
                    }
                else:
                    inner_link_map[u] = []
                    return {
                        "URL": u,
                        "Title": None,
                        "Title Length": 0,
                        "Meta Description": None,
                        "Description Length": 0,
                    }

            with ThreadPoolExecutor(max_workers=5) as executor:
                inner_results = list(executor.map(fetch_meta, url_list))
            # Compute summary stats
            titles = [(row["Title"] or "").strip() for row in inner_results]
            descs = [(row["Meta Description"] or "").strip() for row in inner_results]
            missing_titles = sum(1 for t in titles if not t)
            missing_titles_list = [
                row["URL"] for row in inner_results if not (row["Title"] or "").strip()
            ]
            missing_descriptions = sum(1 for d in descs if not d)
            missing_descriptions_list = [
                row["URL"]
                for row in inner_results
                if not (row["Meta Description"] or "").strip()
            ]
            from collections import Counter

            title_counts = Counter(titles)
            dup_titles = [t for t, c in title_counts.items() if t and c > 1]
            duplicate_titles = []
            for t in dup_titles:
                duplicate_titles.extend(
                    [
                        row["URL"]
                        for row in inner_results
                        if (row["Title"] or "").strip() == t
                    ]
                )
            dup_title_groups = len(dup_titles)
            dup_title_pages = sum(title_counts[t] for t in dup_titles)
            desc_counts = Counter(descs)
            dup_descs = [d for d, c in desc_counts.items() if d and c > 1]
            duplicate_desc_groups = []
            for d in dup_descs:
                duplicate_desc_groups.extend(
                    [
                        row["URL"]
                        for row in inner_results
                        if (row["Meta Description"] or "").strip() == d
                    ]
                )
            dup_desc_groups = len(dup_descs)
            dup_desc_pages = sum(desc_counts[d] for d in dup_descs)
            title_length_issues = [
                row["URL"]
                for row in inner_results
                if row["Title Length"] < 50 or row["Title Length"] > 60
            ]
            desc_length_issues = [
                row["URL"] for row in inner_results if row["Description Length"] > 160
            ]
            inbound_counts = {u: 0 for u in url_list}
            for src, targets in inner_link_map.items():
                for target in targets:
                    if target in inbound_counts:
                        inbound_counts[target] += 1
            root_norm = base_url.rstrip("/")
            orphan_pages = [
                u
                for u, cnt in inbound_counts.items()
                if cnt == 0 and u.rstrip("/") != root_norm
            ]
            inner_summary = {
                "total_pages": len(inner_results),
                "missing_titles": missing_titles,
                "missing_titles_list": missing_titles_list,
                "missing_descriptions": missing_descriptions,
                "missing_descriptions_list": missing_descriptions_list,
                "duplicate_groups": dup_title_groups,
                "duplicate_titles": duplicate_titles,
                "duplicate_pages": dup_title_pages,
                "duplicate_desc_groups_count": dup_desc_groups,
                "duplicate_desc_groups": duplicate_desc_groups,
                "duplicate_desc_pages": dup_desc_pages,
                "title_length_issues": title_length_issues,
                "desc_length_issues": desc_length_issues,
                "orphan_pages": orphan_pages,
                "orphan_count": len(orphan_pages),
            }
            analysis["inner_audit_df"] = inner_results
            analysis["inner_summary"] = inner_summary

        yield "Done.\n"

        recommendations = get_seo_recommendations(analysis)
        output = {
            "url": url,
            "canonical_url": base_url,
            "analysis": analysis,
            "recommendations": recommendations,
        }
        # Send the final JSON object
        yield json.dumps(output, indent=2, ensure_ascii=False)

    return StreamingResponse(
        audit_generator(), status_code=200, media_type="application/json"
    )
