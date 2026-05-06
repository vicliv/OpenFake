#!/usr/bin/env python3
"""
Sora Video Scraper
==================
Scrapes public videos from OpenAI's Sora explore feed (sora.chatgpt.com/explore).

Uses Sora's internal backend API to:
  1. Fetch the explore/community feed with cursor-based pagination
  2. Extract video metadata (title, prompt, encodings, etc.)
  3. Download video files in the desired quality

Cloudflare Protection
---------------------
Sora sits behind Cloudflare which fingerprints TLS connections. Plain
`requests` gets blocked with a "Just a moment..." challenge page.

This script uses `curl_cffi` to impersonate a real browser's TLS
fingerprint, which bypasses Cloudflare transparently.

Requirements:
    pip install curl_cffi

    Optional (auto-extract cookies from your browser, no manual copy-paste):
    pip install browser_cookie3

Usage:
    1. Configure the variables in the CONFIGURATION section below.
    2. Set your auth cookies (see AUTH section for 3 options).
    3. Run:  python sora_scraper.py
"""

import json
import os
import re
import sys
import time
import logging
from urllib.parse import urljoin

# curl_cffi impersonates a real browser TLS fingerprint so Cloudflare
# doesn't block us with a JS challenge page.
try:
    from curl_cffi.requests import Session as CffiSession
except ImportError:
    print(
        "ERROR: curl_cffi is required to bypass Cloudflare.\n"
        "Install it with:\n"
        "    pip install curl_cffi\n"
    )
    sys.exit(1)

# =============================================================================
# CONFIGURATION — edit these to suit your setup
# =============================================================================

# Directory where downloaded videos will be saved
OUTPUT_DIR = "data/sora_videos"

# Directory for saving metadata JSON alongside videos
METADATA_DIR = "./data/logs/sora_scraping/sora_metadata"

# Quality to download: "source" (highest), "md" (medium), "ld" (low)
DOWNLOAD_QUALITY = "source"

# Maximum number of videos to download (set to 0 or None for unlimited)
MAX_VIDEOS = 50

# Seconds to wait between API requests (be polite)
REQUEST_DELAY = 1.0

# Seconds to wait between video downloads
DOWNLOAD_DELAY = 0.3

# Feed type for the explore page: "latest" or "top_7d" (top of last 7 days)
FEED_TYPE = "latest"

# Number of posts to request per page (the API typically supports 20-50)
PAGE_SIZE = 20

# Request timeout in seconds
REQUEST_TIMEOUT = 30

# Download timeout in seconds (videos can be large)
DOWNLOAD_TIMEOUT = 120

# Whether to save metadata JSON files alongside videos
SAVE_METADATA = True

# Whether to skip videos that have already been downloaded
SKIP_EXISTING = True

# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = "DEBUG"

# -------------------------------------------------------------------------
# AUTHENTICATION
# -------------------------------------------------------------------------
# Sora's backend API requires a valid session. You have 3 options:
#
# >>> OPTION 1 (easiest): Auto-extract cookies from your browser <<<
#   Install browser_cookie3:  pip install browser_cookie3
#   Set BROWSER_NAME below to "chrome", "firefox", "edge", etc.
#   Make sure you're logged into sora.chatgpt.com in that browser.
#   Leave AUTH_COOKIE empty — the script will grab cookies automatically.
#
# >>> OPTION 2: Manually paste cookies from DevTools <<<
#   1. Open Chrome -> sora.chatgpt.com/explore (logged in)
#   2. F12 -> Network tab -> filter "backend-api" -> refresh page
#   3. Click any request -> Headers -> Request Headers
#   4. Copy the full "Cookie:" header value
#   5. Paste it into AUTH_COOKIE below
#
# >>> OPTION 3: Bearer token <<<
#   Same DevTools steps, but copy the "Authorization:" header instead.
#   Paste into AUTH_BEARER (including the "Bearer " prefix).
# -------------------------------------------------------------------------

# Option 1: auto-extract from browser (set to "chrome", "firefox", "edge",
#            "opera", "brave", or "" to disable)
BROWSER_NAME = ""

# Option 2: manually paste cookie string
AUTH_COOKIE = "oai-nav-state=1; oai-thread-sidebar=%7B%22isOpen%22%3Afalse%7D; oai-did=c8fa6119-ee43-4b84-9952-047f5ecfdfad; oai-hlib=true; _account_is_fedramp=false; _dd_s=aid=02d231bd-3f0d-4c5b-9aaf-646c1a6c8662&rum=0&expire=1774293756783&logs=1&id=b114fcd9-ad13-4ae7-8e51-850591bd6f6c&created=1774292409296; __cflb=0H28vzvP5FJafnkHxiscDFksMo5URnbBB65B7PmukJP; cf_clearance=dBazo_S5BFdNMzevZ82lOo6R_zO1W.AXPMI_ZLXR4RI-1775360387-1.2.1.1-ZIBiXfmwAvIkeB61IgV3mTcTw3epq9Mcj3tgHzh3xpJe2KTt93VsRRHXZ8NZengdATWCqZTnUFmCOYVCEnDy_EGajFRIU6KXMM87W3KHkPCNegKFE3ub0Z4NAnt5ewdBN.rRvMyD5TMQVOCv0OvXLJQ3onWSSHIOx2QEt3s6jLPJ.qxE0U58VW9FTVr14l1udW.fbmNJPxUssxgMeYOBQu09XPi.VofBLB2MSVm_u53rkpvP0K0rnOX9O9XtgylD87n68aO70.5iR8EpgJSjbzS0wT9.jUh3i.0QwujKWVluTq9K4SUh39KPcLHUoed0kPyp4GYByAX1tDE8AdBASA; oai-sc=0gAAAAABp0dmwXa27KNmmC-xqPcGOlYBkJu6WkhN6BhZ9R_itbT-MsZFTHoMxyd5B5MGG3SS0ufPaGn49WHfkJ8wxKlX7zmG64AUegQw00L92LufB2aLitImEIIqemAi_tfalBEPk1dgqHeb0UW37XLYT_rG6cgclDipASgA1cTqR2qsdJc1smanYj4yfeb7_BxPsD7lsLneWUbOO-hFsvENxAXcpTb4ZSEP4bXpykbq0QRUp7U6HArk; __cf_bm=lwRkrMywFseVSuztNjD0J5Z_EgplEFy_l8c09eA2cWw-1775361428.06045-1.0.1.1-bAm0n07w2CWDKjUdOjB1cvMm90MnPE34GeK9DlwgxZ7ZCIVVDYHvvcRwvk.HsW8MF2iuEZhZ1vxN_X7dZk5FJbp5pmocaNDT4DpDMvPbt7mBajXkprJZK6HJ4ZzpiqMI; __Secure-next-auth.session-token.0=eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..u4wxAmmEKKhkWNEJ.kHO97iH1KWOIKY5a_uUb-TKuG-mL1mBonjQ05GXVx5iZ9meaQmxCMxtm6_BUrf7VL_WWsiqVlgydM-mhhHyTlYLxQ1pgYUm1qpnAfTdw-PZkEpD-V2F2nkif7YXyoUEvp-WTLIaZDHeFIX2OzRbx_MfmPNwF7UKah4fSRiyd3ekcR4rs0VpHhraQx0PijR0OEpvtViDWJ2CMhF1ae_uC93Bnxk4WoCbso8UGPIIoE0Q3msk0zBA0w6--xG9Sk5x_TXzMj6M7oL3mvXqyjdKDNUnCzYJZYaERpmaK9uIYqYYL7Kj_btOH-kE3gH5VrwwdoWwwXgwSw-NCNCLUkFT6_Vjif0Oav4uGx6F3IR5fAaGgJVWYl-lWXbf8c6qCxTHrlRP0NC7JD41fNyoH5GOMiX0l2scAokytFQ-5ZJTtsn9ePm394IViZfHuAni8jKeLP3jdLlbsA_JM-BrCIBbr3Liu2tWPfEBsv6k90yjrJjF9onsTQO5ySaRSjFTQ_CjearqLQ1XtgsIt7mq2dKXLLkxeiMtMDvp4_Z9EukjOAGrB7YalbazMs7qNYiA0Yj0B-GutWzP-P2vHHXqq_SAER5mX4DW8Dg1SPloEQR4bTNBAIv3DC2fJpA5gVPalXe-dCnx2n3ExowyrAVlxmiX5khWo_33MvSWZWHLke0O3pwgbWNKge4hxTD1Wr9QR2Q-E9tyDJ1qchcCKtsjeHk2zt1QbJxolp0u0Aj_A6S97xEf_uXXeSpNI1TgwH3Ha8kkDvAh9vgqMPhuHA5KqVsghwKLZPwvLkNqJZyPcu09xT6MNYic-QmEdtqCsiZEG-UfCG9AXUHTJ_AWSUXmSMff016Jf7_FH0xbvAt1kuEyWl-sxU7_y3UwwPYoJFw_-T6xjZyoWb9qyGDk_xaeL-PvCnKxYA5-Dc9Pb2USy4gooss6rwEHFYBoHHKZJmGlEuv98nu563Er1niBTHgCXKMKU2AhGYENpUYaMJyn9GZD_N25dBOHlOOO3K7_VaMj71yIZHGN1vUf2QFZ5HCeHbsT22WVBVdAM8DB07SJc9wL5V9f7ZU909idH_o-mqFwsYJbvgFVwQJuCpuV1jDHkNY5CEQq2zvY3sZyJdf1qgRGm7AjSXUJbocetskfaC6750iAeNRdHI6tH-Tciard88qYerlcO1MQcomM68KQagYCzb3JbZctkiXUknq-N-rlKjtekSSFZieT4VLjAa6WJ66IN49VGunjTsAwjEZNAQukdcSyYm1xGqwUlOgPYDVRFBtKKc7fFduyNgEnjEt1KAEwYKmWUvKu01_nidyMSNo__kNJBwRhhacpla9TvJ6JX790U9PchStjKIGydqddKHXbYYOnRO8-YCE7_cASG76k6a-f7yX4U0P01OnZCH93d6_5domQ8qA42JNQiPIOvdYB4AiW3s6rmNvZKujFWh4vqgRmPhiy7JM4tO5LbYRgVzKvabNs9k4LtCV1rmSxCt0QYFU7zKJz_puiyf_VdEWRMjrNAtMXJ1hA7EnJdm2uMCJeatj4gQQTmvP1HUQHVMXlOP8NfyYi5VVjpg8bjrCaapcSZ3Y0OdLxH-iiOY5zq91i8KyJ1xmK53EmO76TxV8sA4_Y1G7DguYCychBy7tombbmko9Ql-vbm_wBnT22IVP1CtKFcDonw7P6YZVlOnD7CxyET8kvscPlTwSRsfKDsytWOvHEEuM5Wg9s6Ciuy2Gx-qym0fZM3jcOecbSF2rwL3Yc20gUoUTFTq7xfydGeP59FYVMFpqJEkJseotY2ydHUyf3LmoedsGL2FeahAzccak5igFX_XC2NyXz1tyGbWxLs17hxmOhw8Wz1Qt8Y52cskHNJY0g-Cbq65Xvs7giWBBMbndqbF74JTup07VhxIX1AUDsJSXztvHmuuNlV00ZyVuegbN3Y7WLYBSpfro3bw3i3H7di_D6b94_RPyibxNZ4DxoJDzGsDSchCXlKf3o77JUibL8EyeomBkbWnD6u2m9d12w5Hv2zGrkIT-pmgMLiiBYP5AjNoLMWMCdVUhlN5lK7ik23jZ0FN5HfFGZtI0wBGqqARPFLFGKRVQ3LhzDT5yOku2AW8zwQRn1SUWa_jOlmT70jIwV2Se53PtuO4Kv4pYhRHx6Bhb9HILozRy2tmE2VoMPgW2M5R3Tyi2IdEHpk4WpoE0sdOdpa_p_gVEICn0lowf5ItU1jWRG2mIi_Tm8rbL3C1Ls2vSMR1c_pZkOTVukjrdjOtcy813vrhkcdo5AC8_hsS1vxn6Wdg4P0rpxbQNj6j2VpEAzJ2hBNSYhSzVijrShr53o0HR8oKxWGEOJom-1oYhZ4MjA75UTaiLQ_GQmyPdwyUJPbvdl11BsKcTJ_ZNt3vsqVsgmuFl6iFff5bw0I5jXSuqHW4EL8gKt4CamN7kDo9m4hIO_h_1IDP5GqXWSOW8BgDUauHQ58Z6XyW37qqsR3SBFJ3JfDEPGA-RrP9JyVKx7rKHt321gtWjpJ7503pJLX8sTtEI4XLD1aTmj__XdSgzZYIIuGV8BcJ42Cn15Q3-tNT2R5k4eAgMZuQ1wk4HqGT0z1n8WrF_IrMPFBlFj9Z_Z8M1C9j73qmvcBstCHe4pewt62XiUXHnaxGdsL9VTVWBMpqnTY30L-_IPGk82LtsX8qQc0PE9QUA5DoQrWbpQzMe8TFSmdnuO03zbHkMx4odOP-o7kSXHGyxFjaDKvrxwfE4017S2-8DeJKJUDb4VyzHlBeAwVjHKn9IM0dKPyf-ZU6Vp0cQrWLpbj_CQnmZVU7e6YsoWXM60j2RS_PYzCAzsFkVfaxk9MJmSqZlRqPbIdCXWkGi5buhG8AIdbZS3lHwd__criwVk1eDvc_9XrVVl0ISwHHrQwGUx-2a59ZDJsWeM452MdTyERG7btw6Frx5tO4qt-3_eOyZXtis6ewbOtGdmHOQ6lzS_msuPPAJXhmEndRkWgp9TnFMfb5iDqPBOz7szycr9K9afI3Pc5NfO8enPP_h1TP5yACNEir-wgWNR7mebGMNkqHl6XJ_o8nwUS8W0QZNWCvuddHyHX5K5z2GDrPcIS6badfwwEvfH-nWaKDLCUs0QpxtLJKJDgCP1R8CBWQXurvWiJdLtA-MfKSZ4l1Rqs0BV_r33NQyNpXZn4qMaXiZU91WZsqrBPJS3cRYeSmIjzzQAvftXOLVIYPS01Ru7YAnpx145qDC8AOxxJb_QiUyYt8mFx4V7NnfNl4Hrkur7sGO-zVCvEAFRQ7n864a9oRKDjMdyXF2WIt-YavZnaGddwgzKfzU0pAPWcQsPJ9GSRLSZ2pqSPgUZbrPRRC9ZMdVcn96JGpFKM2x_KRG842SNHTsVMd4_t1_sKogpzAGbf75oaaz6ayUv_z-FGB9Jzfe1nCAM7vTWXv6JQ5ARSnLLcm4My53sZaPxtGDTnq25vPrK1wROm5ml01kgMdErx-T501SExX3hrNQBPuL3y43xjdZtxU8Z_wJWYN0NhVnpKDHokZxQ10jt8xb-kFQreWcVk0joHK6KUqOWFcZkmigI091pxMjNvWA7IULoZjmnfFbR9aRTFaHfxof39vBdcLTsUNBHRZWm1S9OU66sWwaoBPhy3gDZOEPQ5f9mebG0gdBgCpg9vJScBP5XodXped_lwpl2vYXRptLCmsjD1jBV3UZ8m3WjaYCrysl0iqaP1j8jo-PZLRk-DEXQ4iJGAb-uohclrNJ7rafqbYoz4IVmk79xZCeXqz34tGDWdmGHzZLs-7n3B_S6VBeOHwxKOk8D7u3N2WRs7higPR97tyZWHofbc9apD3ZEJQPCUvziKWXK5VWNLLkZ-M-A_RCc1WegZ0Luw7uz6hO_H1t0jsK9OybrSJH; __Secure-next-auth.session-token.1=6H7IEGgYk9Dzv7xmGiPniL1nNoqZ7TgD2wRB1wMeBcO07_Bert-a80zi9P1owZE_ZnHloDejy18BJwxMWxAidkPuW_9jJgtaZmJX3zoEkkCnQ3D9KYJtxehxa-qCHuFxVLq4zryuD-w3vI-w_Gx8Rug3RRK7cutz940e66cazmQEXkzX6Tkhc8WzDna2snVvB76eFb54A8-MLzYsOXiBY17Yw1F4S8N_HPrDtTUUhpO56j-d2hjGNG8oLzTnan4GNZrFKej6br1XFn4Y4XDBYr6p-885ZmbAigDPinLodd2xM-K5rv-Q2n1nn_WjgxMhA9auk9w4o_zTVxk3HI9EkAkA_6UHgwQNwieGec8_0Zn_Z_CRmO1o12rpyWyvsNJsXR0JPl_D6Z4qfjrgh7RkrIAUmFaRkSLbnhkTU3FkwEcspuZXdLe9vVRpHGZjneFsog_0g7MJ7FlLMe1oTfnyTJUUE-BYEf_rhfJ9xF2nRZe9fBzcE.yTS9izkZ2R3Pjox9dX-kew"

# Option 3: bearer token
AUTH_BEARER = ""

# -------------------------------------------------------------------------
# API ENDPOINTS — update these if Sora changes their backend paths.
# You can discover the current paths from your browser's DevTools Network tab.
# -------------------------------------------------------------------------
SORA_BASE_URL = "https://sora.chatgpt.com/explore"

# The explore/community feed endpoint:
EXPLORE_API_PATH = "/backend-api/v1/community/posts"

# Individual post/video detail endpoint (append /{post_id}):
POST_DETAIL_API_PATH = "/backend-api/v1/community/posts"

# Individual generation detail endpoint (append /{gen_id}):
GENERATION_DETAIL_API_PATH = "/backend-api/v1/generations"

# The browser TLS fingerprint to impersonate (curl_cffi).
# Options: "chrome120", "chrome124", "chrome131", "safari17_0", "edge101", etc.
# Full list: https://curl-cffi.readthedocs.io/en/latest/impersonate.html
IMPERSONATE_BROWSER = "chrome131"

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sora_scraper")


def load_browser_cookies(browser_name: str) -> dict:
    """
    Use browser_cookie3 to pull session cookies from a local browser.
    Returns a dict of {cookie_name: cookie_value} for sora.chatgpt.com.
    """
    try:
        import browser_cookie3
    except ImportError:
        logger.error(
            "browser_cookie3 is not installed. Install it with:\n"
            "    pip install browser_cookie3\n"
            "Or set AUTH_COOKIE manually instead."
        )
        return {}

    browser_funcs = {
        "chrome": browser_cookie3.chrome,
        "firefox": browser_cookie3.firefox,
        "edge": browser_cookie3.edge,
        "opera": browser_cookie3.opera,
        "brave": browser_cookie3.brave,
        "chromium": browser_cookie3.chromium,
    }

    func = browser_funcs.get(browser_name.lower())
    if not func:
        logger.error(
            f"Unknown browser '{browser_name}'. "
            f"Supported: {', '.join(browser_funcs.keys())}"
        )
        return {}

    try:
        logger.info(f"Extracting cookies from {browser_name}...")
        cj = func(domain_name="sora.chatgpt.com")
        cookies = {c.name: c.value for c in cj}
        logger.info(f"Got {len(cookies)} cookies from {browser_name}.")
        return cookies
    except Exception as e:
        logger.error(f"Failed to load cookies from {browser_name}: {e}")
        logger.error(
            "Make sure the browser is closed (some browsers lock the cookie DB), "
            "and that you're logged into sora.chatgpt.com."
        )
        return {}


def build_session() -> CffiSession:
    """
    Create a curl_cffi session that impersonates a real browser.

    curl_cffi sends requests with the same TLS fingerprint, HTTP/2
    settings, and header order as a real browser, which is what gets
    past Cloudflare's bot detection.
    """
    session = CffiSession(impersonate=IMPERSONATE_BROWSER)

    session.headers.update({
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"{SORA_BASE_URL}/explore",
        "Origin": SORA_BASE_URL,
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
    })

    # --- Load cookies ---
    # Priority: BROWSER_NAME > AUTH_COOKIE > AUTH_BEARER

    if BROWSER_NAME:
        cookies = load_browser_cookies(BROWSER_NAME)
        for name, value in cookies.items():
            session.cookies.set(name, value, domain="sora.chatgpt.com")

    if AUTH_COOKIE:
        # Parse "name=value; name2=value2; ..." format into the cookie jar
        for pair in AUTH_COOKIE.split(";"):
            pair = pair.strip()
            if "=" in pair:
                name, value = pair.split("=", 1)
                session.cookies.set(name.strip(), value.strip(), domain="sora.chatgpt.com")

    if AUTH_BEARER:
        token = AUTH_BEARER if AUTH_BEARER.startswith("Bearer ") else f"Bearer {AUTH_BEARER}"
        session.headers["Authorization"] = token

    return session


def fetch_explore_feed(session: CffiSession, cursor: str = None) -> dict:
    """
    Fetch a page of the explore/community feed.

    Returns the raw JSON response which typically contains:
        {
            "posts": [ ... ],
            "cursor": "next_page_token",
            "has_more": true/false
        }
    """
    url = f"{SORA_BASE_URL}{EXPLORE_API_PATH}"

    params = {
        "limit": PAGE_SIZE,
        "sort": FEED_TYPE,
    }
    if cursor:
        params["cursor"] = cursor

    logger.debug(f"Fetching explore feed: {url} params={params}")

    try:
        resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)

        # Check if Cloudflare blocked us (returns HTML instead of JSON)
        content_type = resp.headers.get("content-type", "")
        if resp.status_code == 403 and "text/html" in content_type:
            logger.error(
                "Cloudflare blocked the request (got HTML challenge page).\n"
                "This means your cookies are missing or expired.\n\n"
                "To fix this:\n"
                "  1. Log into sora.chatgpt.com in your browser\n"
                "  2. Re-copy your cookies from DevTools (they expire!)\n"
                "  3. Or install browser_cookie3 and set BROWSER_NAME\n"
            )
            raise SystemExit(1)

        resp.raise_for_status()

        data = resp.json()
        logger.debug(f"Feed response keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
        return data

    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Error fetching feed: {e}")
        # Log a snippet of the response body for debugging
        try:
            body = resp.text[:300]
            logger.debug(f"Response body: {body}")
        except Exception:
            pass
        raise


def extract_posts_from_response(data: dict) -> tuple:
    """
    Extract the list of posts and the next cursor from an API response.

    Handles multiple possible response shapes since the exact API contract
    may change. Returns (posts_list, next_cursor_or_None).
    """
    posts = []
    next_cursor = None

    # Try common field names for the posts array
    for key in ("posts", "items", "results", "data", "generations"):
        if key in data and isinstance(data[key], list):
            posts = data[key]
            break

    # If the whole response is a list, use it directly
    if not posts and isinstance(data, list):
        posts = data

    # Try common field names for the pagination cursor
    for key in ("cursor", "next_cursor", "next", "nextCursor", "next_page"):
        if key in data and data[key]:
            next_cursor = data[key]
            break

    # Some APIs nest pagination info
    if not next_cursor and "pagination" in data:
        pag = data["pagination"]
        for key in ("cursor", "next_cursor", "next"):
            if key in pag and pag[key]:
                next_cursor = pag[key]
                break

    # Check has_more flag
    has_more = data.get("has_more", data.get("hasMore", True))
    if not has_more:
        next_cursor = None

    return posts, next_cursor


def extract_video_info(post: dict) -> dict | None:
    """
    Extract video download info from a single post/generation object.

    The Sora API returns generation objects with this structure:
        {
            "id": "gen_...",
            "title": "...",
            "url": "https://videos.openai.com/vg-assets/.../source.mp4",
            "encodings": {
                "source": {"path": "https://videos.openai.com/.../source.mp4", ...},
                "md":     {"path": "https://videos.openai.com/.../md.mp4", ...},
                "ld":     {"path": "https://videos.openai.com/.../ld.mp4", ...},
                ...
            },
            "user": {"username": "..."},
            "actions": {"100": "prompt text here"},
            ...
        }

    Returns a normalized dict or None if no video URL found.
    """
    info = {}

    # --- ID ---
    info["id"] = (
        post.get("id")
        or post.get("generation_id")
        or post.get("gen_id")
        or post.get("post_id")
        or "unknown"
    )

    info["title"] = post.get("title") or post.get("text") or ""
    info["created_at"] = post.get("created_at") or post.get("createdAt") or ""
    info["model"] = post.get("model") or ""

    # --- Prompt / action text ---
    info["prompt"] = ""
    if "prompt" in post and post["prompt"]:
        info["prompt"] = post["prompt"]
    elif "actions" in post and isinstance(post["actions"], dict):
        info["prompt"] = " | ".join(str(v) for v in post["actions"].values())
    elif "text" in post:
        info["prompt"] = post["text"]

    # --- Creator ---
    user = post.get("user") or post.get("author") or {}
    info["creator"] = user.get("username") or user.get("name") or "unknown"

    # --- Find the video URL ---
    video_url = None
    width = post.get("width", 0)
    height = post.get("height", 0)
    duration = 0

    # Method 1: 'encodings' dict (the known Sora API structure)
    encodings = post.get("encodings") or {}
    quality_preference = [DOWNLOAD_QUALITY, "source", "md", "ld"]
    for q in quality_preference:
        if q in encodings and isinstance(encodings[q], dict):
            enc = encodings[q]
            video_url = enc.get("path") or enc.get("url")
            width = enc.get("width", width)
            height = enc.get("height", height)
            duration = enc.get("duration_secs", 0)
            info["quality"] = q
            if video_url:
                break

    # Method 2: direct 'url' or 'video_url' field
    if not video_url:
        candidate = post.get("url") or post.get("video_url") or post.get("videoUrl")
        if candidate and ("videos.openai.com" in candidate or candidate.endswith(".mp4")):
            video_url = candidate
            info["quality"] = "unknown"

    # Method 3: nested inside 'media' or 'video' object
    if not video_url:
        media = post.get("media") or post.get("video") or {}
        if isinstance(media, dict):
            video_url = media.get("url") or media.get("path") or media.get("source")
            width = media.get("width", width)
            height = media.get("height", height)
            duration = media.get("duration", media.get("duration_secs", 0))
            info["quality"] = "unknown"
        elif isinstance(media, list) and media:
            for m in media:
                if isinstance(m, dict) and m.get("type", "video") == "video":
                    video_url = m.get("url") or m.get("path")
                    width = m.get("width", width)
                    height = m.get("height", height)
                    break

    # Method 4: look inside nested 'generation' or 'generations'
    if not video_url:
        gen = post.get("generation") or {}
        if isinstance(gen, dict) and gen:
            return extract_video_info(gen)

        gens = post.get("generations") or []
        if isinstance(gens, list) and gens:
            return extract_video_info(gens[0])

    if not video_url:
        logger.debug(f"No video URL found for post {info['id']}")
        return None

    info["video_url"] = video_url
    info["width"] = width
    info["height"] = height
    info["duration"] = duration
    info["raw_post"] = post

    return info


def fetch_post_detail(session: CffiSession, post_id: str) -> dict | None:
    """
    Fetch full details for a single post/generation if the feed only
    returned summary data without video URLs.
    """
    for base_path in (POST_DETAIL_API_PATH, GENERATION_DETAIL_API_PATH):
        url = f"{SORA_BASE_URL}{base_path}/{post_id}"
        logger.debug(f"Fetching detail: {url}")
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"Detail fetch failed for {url}: {e}")
            continue

    return None


def sanitize_filename(name: str, max_len: int = 80) -> str:
    """Make a string safe for use as a filename."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = name.strip('. ')
    if len(name) > max_len:
        name = name[:max_len].rstrip('. ')
    return name or "untitled"


def download_video(session: CffiSession, video_info: dict, output_dir: str) -> bool:
    """Download a single video file. Returns True if successful."""
    video_url = video_info["video_url"]
    vid_id = video_info["id"]
    title = sanitize_filename(video_info.get("title") or vid_id)

    ext = ".mp4"
    if ".webm" in video_url:
        ext = ".webm"
    elif ".webp" in video_url:
        ext = ".webp"

    filename = f"{title}_{vid_id}{ext}"
    filepath = os.path.join(output_dir, filename)

    if SKIP_EXISTING and os.path.exists(filepath):
        logger.info(f"  Skipping (exists): {filename}")
        return True

    logger.info(f"  Downloading: {filename}")
    logger.debug(f"  URL: {video_url}")

    try:
        # Video CDN URLs (videos.openai.com) generally don't need the same
        # Cloudflare bypass, but we use the same session for consistency.
        resp = session.get(video_url, timeout=DOWNLOAD_TIMEOUT)
        resp.raise_for_status()

        content = resp.content
        if not content:
            logger.warning(f"  Empty response for {vid_id}")
            return False

        with open(filepath, "wb") as f:
            f.write(content)

        size_mb = len(content) / (1024 * 1024)
        logger.info(f"  Saved: {filename} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        logger.error(f"  Download failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def save_metadata(video_info: dict, metadata_dir: str):
    """Save video metadata as a JSON file."""
    vid_id = video_info["id"]
    title = sanitize_filename(video_info.get("title") or vid_id)
    filename = f"{title}_{vid_id}.json"
    filepath = os.path.join(metadata_dir, filename)

    meta = {k: v for k, v in video_info.items() if k != "raw_post"}

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.debug(f"  Metadata saved: {filename}")


def scrape_explore_feed(session: CffiSession) -> list:
    """
    Iterate through the explore feed with cursor-based pagination,
    collecting video info across all pages.
    """
    all_videos = []
    cursor = None
    page_num = 0

    while True:
        page_num += 1
        logger.info(f"Fetching page {page_num}... (cursor={cursor!r})")

        try:
            data = fetch_explore_feed(session, cursor=cursor)
        except SystemExit:
            raise
        except Exception:
            logger.error("Failed to fetch feed page, stopping.")
            break

        posts, next_cursor = extract_posts_from_response(data)

        if not posts:
            logger.info("No more posts found, done.")
            break

        logger.info(f"  Got {len(posts)} posts on page {page_num}")

        for post in posts:
            video_info = extract_video_info(post)

            # If the feed returned summary-only data, try fetching full detail
            if video_info is None:
                post_id = (
                    post.get("id")
                    or post.get("post_id")
                    or post.get("generation_id")
                )
                if post_id:
                    logger.debug(f"  Fetching detail for {post_id}...")
                    detail = fetch_post_detail(session, post_id)
                    if detail:
                        video_info = extract_video_info(detail)
                    time.sleep(REQUEST_DELAY / 2)

            if video_info:
                all_videos.append(video_info)

            if MAX_VIDEOS and len(all_videos) >= MAX_VIDEOS:
                logger.info(f"Reached MAX_VIDEOS limit ({MAX_VIDEOS}).")
                return all_videos

        if not next_cursor:
            logger.info("No next cursor, reached end of feed.")
            break

        cursor = next_cursor
        time.sleep(REQUEST_DELAY)

    return all_videos


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Sora Video Scraper")
    logger.info("=" * 60)
    logger.info(f"Output directory:  {OUTPUT_DIR}")
    logger.info(f"Quality:           {DOWNLOAD_QUALITY}")
    logger.info(f"Max videos:        {MAX_VIDEOS or 'unlimited'}")
    logger.info(f"Feed type:         {FEED_TYPE}")

    # Check auth configuration
    has_auth = bool(BROWSER_NAME or AUTH_COOKIE or AUTH_BEARER)
    if not has_auth:
        logger.warning(
            "No authentication configured!\n"
            "Set BROWSER_NAME, AUTH_COOKIE, or AUTH_BEARER in the script.\n"
            "Without auth, Cloudflare will block API requests."
        )
    logger.info("")

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_METADATA:
        os.makedirs(METADATA_DIR, exist_ok=True)

    session = build_session()

    # Step 1: Scrape the explore feed
    logger.info("Step 1: Scraping explore feed...")
    videos = scrape_explore_feed(session)
    logger.info(f"Found {len(videos)} videos with downloadable URLs.")

    if not videos:
        logger.warning(
            "No videos found. This may mean:\n"
            "  1. Cloudflare is blocking requests (check cookie / BROWSER_NAME)\n"
            "  2. The API endpoints have changed (update EXPLORE_API_PATH etc.)\n"
            "  3. The feed is empty or rate-limited\n"
            "\n"
            "Debugging tip: set LOG_LEVEL = 'DEBUG' and check the response bodies.\n"
            "Also try opening DevTools on sora.chatgpt.com/explore to see the\n"
            "actual API paths your browser uses."
        )
        return

    # Step 2: Download each video
    logger.info(f"\nStep 2: Downloading {len(videos)} videos...")
    success_count = 0
    fail_count = 0

    for i, video_info in enumerate(videos, 1):
        logger.info(f"\n[{i}/{len(videos)}] {video_info.get('title', video_info['id'])}")

        if SAVE_METADATA:
            save_metadata(video_info, METADATA_DIR)

        ok = download_video(session, video_info, OUTPUT_DIR)
        if ok:
            success_count += 1
        else:
            fail_count += 1

        time.sleep(DOWNLOAD_DELAY)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Scraping complete!")
    logger.info(f"  Downloaded: {success_count}")
    logger.info(f"  Failed:     {fail_count}")
    logger.info(f"  Skipped:    {len(videos) - success_count - fail_count}")
    logger.info(f"  Output dir: {os.path.abspath(OUTPUT_DIR)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
