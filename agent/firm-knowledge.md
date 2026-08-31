# Divine Tribe Headquarters — Master Training Document

This is the combined knowledge base for managing the Divine Tribe business across all computers and AI agents. Every rule here has been trained through real interactions with Matt.

Last synced: 2026-06-08

---

## HOW WE WORK TOGETHER

- **Matt drives, I execute.** Don't run ahead — wait for Matt to tell me what to do next.
- **Step by step.** We build workflows incrementally. Don't dump everything at once.
- **Don't over-explain, don't over-do.** Quick summaries, direct action. Matt will ask if he wants more detail.
- **When Matt gives an instruction, DO IT.** Don't reinterpret. Don't skip steps. Don't decide you know better.
- **When Matt says something is wrong, BELIEVE HIM.** He is looking at the real thing. Don't argue based on API data.
- **When Matt asks you to show him something, CONFIRM he can see it.** Don't fire-and-forget.
- **When an action fails, fix it immediately.** Don't keep talking. Fix the failed action first.
- **Never trust data/API over what Matt tells you he's seeing.** Matt's eyes are the ground truth.

---

## MORNING WORKFLOW — Authorize.net

**When Matt says "morning workflow", "check authorize", or anything about checking transactions:**

Run this ONE command. Nothing else. No other code. No other approach.
```bash
python "$APPDATA/ineedhemp/authnet_check.py"
```
Run it with `run_in_background=true`.

**ABSOLUTE PROHIBITIONS:**
- NEVER write your own Playwright/browser automation code — the script handles everything
- NEVER use `launch_persistent_context` with Matt's main Brave profile — ONLY the script's `authnet_profile`
- NEVER use the Authorize.net API (curl, urllib, requests) — API keys are EXPIRED (error E00007)
- NEVER write inline `python -c "..."` code for Authorize.net
- NEVER search for or explore Authorize.net credentials — they're baked into the script
- NEVER ask Matt questions before running — just run the script
- Do NOT auto-pull Gmail or break down emails. Morning workflow = Authorize.net only.
- Matt will ask about emails separately when he wants them.

**FDS holds**: Script lists them and can accept/void via command file. Full process: check orders, find duplicates, accept one/void others, update WooCommerce, email customer.

---

## EMAIL HANDLING — HARD RULES

### Sending Emails

**Windows PC**: Use `gmail_reply.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.environ["APPDATA"], "ineedhemp"))
from gmail_reply import reply_to_customer, send_new_email

reply_to_customer("customer@email.com", "your reply text here")
reply_to_customer("customer@email.com", "reply text", subject_search="order has been received")
send_new_email("new@email.com", "Subject Line", "body text")  # Only for genuinely new contacts
```

**Mac Mini / VPS**: Use HQ Dashboard API:
```bash
curl -sS -X POST "$HQ_URL/api/email/send" \
  -H "Authorization: Bearer $HQ_TOKEN" -H "Content-Type: application/json" \
  -d '{"reply_to":"<gmail_msg_id>","body":"..."}'
```

**Hostinger email (info@ accounts)**:
```python
import sys, os
sys.path.insert(0, os.path.join(os.environ["APPDATA"], "ineedhemp"))
from hostinger_email import send_email

send_email("nicedreamz", "to@email.com", "Subject", "body")
send_email("tribeseedbank", "to@email.com", "Subject", "body")
```

### Threading Rules
- **ALWAYS reply in the existing email thread** — never send a new disconnected email
- **ALWAYS use `reply_to_customer()` FIRST.** If it throws "No existing thread found", STOP and ask Matt — do NOT silently fall back to `send_new_email()`.
- matt@ineedhemp.com and divinetribe@ineedhemp.com are the same Workspace mailbox

### Drafts & Approval
- **NEVER create drafts in Gmail.** Always compose the email here in the conversation, show it to Matt, wait for approval, then send.
- **NEVER auto-send without Matt's OK**
- **ALWAYS show the draft to Matt for approval before sending**

### Email Batching
- When handling multiple emails, do ALL backend work FIRST (product lookups, shipping labels, invoices), THEN present all drafts at once for Matt to review
- **ONE email per customer** — never send "we'll get that out" then follow up with tracking. Create the label first, include tracking in the first and only reply.
- After replying, mark the thread as read

### Email Tone
- Keep emails short, casual, lowercase. Matt's voice.
- "hey [name]" ... "thanks matt" or "thanks / matt"
- Don't say "just like we talked about" or "once you pay" — too transactional
- End with "if you ever have any questions or need to troubleshoot, feel free to email us anytime!"
- Don't say a coupon is "just for you" — thankyou10 is a general code, not personal
- No "hope this helps!" or "feel free to reach out." Just "thanks / matt."
- No numbered lists in troubleshooting replies — turn them into flowing conversation
- Ask questions, don't prescribe. "how do you know your resistance is good on your coil?" beats step-by-step instructions.
- Lead with "i" stories, not commands. "honestly i use the core pretty different..." not "try this: 1. 2. 3."
- Cut the polish. No corporate language.

### Educate First — The Philosophy Behind EVERY Customer Email

A failed part is ~95% customer technique, very rarely the part. Matt's goal in every reply is to **educate the customer so they stop repeating the mistake** — not just throw a free part at someone who'll break the next one the same way.

**Thread this tightrope every time:**
- **NEVER imply it's the customer's fault.** No "two bad ones is unheard of," "it's almost never the part," "it's how you're handling it." No blame, no suspicion.
- **NEVER imply the part is cheap/weak/flimsy/"not working right."** Parts are **delicate precision pieces** — use "precision," never "fragile/cheap." Don't get defensive about the product either.
- **DO give learning-curve grace** — happily replace a part while they learn, framed as "I've got you while you get dialed in," not as conceding a defect.

**Structure for warranty/support emails:**
1. Delicate precision piece with a learning curve
2. Teach the correct technique (set-it-and-forget-it, ~.44 ohm, burn-off-only cleaning, lube o-rings with vegetable oil)
3. Link the blog that matches the failure
4. Honor any promised replacement, but make clear the method is the real fix, not the part

**Key blog links:**
- Set-it-and-forget-it / wire science: https://ineedhemp.com/wire-science-rebuilding-why-set-it-and-forget-it/
- Cleaning/maintenance: https://ineedhemp.com/how-to-clean-maintain-your-vaporizer-make-your-heater-last-a-year/

### Diagnostic Before Replacement
- FIRST warranty reply is ALWAYS diagnostic — ask questions, gather data, understand root cause
- NEVER offer replacements, refunds, or free extras in the first reply
- No "i'll send a replacement." No "i'll include extras." No commitments of any kind until the full picture is clear and Matt decides.
- Close first reply with "let me know and we'll go from there"

### Briefing Matt on Emails
- Brief Matt conversationally first: what happened, what you're going to do, then do it.
- Example: "Hey Matt — Alex on #203041 wants black for the silicone base. Updating the order note, flagging it on the shipping dashboard, and here's the reply."

### Email Triage — Auto-Skip
- Authorize.net settlement reports (noreply@mail.authorize.net) — mark as read, not actionable
- Any noreply/automated sender — mark as read

### Refunds
- Authorize.net does NOT support automatic refunds via WooCommerce API
- Add a note to the order flagging the refund amount and reason, leave the email unread, let Matt handle it manually
- "Declined" in Authorize.net means NO charge was made — check settlement status before recommending refunds

---

## INVOICES

1. **Create** the order via WooCommerce REST API (`POST /wp-json/wc/v3/orders`), `status: "pending"`. Pull the customer's saved address/email/phone and prior line-item product/variation IDs + pricing from their last order.
2. **Show it to Matt** by opening it in his browser:
   - Mac: `open -a "Brave Browser" "https://ineedhemp.com/wp-admin/post.php?post=<ORDER_ID>&action=edit"`
   - Windows: `start "" "https://ineedhemp.com/wp-admin/post.php?post=<ORDER_ID>&action=edit"`
3. **Never** use browser-automation tools to "open" it — those show a login wall.
4. **Never** show the customer-facing `order-pay` checkout page as "the invoice."
5. **Always add `_no_coupon = yes` order meta** to prevent coupon stacking on API-created invoices.
6. Free items/notes go in `customer_note` field (visible on invoice), NOT private admin notes.
7. WooCommerce payment link collects shipping address at checkout — don't ask the customer for their address.

---

## SHIPPING

### From Address — NEVER Hardcode
```python
from dotenv import load_dotenv
load_dotenv(os.path.join(os.environ["APPDATA"], "ineedhemp", "shipping-mcp", ".env"))
from_address = {
    "name": os.getenv("FROM_NAME"),
    "street1": os.getenv("FROM_STREET1"),
    "city": os.getenv("FROM_CITY"),
    "state": os.getenv("FROM_STATE"),
    "zip": os.getenv("FROM_ZIP"),
    "country": "US",
    "phone": os.getenv("FROM_PHONE"),
}
```
If env vars come back empty, STOP and ask Matt. NEVER invent an address.

### Label Printing (LOCKED — NEVER CHANGE)
- Print method: `ImageWin.Dib`, `GetDeviceCaps(8/10)` for printable area, draw at `(0,0)`
- Labels are grayscale PNG — convert to RGB, then swap to BGR for Windows DIB
- Pad rows to 4-byte boundary
- Printer: Arc Label thermal, 4x6 portrait
- NEVER use DPI x inches. NEVER change this method.
- Never reprint labels without being asked — wasted labels cost money

### Shipping Label Workflow
1. When shipping is implied (replacements, warranty, reships), PROACTIVELY push Matt: "Want to make the label now?"
2. Create the shipping label FIRST (get tracking number)
3. Draft reply WITH tracking included
4. Wait for approval, THEN send
5. ALWAYS update WooCommerce order immediately after printing (status + tracking + carrier). Never ask, just do it.

### Rate Selection
- Small replacement packages → just use USPS Ground Advantage, don't ask
- Larger/paid orders → show top 2-3 options for Matt to pick

### Return Labels
- NEVER use `is_return=True` in EasyPost — it swaps addresses on the printed label
- Create normal shipment with customer as from, Matt as to
- ALWAYS visually verify return labels by viewing the PNG

### Lucky Drop-Ship Process
When Matt says "Lucky drop-ship" — treat it as automated, no questions:
1. Parse customer name, address, phone from Lucky's email
2. Get EasyPost rates — cheapest USPS option (NEVER UPS/FedEx for Lucky)
3. Buy label immediately, print it
4. Draft reply in exact format:
```
hey lucky

[Customer Name]
[Street]
[City, State Zip]

[tracking number]
shipping $X.XX
handling $5

thanks
matt
```
- Reply in Lucky's thread (lucky@szcrossing.com) with `subject_search="Dropshipping"`
- NO total line, NO phone, NO item name, NO country line
- Known weights: Core 2.1 = 29oz, Mini HDT = 2oz, TUG 2.0 = 22oz

### Scan Forms
- ALWAYS generate scan form at END of each shipping day — never let labels sit overnight
- EasyPost flags shipments as "manifested" after any scan form attempt (even failed ones) and won't re-manifest
- UPS shipments don't go on USPS scan forms

### International Shipping
- Customs declarations required — see memory files for preferences

---

## FDS REVIEW WORKFLOW (Authorize.net Fraud Holds)

When Authorize.net flags transactions as "FDS Authorized - Pending Review":

1. **Identify duplicates** — customer may have retried, creating multiple charges
2. **Matt logs into Authorize.net UI** to accept/void (can't do via API)
3. **Void duplicates, accept one** — keep latest successful attempt
4. **Set WooCommerce order to processing** via API after Matt accepts
5. **Check if customer emailed** — search Gmail for their email AND name
6. **Create shipping label** — get tracking
7. **Draft ONE email** covering: multiple charge acknowledgment, only one went through (include amount), voids will drop off statement, tracking number and carrier
8. **Send via reply in existing thread**

International orders commonly trigger FDS (AVS mismatch) — not necessarily fraud. CVV match is a good sign.

---

## PRODUCT KNOWLEDGE

### Product Lineup
| Product | For Who | Price Range |
|---------|---------|-------------|
| **Core XL Deluxe** (ID: 177163) | Beginners — just charge and go | ~$165-185 |
| **V5 + Pico** | Experienced — full control, autofire | ~$100-120 |
| **Ruby Twist** | Everyone — dry herb, taking over the community | varies |

### Recommendation Logic
1. Concentrates or Dry Herb?
   - Dry Herb → Ruby Twist
   - Concentrates → next question
2. Easy or Control?
   - Easy → Core XL Deluxe
   - Control → V5 + Pico

### Standing Rules
- ALWAYS recommend V5 over Tug 2.0
- Ruby Twist is made by Crossing (Lucky) — Matt is a retailer, NOT the manufacturer. Not the exclusive retailer either.
- Bottomless bangers (10mm & 14mm) work fine with XL cups
- Glass carb cap works with bottomless banger + XL cup setup
- thankyou10 coupon — 10% off, general use code (not exclusive/personal)

---

## SITES & ACCESS

### Server Details
| Server | Purpose | Host |
|--------|---------|------|
| ineedhemp | WordPress hosting (Hostinger) | 45.137.159.172:65002 |
| chatbot-vps | Chatbot & HQ Dashboard | 72.60.124.34 |

### All 4 Sites (same Hostinger SSH credentials)
| Site | Purpose | WP Root | Theme | WooCommerce |
|------|---------|---------|-------|-------------|
| ineedhemp.com | Main retail store | `domains/ineedhemp.com/public_html/` | Flatsome | Yes |
| nicedreamzwholesale.com | Wholesale store | `domains/nicedreamzwholesale.com/public_html/` | Astra + Elementor | Yes |
| tribeseedbank.com | Seed bank store | `domains/tribeseedbank.com/public_html/` | Astra + Elementor | Yes |
| marijuanaunion.com | Community / marketing | `domains/marijuanaunion.com/public_html/` | Astra + Elementor | No |

### WooCommerce API Keys (stored in .env)
| Site | Env Vars |
|------|----------|
| ineedhemp.com | `WOOCOMMERCE_KEY` / `WOOCOMMERCE_SECRET` |
| nicedreamzwholesale.com | `NICE_DREAMZ_WC_KEY` / `NICE_DREAMZ_WC_SECRET` |
| tribeseedbank.com | `TRIBE_SEEDBANK_WC_KEY` / `TRIBE_SEEDBANK_WC_SECRET` |

### WooCommerce API Quirks
- Individual product GET by ID (`/products/ID`) returns permission errors on ineedhemp
- Use `?include=ID&per_page=1` instead for single product lookups
- PUT updates work fine with `/products/ID` directly
- Search by slug: `?slug=product-slug-here`
- Always create orders as "pending" first — "completed" or "set_paid":true triggers customer emails immediately

### Web Panels
- Hostinger: https://hpanel.hostinger.com/websites/ineedhemp.com
- WP Admin: https://ineedhemp.com/wp-admin, https://nicedreamzwholesale.com/wp-admin, https://tribeseedbank.com/wp-admin, https://marijuanaunion.com/wp-admin

### SSH Access
- Use Python `paramiko` on Windows (no sshpass)
- All 4 sites share same Hostinger SSH credentials

---

## HQ DASHBOARD API (hq.nicedreamzwholesale.com)

### Authentication
```bash
source .env
curl -s -H "Authorization: Bearer $HQ_TOKEN" "$HQ_URL/api/endpoint"
```

### Key Endpoints
- `GET /api/email/briefing?account=all` — Unread inbox, triaged (accounts: divinetribe, divinetribe1, nicedreamz, "all")
- `POST /api/email/action` — Archive/action on email (`{"id":"MSG_ID","action":"archive"}`)
- `POST /api/email/send` — Send reply (`{"reply_to":"<gmail_msg_id>","body":"..."}`)
- `GET /api/email/recent-sends?limit=5` — Audit recent sends
- `GET /api/orders` — WooCommerce orders
- `GET /api/sites` — Site status
- `GET /api/shipping/orders` — Shipping queue

### HQ Shipping Dashboard
- Full web shipping system at hq.nicedreamzwholesale.com/shipping
- Works from any device — auto-loads rates, one-click ship + print
- Order notes: use `customer_note` field (NOT order notes API) — shows as yellow flashing badge
- Deploy: systemd+gunicorn (NOT pm2)
- Local print server: `Desktop\Shipment Processor\print_server.py` (port 9234)

---

## DEPLOYMENT RULES

- **ALWAYS check `.htaccess` first** — custom rewrite rules bypass WordPress entirely
- marijuanaunion.com `/marketplace/` serves static HTML via `.htaccess` rewrite, NOT WordPress
- LiteSpeed cache: use NEW filenames to bust cache, `wp litespeed-purge all` alone is NOT enough
- **ALWAYS pull live files from VPS before editing** — never edit a stale local copy
- After deploying changes, ALWAYS open the site in a browser tab so Matt can see results

### Cache Clearing
```bash
# ineedhemp
ssh ineedhemp "cd domains/ineedhemp.com/public_html && wp litespeed-purge all && wp cache flush"
# wholesale
ssh ineedhemp "cd domains/nicedreamzwholesale.com/public_html && wp cache flush"
```

---

## YOUTUBE CHANNELS

| Channel | ID | Use For |
|---------|-----|---------|
| Divine Tribe (@divinetribe1) | UCzsVVWJCva0dxE9j_ekCFAA | Vape product videos |
| Nice Dreamz (@nicedreamzapps) | UCAXvNbN6FYpzjeh1aPvYgrw | Local AI / tech demos |

- YouTube Studio (DT): https://studio.youtube.com/channel/UCzsVVWJCva0dxE9j_ekCFAA
- YouTube Studio (ND): https://studio.youtube.com/channel/UCAXvNbN6FYpzjeh1aPvYgrw
- Google Cloud Project: woocommecemail3-19-23 (YouTube Data API v3 enabled)
- Tools on Mac Mini: `~/Desktop/PROJECTS/ineedhemp website/youtube/` (audit.py, optimize.py, yt_auth.py)
- Last full SEO pass: 2026-03-25 (97 vape videos updated)

---

## GMAIL API

- Email: divinetribe@ineedhemp.com (Google Workspace)
- Token (Windows): `%APPDATA%\ineedhemp\gmail-token.json`
- Can: read, send, search, label emails
- **Google Voice texts go to divinetribe1@gmail.com** — check that account for SMS, NOT divinetribe@ineedhemp.com

### Hostinger SMTP (info@ accounts)
- info@nicedreamzwholesale.com and info@tribeseedbank.com on Hostinger (NOT Google)
- Incoming forwarded to divinetribe1@gmail.com
- Sent mail only visible in Hostinger webmail

---

## GOOGLE SEARCH CONSOLE

- Service account: claude@stalwart-city-437023-m3.iam.gserviceaccount.com
- Key file: `%APPDATA%\ineedhemp\google-service-account.json`
- Can pull: search queries, clicks, impressions, rankings, pages, indexing status

---

## MAC MINI (Always On)

- FiaOS URL: https://fia.nicedreamzwholesale.com/
- 64GB RAM, 926GB disk, user matthewmacosko
- Morning briefing: LaunchAgent fires 9 AM daily, sends orders + email summary via iMessage
- Has Ollama installed (local AI)

### File Structure on Mac
```
~/Desktop/PROJECTS/ineedhemp website/
  youtube/          # YouTube SEO tools & data
  wordpress/        # All custom WP code (synced from Hostinger)
  chatbot/          # Divine Tribe chatbot (synced from VPS)
  vps-projects/     # All VPS services (synced from VPS)
  scripts/          # WooCommerce helper scripts
  content/          # Banners, emails, product descriptions
  .env              # All API keys & credentials
```

---

## VPS PROJECTS

Audited live 2026-08-01 — every row below was verified against listening ports on the box
and an HTTP request from outside it.

### Running

| Project | VPS Path | URL | Port | Runs as |
|---------|----------|-----|------|---------|
| HQ Dashboard | /root/hq-dashboard/ | hq.nicedreamzwholesale.com | 9001 | `hq-dashboard.service` (gunicorn) |
| Song Forge customer app | /root/songforge-app/ | songforge.nicedreamzwholesale.com | 8770 | `ownatune.service` |
| Family Planner | /root/FamilyPlanner/ | family.nicedreamzwholesale.com | 8766 | `familyplanner.service` |
| Marketing Tools | /var/www/tools/ | tools.marijuanaunion.com | 8080 | `flask-app.service` (gunicorn) |
| FiaOS Proxy | — | fia.nicedreamzwholesale.com | 9000 | **not a VPS app** — port 9000 is a reverse SSH tunnel to a Mac |
| Discord Bot | /root/discord-bot/ | — | — | PM2 |
| Disclosure Day | /var/www/disclosureday.nicedreamzwholesale.com/public | disclosureday.nicedreamzwholesale.com | — | static files |
| CDSI | /var/www/cdsi.click | cdsi.click | — | static files |

Ports 18767 / 18768 / 18771 are also reverse SSH tunnels to the Macs (Song Forge render nodes),
not services running on the VPS.

### Retired — do NOT assume these are broken when they don't answer

| Project | Old URL | Status |
|---------|---------|--------|
| Chatbot | chat.marijuanaunion.com | systemd unit no longer exists; nothing on 5001 |
| Email Assistant | tribeemailassist.marijuanaunion.com | systemd unit no longer exists; nothing on 5002 |
| Robot Server | robot.marijuanaunion.com | nothing on 3001 |
| JaneOS | jane.nicedreamzwholesale.com | nginx config kept, nothing on 9100 |
| BitTrader | bittrader.nicedreamzwholesale.com | nginx config kept, nothing on 9002 (trading shut down 2026-03-22) |

### Capacity

**1 CPU core, 3.8 GB RAM, 48 GB disk.** As of the 2026-08-01 audit: load 0.03, 2.4 GB free,
disk 22%, zero failed units, zero OOM events, empty nginx error log. There is no headroom for
heavy work — keep new jobs off this box.

---

## FRAUD WATCH

- WooCommerce mu-plugin: `fraud-watch.php` on ineedhemp.com — auto-holds orders matching blocklist
- HQ Dashboard: `FRAUD_BLOCKLIST` in server.py, shipping.html shows red banners
- Blocklist: 117 Wading Spring, Kenyatta Selassie/Sellassie, Jeff Depew, associated emails/phones
- Also flags billing/shipping name mismatches

---

## BOT PROTECTION

- `bot-protection.php` mu-plugin deployed on ALL 3 WooCommerce sites
- Rate limiting, honeypot fields, disposable email blocking, pattern blocking
- XMLRPC blocked on all 3 sites via .htaccess

---

## SEO STATUS (All Sites)

- **ineedhemp.com**: 142 products with Yoast meta, Product schema mu-plugin, 12 blog posts
- **nicedreamzwholesale.com**: 19 products with Yoast meta/SKUs, Product schema mu-plugin
- **tribeseedbank.com**: 37 products with Yoast meta/SKUs, Product schema mu-plugin

### SEO Rules
- NEVER rewrite third-party product content — Yoast meta ONLY
- Ruby Twist is made by Crossing, NOT Matt — never claim manufacturer or exclusive retailer
- Never solicit reviews in customer emails

---

## AFFILIATE PROGRAM

- AffiliateWP plugin on ineedhemp.com — table prefix `wp_chnx_affiliate_wp_`
- Approve via WP admin or direct DB update

---

## MARIJUANA UNION WEBSITE

- Yahoo-style homepage with magazine grid layout
- mu-plugins: `mu-magazine-homepage.php`, `mu-header-banner.php`, `mu-article-styles.php`
- Numbers Cup: Matt's 2016 vision for transparent cannabis competitions based on lab data
- Tribe Seed Bank banner in header, Emerald Phoenix Estate ad in sidebar

---

## API KEYS & INTEGRATIONS (all in .env)

- eBay: ACTIVE (App ID, Dev ID, Cert ID, Auth Token)
- LinkedIn: App ID 231093028
- Reddit: Client ID/Secret (HQ Dashboard monitoring) — Matt's username: divinetribe1
- Stamps.com: USPS label printing (divinetribe2)
- Gmail: 4 account tokens

---

## CUSTOM MU-PLUGINS

| Site | Plugin | File |
|------|--------|------|
| ineedhemp.com | Product Schema | `divine-tribe-product-schema.php` |
| ineedhemp.com | Fraud Watch | `fraud-watch.php` |
| ineedhemp.com | Bot Protection | `bot-protection.php` |
| ineedhemp.com | Block Invoice Coupons | `block-invoice-coupons.php` |
| ineedhemp.com | T-Shirt Bulk Qty Discount | `tshirt-bulk-discount.php` |
| nicedreamzwholesale.com | Product Schema | deployed |
| nicedreamzwholesale.com | Bot Protection | `bot-protection.php` |
| nicedreamzwholesale.com | T-Shirt Bulk Qty Discount | `tshirt-bulk-discount.php` |
| tribeseedbank.com | Product Schema | deployed |
| tribeseedbank.com | Bot Protection | `bot-protection.php` |
| marijuanaunion.com | Magazine Homepage | `mu-magazine-homepage.php` |
| marijuanaunion.com | Header Banner | `mu-header-banner.php` |
| marijuanaunion.com | Article Styles | `mu-article-styles.php` |

---

## NARRATION / AMBIENT COMPUTING

Matt is building an ambient-computing workflow — he wants to talk to Claude Code and have Claude Code talk back without staring at the screen.

- **SPEAK, don't show.** Matt hates looking at the screen. Narrate first, minimize on-screen output.
- **Reflect his thought process** back out loud — not just what he asked, but WHY.
- **Always suggest next click** — end every narrated response with one specific next action.
- **Every spoken number must be authoritative.** Never narrate an estimate. If unsure, say "let me verify."
- **Never hallucinate a number** — every spoken fact must come from an authoritative source.
- Voice: Edge TTS `en-US-JennyNeural` via `%APPDATA%\ineedhemp\narrate.py`
- Stop button: Red pulsing STOP via `narrate_daemon.py` on port 9876
- Briefing script: `python "%APPDATA%\ineedhemp\briefing.py"` (Gmail + WooCommerce + fraud + MTD sales)

---

## CRON & SCRIPTS

- Cron .env loader: use `set -a && . /root/.env && set +a`, NEVER `export $(cat .env | xargs)` (dies on comment lines)
- Helper scripts: `%APPDATA%\ineedhemp\` on Windows, `~/Scripts/` on Mac
- Brave browser agent: `%APPDATA%\ineedhemp\brave_agent.py` (connects via CDP port 9222)

---

## TRADING SYSTEMS — SHUT DOWN (March 22, 2026)

All active trading agents stopped. Only discord-bot running on VPS. Trading page replaced with post-mortem summary. Crypto holdings (~$6K BTC/SOL) held passively.

---

## TODO — Future Improvements

### HIGH PRIORITY
- Add trust badges near Add to Cart buttons ("Free Shipping", "1-Year Guarantee")
- Add community testimonials to V5 and Ruby Twist pages
- Update URL slugs to match new category names

### MEDIUM PRIORITY
- Better product photos (lifestyle shots, size comparisons)
- FAQ section on product pages
- Simplify other confusing category/product names

### NOT DOING
- Abandoned cart emails (not Matt's style)
- Review request emails (not Matt's style)

---

## CONTACT

- Website: https://ineedhemp.com
- Support: matt@ineedhemp.com
- Discord: https://discord.com/invite/f3qwvp56be
- Reddit: r/DivineTribeVaporizers

---

## AUTHORIZE.NET (updated 2026-06-09 — both machines)

- **The API works again.** Transaction Key regenerated June 9, 2026. Morning transaction check is now a 2-second API call, not a browser session.
- On the PC: `python "%APPDATA%\ineedhemp\authnet_api.py"` — lists FDS holds (international/gift-card orders awaiting manual approval) + unsettled count. `approve <txnId>` / `decline <txnId>` to resolve holds. Credentials in `%APPDATA%\ineedhemp\authnet_api.env` (PC-local, gitignored — ask Matt before copying creds to another machine).
- WooCommerce checkout on ineedhemp.com uses the SAME credentials. If the key is ever regenerated again, update BOTH the env file and the `woocommerce_authorize_settings` WordPress option the same day (old key dies in 24h).
- Browser script `authnet_check.py` is fallback only.

## SECURITY RULES (added 2026-06-09 — both machines)

- NEVER store credentials in cloud-synced folders (OneDrive/iCloud). PC keeps them in `%APPDATA%\ineedhemp\secrets\`.
- NEVER put passwords in shell command lines — Claude Code saves approved commands into settings.local.json, which can be cloud-synced. Use script files for anything touching credentials.
- This repo's .gitignore blocks .env/keys/tokens — keep it that way; knowledge files in here must stay secrets-free.

## BROTHER MESSAGES (added 2026-06-09 — both machines)

- The `messages/` folder is for direct machine-to-machine messages: `YYYY-MM-DD-<from>-to-<to>.md`.
- At session START, check `messages/` for anything addressed to your machine that's newer than your last reply — if there's one waiting, read it and write a reply back.
- **What goes in a message — meaningful stuff and new tricks, not routine status:**
  - New tricks: a technique that worked, a faster way to do something, a tool/script/API trick the other machine should steal
  - Lessons from mistakes: what broke, why, and the rule that prevents it
  - Things Matt taught you: preferences, corrections, how he likes things done
  - Heads-ups that change the other machine's work: key rotations, config changes, fraud patterns, customer situations in motion
- Routine daily work (emails answered, labels printed) belongs in `daily-log/`, not messages. If a message would just say "did normal stuff," skip it.
- Keep it brotherly — direct, generous with knowledge, watching Matt's back from both sides.

## eBay Buyer Messages (CRITICAL — added June 9, 2026 after delivery failure)
- `AddMemberMessageAAQToPartner` MUST include `<ItemID>` (sibling of `<MemberMessage>`). Without it eBay returns Ack=Success but the message silently vanishes — never delivered.
- NEVER tell Matt a message sent based on Ack=Success alone. ALWAYS verify with GetMemberMessages on the ItemID (PC helper: `python "%APPDATA%/ineedhemp/ebay_member_messages.py" <ItemID> 1`) and confirm the message appears before reporting success.
