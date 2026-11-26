# Windows Setup — Paradox Trading Terminal

## Sach (honest answer): kya apne aap connect ho jayega?

| Feature | Windows pe kya hoga |
|---|---|
| Backtesting (bar + per-tick audit) | ✅ **Bilkul apne aap** — data files copy karo, bas |
| AI Strategy Builder (Gemini) | ✅ Apne aap — `.env` me key daal do |
| Paper trading engine | ✅ Apne aap |
| Prop-firm rules, Strategy Lab, Time Machine | ✅ Apne aap |
| **Live MT5 data + real execution** | ⚠️ **Windows pe HI chalega** — yahi asli faida hai (Mac pe Wine-jugaad tha) |
| Tick archive (207M ticks) | ✅ Portable — `data/ticks/` folder copy karna hai |

## Steps (one-time)

1. **Python 3.12+** install karo — [python.org](https://www.python.org/downloads/) se, aur installer me **"Add Python to PATH"** tick karo.
2. Poora `prototype/` folder Windows pe copy karo — **`data/` folder zaroori hai** (1.2 GB, isme tick archive hai). `.venv` copy MAT karo — wo Mac ka hai, script naya banayega.
3. `prototype/` folder me double-click: **`setup_windows.bat`** — venv banata hai, sab dependencies install karta hai, `.env` template banata hai.
4. Start: **`start_windows.bat`** → browser me `http://127.0.0.1:8799`
   - Password: `.env` me `DSH_ADMIN_PASSWORD` set karo (default script me `test-pass-1234`).

## Live MT5 (Windows ka asli bonus)

Mac pe MT5 official Python package nahi chalta (Wine jugaad tha). Windows pe:

1. Apne broker (Exness/IC Markets/...) ka **MetaTrader 5 terminal** install karo aur login karo.
2. `.venv\Scripts\pip install MetaTrader5`
3. `prototype/.env` me bharo:
   ```
   MT5_LOGIN=tera_account_number
   MT5_PASSWORD=tera_password
   MT5_SERVER=Broker-Server-name   (MT5 terminal me right-bottom corner pe dikhta hai)
   ```
4. Platform restart karo. Data manager ka chain hai: `MT5Broker → MacMT5Broker → PAXG → Yahoo → synthetic`.
   MT5 available hoga to **wo primary ban jayega** — live XAUUSD data + real order execution dono.

Agar MT5 setup nahi karna: platform phir bhi chalta hai — cached CSV data + PAXG/Yahoo live prices se. Sirf real execution nahi hogi (paper mode me sab chalega).

## Kya copy karna hai / kya nahi

```
prototype/
├── data/            ✅ COPY (1.2 GB — ticks + CSV cache, ye sab kuch hai)
├── core/ strategies/ dashboard/ ai/ tests/   ✅ COPY (code)
├── results/ state/  ✅ COPY (runs, configs)
├── .env             ✅ COPY (Gemini key isme hai — gitignored, safe)
├── .venv/           ❌ MAT copy — setup_windows.bat naya banayega
├── __pycache__/     ❌ zaroorat nahi
└── setup_windows.bat, start_windows.bat   ✅ COPY (ye hi sab kuch set karte hain)
```

## Verification (copy ke baad)

```
cd prototype
.venv\Scripts\python -m pytest tests\ -q     →  879 passed aana chahiye
start_windows.bat                            →  browser: http://127.0.0.1:8799
```

Admin panel (`/legacy`) → System page me data source ka status dikhega: `mt5` (live) ya `mt5_mac` (cached CSV) ya `paxg`/`yahoo` (live fallback). Jo bhi dikhe, wo sach hai — platform kabhi fake data ko real nahi batata.
