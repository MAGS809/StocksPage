from flask import Flask, request, jsonify, render_template_string
import alpaca_trade_api as tradeapi
import threading
import time

app = Flask(__name__)

# --- Global Bot State ---
bot_state = {
    "is_running": False,
    "logs": ["System initialized. Waiting for API credentials..."],
    "api_key": "",
    "secret_key": "",
    "asset": "BTC/USD",
    "in_position": False,
    "buy_price": 0.0
}

def log_msg(msg):
    timestamp = time.strftime('%H:%M:%S')
    bot_state["logs"].insert(0, f"[{timestamp}] {msg}")
    # Keep only the last 50 logs to prevent memory bloat
    if len(bot_state["logs"]) > 50:
        bot_state["logs"].pop()

# --- The Autonomous Trading Daemon ---
def trading_loop():
    log_msg("🤖 Autonomous Daemon Thread Started.")
    
    # Initialize Alpaca connection
    try:
        alpaca = tradeapi.REST(
            bot_state["api_key"], 
            bot_state["secret_key"], 
            "https://paper-api.alpaca.markets", 
            api_version='v2'
        )
        account = alpaca.get_account()
        log_msg(f"✅ Connected to Alpaca. Buying Power: ${account.buying_power}")
    except Exception as e:
        log_msg(f"❌ Connection Failed: {e}")
        bot_state["is_running"] = False
        return

    last_price = None
    buy_dip_pct = 0.005  # Buy on 0.5% drop
    take_profit_pct = 0.01  # Sell on 1% gain

    while bot_state["is_running"]:
        try:
            # Fetch current crypto price
            quote = alpaca.get_latest_crypto_trade(bot_state["asset"], exchange='CBSE')
            current_price = quote.price
            log_msg(f"{bot_state['asset']} Price: ${current_price:.2f}")

            if last_price is not None:
                # --- BUY LOGIC ---
                if not bot_state["in_position"]:
                    price_drop = (last_price - current_price) / last_price
                    if price_drop >= buy_dip_pct:
                        log_msg(f"📉 Sharp dip detected. EXECUTING BUY.")
                        buying_power = float(alpaca.get_account().cash) * 0.95
                        if buying_power > 10:
                            alpaca.submit_order(symbol=bot_state["asset"], notional=buying_power, side='buy', type='market', time_in_force='gtc')
                            bot_state["buy_price"] = current_price
                            bot_state["in_position"] = True
                            log_msg(f"✅ BOUGHT at roughly ${bot_state['buy_price']:.2f}")

                # --- SELL LOGIC ---
                elif bot_state["in_position"]:
                    profit_margin = (current_price - bot_state["buy_price"]) / bot_state["buy_price"]
                    if profit_margin >= take_profit_pct:
                        log_msg(f"📈 Profit target hit. EXECUTING SELL.")
                        positions = alpaca.list_positions()
                        for position in positions:
                            if position.symbol == bot_state["asset"]:
                                alpaca.submit_order(symbol=bot_state["asset"], qty=position.qty, side='sell', type='market', time_in_force='gtc')
                                bot_state["in_position"] = False
                                log_msg(f"💰 SOLD for a profit.")

            last_price = current_price

        except Exception as e:
            log_msg(f"⚠️ Loop Error: {e}")

        time.sleep(15) # Scan the market every 15 seconds

# --- Web Dashboard Routes ---
@app.route('/')
def index():
    # A sleek, dark-mode terminal UI built directly into the file
    html_dashboard = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>QuantBot Control Center</title>
        <style>
            body { background-color: #0d1117; color: #c9d1d9; font-family: 'Courier New', Courier, monospace; padding: 20px; }
            .container { max-width: 800px; margin: auto; }
            h1 { color: #58a6ff; }
            input, button { padding: 10px; margin: 5px 0; border-radius: 5px; border: 1px solid #30363d; background: #161b22; color: white; width: 100%; box-sizing: border-box;}
            button { background-color: #238636; cursor: pointer; font-weight: bold; }
            button:hover { background-color: #2ea043; }
            .btn-stop { background-color: #da3633; }
            .btn-stop:hover { background-color: #f85149; }
            #logs { background-color: #010409; padding: 15px; border: 1px solid #30363d; height: 300px; overflow-y: scroll; border-radius: 5px; }
            .log-line { border-bottom: 1px solid #21262d; padding: 5px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚡ QuantBot Control Center</h1>
            
            <label>Alpaca API Key:</label>
            <input type="password" id="api_key" placeholder="Paste Paper API Key here...">
            
            <label>Alpaca Secret Key:</label>
            <input type="password" id="secret_key" placeholder="Paste Paper Secret Key here...">
            
            <button onclick="startBot()">🚀 START AUTONOMOUS TRADING</button>
            <button class="btn-stop" onclick="stopBot()">🛑 STOP BOT</button>

            <h3>Live Terminal Logs</h3>
            <div id="logs"></div>
        </div>

        <script>
            function startBot() {
                const api = document.getElementById('api_key').value;
                const secret = document.getElementById('secret_key').value;
                fetch('/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({api_key: api, secret_key: secret})
                }).then(res => res.json()).then(data => alert(data.status));
            }

            function stopBot() {
                fetch('/stop', {method: 'POST'})
                .then(res => res.json()).then(data => alert(data.status));
            }

            // Fetch logs every 2 seconds
            setInterval(() => {
                fetch('/logs').then(res => res.json()).then(data => {
                    const logDiv = document.getElementById('logs');
                    logDiv.innerHTML = data.logs.map(l => `<div class="log-line">${l}</div>`).join('');
                });
            }, 2000);
        </script>
    </body>
    </html>
    """
    return render_template_string(html_dashboard)

@app.route('/start', methods=['POST'])
def start_bot():
    data = request.json
    if bot_state["is_running"]:
        return jsonify({"status": "Bot is already running!"})
    
    bot_state["api_key"] = data.get("api_key")
    bot_state["secret_key"] = data.get("secret_key")
    bot_state["is_running"] = True
    
    # Spin up the background thread
    thread = threading.Thread(target=trading_loop)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "Bot Started Successfully."})

@app.route('/stop', methods=['POST'])
def stop_bot():
    bot_state["is_running"] = False
    log_msg("🛑 Shutdown signal sent to daemon.")
    return jsonify({"status": "Bot stopping on next tick."})

@app.route('/logs', methods=['GET'])
def get_logs():
    return jsonify({"logs": bot_state["logs"]})

if __name__ == '__main__':
    # Runs the web app on port 8080 (Standard for Replit)
    app.run(host='0.0.0.0', port=8080)
