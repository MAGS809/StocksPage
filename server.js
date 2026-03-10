require("dotenv").config();
const express = require("express");
const http = require("http");
const WebSocket = require("ws");
const cors = require("cors");
const Alpaca = require("@alpacahq/alpaca-trade-api");
const Anthropic = require("@anthropic-ai/sdk").default;

const app = express();
app.use(cors());
app.use(express.json());

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const alpaca = new Alpaca({
  keyId: process.env.APCA_API_KEY_ID,
  secretKey: process.env.APCA_API_SECRET_KEY,
  paper: true,
});

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const TICKERS = ["SPY", "QQQ"];
const MOVE_THRESHOLD = 0.002;

function broadcastToClients(data) {
  const message = JSON.stringify(data);
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
}

async function analyzeWithAI(ticker, open, close) {
  try {
    const prompt = `You are a risk-averse day trader managing a strict $100 foundation account. ${ticker} just moved from ${open} to ${close} in one minute. Is this a breakout or a trap? Give a 2-sentence analysis.`;

    const response = await anthropic.messages.create({
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 256,
      messages: [{ role: "user", content: prompt }],
    });

    const message = response.content[0].text;

    const alert = {
      type: "AI_ALERT",
      ticker,
      price: close,
      open,
      close,
      message,
      timestamp: new Date().toISOString(),
    };

    console.log(`[AI ALERT] ${ticker}: ${message}`);
    broadcastToClients(alert);
  } catch (error) {
    console.error("[AI ERROR]", error.message);
  }
}

function connectAlpacaStream() {
  const alpacaWs = new WebSocket("wss://stream.data.alpaca.markets/v2/iex");

  alpacaWs.on("open", () => {
    console.log("[ALPACA STREAM] Connected to IEX feed");

    const authMsg = {
      action: "auth",
      key: process.env.APCA_API_KEY_ID,
      secret: process.env.APCA_API_SECRET_KEY,
    };
    alpacaWs.send(JSON.stringify(authMsg));
  });

  alpacaWs.on("message", (data) => {
    const messages = JSON.parse(data.toString());

    for (const msg of messages) {
      if (msg.T === "success" && msg.msg === "authenticated") {
        console.log("[ALPACA STREAM] Authenticated successfully");

        const subMsg = {
          action: "subscribe",
          bars: TICKERS,
        };
        alpacaWs.send(JSON.stringify(subMsg));
        console.log(`[ALPACA STREAM] Subscribed to bars: ${TICKERS.join(", ")}`);
      }

      if (msg.T === "b") {
        const ticker = msg.S;
        const open = msg.o;
        const close = msg.c;
        const move = Math.abs((close - open) / open);

        console.log(`[BAR] ${ticker} O:${open} C:${close} Move:${(move * 100).toFixed(3)}%`);

        broadcastToClients({
          type: "BAR_UPDATE",
          ticker,
          open,
          close,
          high: msg.h,
          low: msg.l,
          volume: msg.v,
          move: (move * 100).toFixed(3),
          timestamp: new Date().toISOString(),
        });

        if (move >= MOVE_THRESHOLD) {
          console.log(`[TRIGGER] ${ticker} moved ${(move * 100).toFixed(3)}% — calling AI...`);
          analyzeWithAI(ticker, open, close);
        }
      }
    }
  });

  alpacaWs.on("error", (error) => {
    console.error("[ALPACA STREAM ERROR]", error.message);
  });

  alpacaWs.on("close", () => {
    console.log("[ALPACA STREAM] Disconnected. Reconnecting in 5s...");
    setTimeout(connectAlpacaStream, 5000);
  });
}

app.get("/api/portfolio", async (req, res) => {
  try {
    const account = await alpaca.getAccount();
    const positions = await alpaca.getPositions();

    res.json({
      account: {
        cash: account.cash,
        buying_power: account.buying_power,
        portfolio_value: account.portfolio_value,
        equity: account.equity,
      },
      positions: positions.map((p) => ({
        symbol: p.symbol,
        qty: p.qty,
        market_value: p.market_value,
        unrealized_pl: p.unrealized_pl,
        current_price: p.current_price,
        avg_entry_price: p.avg_entry_price,
      })),
    });
  } catch (error) {
    console.error("[PORTFOLIO ERROR]", error.message);
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/trade", async (req, res) => {
  try {
    const { symbol, qty, side } = req.body;

    if (!symbol || !qty || !side) {
      return res.status(400).json({ error: "Missing required fields: symbol, qty, side" });
    }

    const order = await alpaca.createOrder({
      symbol,
      qty: Number(qty),
      side,
      type: "market",
      time_in_force: "gtc",
    });

    console.log(`[TRADE] ${side.toUpperCase()} ${qty} ${symbol} — Order ID: ${order.id}`);

    broadcastToClients({
      type: "TRADE_EXECUTED",
      symbol,
      qty,
      side,
      order_id: order.id,
      timestamp: new Date().toISOString(),
    });

    res.json({
      success: true,
      order: {
        id: order.id,
        symbol: order.symbol,
        qty: order.qty,
        side: order.side,
        status: order.status,
      },
    });
  } catch (error) {
    console.error("[TRADE ERROR]", error.message);
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/health", (req, res) => {
  res.json({ status: "ok", uptime: process.uptime(), tickers: TICKERS });
});

wss.on("connection", (ws) => {
  console.log("[WS] Client connected");

  ws.send(
    JSON.stringify({
      type: "CONNECTED",
      message: "Connected to Trade Analyst AI",
      tickers: TICKERS,
      threshold: MOVE_THRESHOLD,
    })
  );

  ws.on("close", () => {
    console.log("[WS] Client disconnected");
  });
});

const PORT = 3001;
server.listen(PORT, () => {
  console.log(`[SERVER] Trade Analyst AI running on port ${PORT}`);
  console.log(`[SERVER] REST API: http://localhost:${PORT}/api`);
  console.log(`[SERVER] WebSocket: ws://localhost:${PORT}`);
  connectAlpacaStream();
});