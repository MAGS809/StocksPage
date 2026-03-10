import { useState, useEffect, useRef } from "react";
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Wallet,
  AlertTriangle,
  Wifi,
  WifiOff,
  DollarSign,
  BarChart3,
} from "lucide-react";

const API_URL = "http://localhost:3001";
const WS_URL = "ws://localhost:3001";

export default function App() {
  const [alerts, setAlerts] = useState([]);
  const [account, setAccount] = useState(null);
  const [positions, setPositions] = useState([]);
  const [connected, setConnected] = useState(false);
  const [tradeStatus, setTradeStatus] = useState({});
  const wsRef = useRef(null);

  useEffect(() => {
    function connectWebSocket() {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === "AI_ALERT") {
          setAlerts((prev) => [data, ...prev].slice(0, 50));
        }

        if (data.type === "BAR_UPDATE") {
          setAlerts((prev) => {
            const existing = prev.find(
              (a) => a.type === "BAR_UPDATE" && a.ticker === data.ticker
            );
            if (existing) {
              return prev.map((a) =>
                a.type === "BAR_UPDATE" && a.ticker === data.ticker ? data : a
              );
            }
            return prev;
          });
        }

        if (data.type === "TRADE_EXECUTED") {
          setTradeStatus((prev) => ({
            ...prev,
            [data.symbol]: { side: data.side, time: Date.now() },
          }));
          fetchPortfolio();
        }
      };

      ws.onclose = () => {
        setConnected(false);
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  async function fetchPortfolio() {
    try {
      const res = await fetch(`${API_URL}/api/portfolio`);
      const data = await res.json();
      setAccount(data.account);
      setPositions(data.positions);
    } catch (err) {
      console.error("Portfolio fetch error:", err);
    }
  }

  useEffect(() => {
    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 5000);
    return () => clearInterval(interval);
  }, []);

  async function executeTrade(symbol, side) {
    try {
      setTradeStatus((prev) => ({
        ...prev,
        [symbol + side]: { loading: true },
      }));

      const res = await fetch(`${API_URL}/api/trade`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol, qty: 1, side }),
      });

      const data = await res.json();

      if (data.success) {
        setTradeStatus((prev) => ({
          ...prev,
          [symbol + side]: { success: true, time: Date.now() },
        }));
      } else {
        setTradeStatus((prev) => ({
          ...prev,
          [symbol + side]: { error: data.error, time: Date.now() },
        }));
      }

      fetchPortfolio();
    } catch (err) {
      setTradeStatus((prev) => ({
        ...prev,
        [symbol + side]: { error: err.message, time: Date.now() },
      }));
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6 font-mono">
      <div className="max-w-7xl mx-auto">
        <header className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-8 h-8 text-emerald-400" />
            <div>
              <h1 className="text-2xl font-bold text-emerald-400">
                Trade Analyst AI
              </h1>
              <p className="text-gray-500 text-sm">
                Event-Driven AI Trading Assistant
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {connected ? (
              <div className="flex items-center gap-2 text-emerald-400 text-sm">
                <Wifi className="w-4 h-4" />
                <span>LIVE</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-red-400 text-sm">
                <WifiOff className="w-4 h-4" />
                <span>DISCONNECTED</span>
              </div>
            )}
          </div>
        </header>

        <div className="grid grid-cols-3 gap-6">
          <div className="col-span-2">
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-400" />
                Terminal Feed
              </h2>

              {alerts.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  <BarChart3 className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>Waiting for market signals...</p>
                  <p className="text-xs mt-1">
                    AI alerts trigger on 0.2%+ moves in SPY/QQQ
                  </p>
                </div>
              ) : (
                <div className="space-y-3 max-h-[600px] overflow-y-auto">
                  {alerts.map((alert, i) => (
                    <div
                      key={`${alert.ticker}-${alert.timestamp}-${i}`}
                      className={`p-4 rounded-lg border ${
                        alert.type === "AI_ALERT"
                          ? "bg-gray-900 border-amber-500/30"
                          : "bg-gray-900/50 border-gray-700"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-lg font-bold text-amber-400">
                            {alert.ticker}
                          </span>
                          <span className="text-sm text-gray-400">
                            ${Number(alert.price).toFixed(2)}
                          </span>
                          {alert.type === "AI_ALERT" && (
                            <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 text-xs rounded">
                              AI ALERT
                            </span>
                          )}
                        </div>
                        <span className="text-xs text-gray-500">
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </span>
                      </div>

                      {alert.message && (
                        <p className="text-sm text-gray-300 mb-3 leading-relaxed">
                          {alert.message}
                        </p>
                      )}

                      <div className="flex gap-2">
                        <button
                          onClick={() => executeTrade(alert.ticker, "buy")}
                          disabled={
                            tradeStatus[alert.ticker + "buy"]?.loading
                          }
                          className="flex items-center gap-1 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 rounded text-sm font-medium transition-colors"
                        >
                          <TrendingUp className="w-3.5 h-3.5" />
                          BUY 1 SHARE
                        </button>
                        <button
                          onClick={() => executeTrade(alert.ticker, "sell")}
                          disabled={
                            tradeStatus[alert.ticker + "sell"]?.loading
                          }
                          className="flex items-center gap-1 px-3 py-1.5 bg-red-600 hover:bg-red-500 disabled:opacity-50 rounded text-sm font-medium transition-colors"
                        >
                          <TrendingDown className="w-3.5 h-3.5" />
                          SELL 1 SHARE
                        </button>

                        {tradeStatus[alert.ticker + "buy"]?.success && (
                          <span className="text-emerald-400 text-xs self-center">
                            Buy filled!
                          </span>
                        )}
                        {tradeStatus[alert.ticker + "sell"]?.success && (
                          <span className="text-red-400 text-xs self-center">
                            Sell filled!
                          </span>
                        )}
                        {(tradeStatus[alert.ticker + "buy"]?.error ||
                          tradeStatus[alert.ticker + "sell"]?.error) && (
                          <span className="text-amber-400 text-xs self-center">
                            {tradeStatus[alert.ticker + "buy"]?.error ||
                              tradeStatus[alert.ticker + "sell"]?.error}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="col-span-1 space-y-6">
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Wallet className="w-5 h-5 text-emerald-400" />
                Portfolio Status
              </h2>

              {account ? (
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 bg-gray-900 rounded">
                    <span className="text-gray-400 text-sm">Cash</span>
                    <span className="text-emerald-400 font-bold">
                      <DollarSign className="w-4 h-4 inline" />
                      {Number(account.cash).toLocaleString("en-US", {
                        minimumFractionDigits: 2,
                      })}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-gray-900 rounded">
                    <span className="text-gray-400 text-sm">Buying Power</span>
                    <span className="text-blue-400 font-bold">
                      <DollarSign className="w-4 h-4 inline" />
                      {Number(account.buying_power).toLocaleString("en-US", {
                        minimumFractionDigits: 2,
                      })}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-gray-900 rounded">
                    <span className="text-gray-400 text-sm">
                      Portfolio Value
                    </span>
                    <span className="text-white font-bold">
                      <DollarSign className="w-4 h-4 inline" />
                      {Number(account.portfolio_value).toLocaleString("en-US", {
                        minimumFractionDigits: 2,
                      })}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="text-center py-6 text-gray-500">
                  <p className="text-sm">Loading account...</p>
                </div>
              )}
            </div>

            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-blue-400" />
                Open Positions
              </h2>

              {positions.length === 0 ? (
                <div className="text-center py-6 text-gray-500">
                  <p className="text-sm">No open positions</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {positions.map((pos) => (
                    <div
                      key={pos.symbol}
                      className="flex justify-between items-center p-3 bg-gray-900 rounded"
                    >
                      <div>
                        <span className="font-bold text-white">
                          {pos.symbol}
                        </span>
                        <span className="text-gray-400 text-sm ml-2">
                          x{pos.qty}
                        </span>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium">
                          ${Number(pos.market_value).toFixed(2)}
                        </div>
                        <div
                          className={`text-xs ${
                            Number(pos.unrealized_pl) >= 0
                              ? "text-emerald-400"
                              : "text-red-400"
                          }`}
                        >
                          {Number(pos.unrealized_pl) >= 0 ? "+" : ""}
                          ${Number(pos.unrealized_pl).toFixed(2)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}