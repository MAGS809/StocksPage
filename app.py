import os
import praw
import alpaca_trade_api as tradeapi

# 1. API Credentials (Securely loaded via Environment Variables in Replit)
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'

# 2. Reddit API Credentials
reddit = praw.Reddit(
    client_id=os.environ.get('REDDIT_CLIENT_ID'),
    client_secret=os.environ.get('REDDIT_SECRET'),
    user_agent='Framd_AI_Bot_v1'
)

# Initialize Alpaca Connection
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')
conn = tradeapi.stream.Stream(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL, data_feed='iex')

# 3. The Reddit Scraper
def get_reddit_news(symbol, limit=3):
    """Pulls the most recent post titles mentioning the ticker."""
    news_headlines = []
    # Searching major stock subreddits for the specific ticker
    subreddit = reddit.subreddit('wallstreetbets+stocks+investing')
    
    # Grab the newest posts matching our symbol
    for submission in subreddit.search(symbol, sort='new', limit=limit):
        news_headlines.append(submission.title)
        
    return news_headlines

# 4. The AI Decision Engine 
def ask_framd_ai(price_data, reddit_context):
    # This is where your AI model analyzes both the price action AND the Reddit hype
    print(f"Current Price: {price_data}")
    print(f"Latest Reddit Chatter: {reddit_context}")
    
    # Placeholder for your AI's actual output
    return "HOLD" 

# 5. The Execution Logic
def execute_trade(action, symbol):
    if action == "BUY":
        try:
            api.submit_order(
                symbol=symbol,
                qty=1,
                side='buy',
                type='market',
                time_in_force='gtc',
                stop_loss={'stop_price': 145.00} # Hardcoded risk management
            )
            print(f"Executed BUY order for {symbol}")
        except Exception as e:
            print(f"Error executing trade: {e}")

# 6. The Data Stream Listener
async def on_quote(q):
    current_price = q.askprice
    symbol = q.symbol
    
    # 1. Fetch the latest Reddit news for the stock
    latest_news = get_reddit_news(symbol)
    
    # 2. Feed BOTH price and news to the AI
    ai_signal = ask_framd_ai(current_price, latest_news)
    
    # 3. Execute if the AI gives the green light
    if ai_signal == "BUY":
        execute_trade("BUY", symbol)

# Start the Engine
print("Connecting to live market data and Reddit APIs...")
conn.subscribe_quotes(on_quote, 'AAPL')
conn.run()
