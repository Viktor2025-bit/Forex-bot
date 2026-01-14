import json
import logging
import urllib.request
import urllib.parse
import ssl

logger = logging.getLogger(__name__)

class TelegramService:
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self._context = ssl.create_default_context()
        self._context.check_hostname = False
        self._context.verify_mode = ssl.CERT_NONE

    def send_message(self, message: str):
        """Send a text message to the configured chat."""
        if not self.enabled or not self.bot_token:
            return

        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            headers = {'Content-Type': 'application/json'}
            req = urllib.request.Request(
                url, 
                data=json.dumps(data).encode('utf-8'), 
                headers=headers
            )
            
            with urllib.request.urlopen(req, context=self._context) as response:
                if response.getcode() != 200:
                    logger.error(f"Telegram send failed: {response.read()}")
                    
        except Exception as e:
            logger.error(f"Telegram Error: {e}")

    def send_trade_alert(self, symbol: str, side: str, price: float, qty: float):
        """Format and send a trade alert."""
        icon = "🟢" if side.upper() == 'BUY' else "🔴"
        msg = f"{icon} *TRADE EXECUTED*\n\n" \
              f"Symbol: `{symbol}`\n" \
              f"Side: *{side.upper()}*\n" \
              f"Price: `{price}`\n" \
              f"Qty: `{qty}`\n" \
              f"Time: `{json.dumps(str(price))}`" # localized time ideally
              
        self.send_message(msg)

if __name__ == "__main__":
    # Test
    # Replace with real token to test
    service = TelegramService("TOKEN", "CHAT_ID", enabled=False)
    service.send_message("Test from Bot")
