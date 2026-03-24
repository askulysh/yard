import telebot
import json
import os

# ==========================================
# LOADING TELEGRAM BOT TOKEN FROM JSON
# ==========================================
config_path = "telegram.json"
if not os.path.exists(config_path):
    print(f"Error: {config_path} not found. Please create it first.")
    exit(1)

with open(config_path, "r") as f:
    config = json.load(f)
    TELEGRAM_BOT_TOKEN = config.get("token")

if not TELEGRAM_BOT_TOKEN:
    print(f"Error: 'token' not found in {config_path}.")
    exit(1)

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Welcome to the Yard Parking Bot! 🚗\nSend /status to get the latest parking slot status and a live image.")

@bot.message_handler(commands=['status'])
def send_status(message):
    img_path = "latest_result.jpg"
    status_path = "status.json"

    # 1. Read the latest textual status
    status_text = "🅿️ **Current Parking Status:**\n\n"
    if os.path.exists(status_path):
        try:
            with open(status_path, "r") as f:
                slots = json.load(f)

            free_slots = 0
            for s in slots:
                icon = "🔴" if s['status'] == "Occupied" else "🟢"
                status_text += f"{icon} {s['name']}: {s['status']}\n"
                if s['status'] == "Free":
                    free_slots += 1

            status_text += f"\nTotal Free: {free_slots}/{len(slots)}"
        except Exception as e:
            status_text += f"Error reading text status: {e}"
    else:
        status_text = "Parking text data not available yet."

    # 2. Send the image and caption
    if os.path.exists(img_path):
        try:
            with open(img_path, 'rb') as photo:
                bot.send_photo(message.chat.id, photo, caption=status_text)
        except Exception as e:
            bot.reply_to(message, f"Failed to send image. Error: {e}")
    else:
        bot.reply_to(message, f"{status_text}\n\n(No image available yet. Is main.py running?)")

if __name__ == "__main__":
    print("Starting Telegram Bot...")
    print("Waiting for messages (press Ctrl+C to stop)...")
    # Using skip_pending=True so it doesn't process old messages when restarting
    bot.infinity_polling(skip_pending=True)
