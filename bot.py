import discord
from discord.ext import commands
import json
import os
from dotenv import load_dotenv
from transformers import pipeline

# 環境変数のロード
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Hugging Faceモデルセットアップ
model_name = "cardiffnlp/twitter-roberta-base-offensive"
classifier = pipeline("text-classification", model=model_name, tokenizer=model_name)

# ファイル操作関数
def load_file(file_name):
    with open(file_name, "r") as file:
        return [line.strip() for line in file]

def save_file(file_name, data):
    with open(file_name, "w") as file:
        for item in data:
            file.write(f"{item}\n")

# 初期化
intents = discord.Intents.default()
intents.messages = True
bot = commands.Bot(command_prefix="/", intents=intents)

# ファイル読み込み
whitelist = load_file("whitelist.txt")
profanity_list = load_file("profanity_list.txt")

# ホワイトリスト追加
@bot.command(name="ホワイトリスト追加")
async def add_to_whitelist(ctx, *, word: str):
    if word not in whitelist:
        whitelist.append(word)
        save_file("whitelist.txt", whitelist)
        await ctx.send(f"ホワイトリストに `{word}` を追加しました！")

# ホワイトリスト削除
@bot.command(name="ホワイトリスト削除")
async def remove_from_whitelist(ctx, *, word: str):
    if word in whitelist:
        whitelist.remove(word)
        save_file("whitelist.txt", whitelist)
        await ctx.send(f"ホワイトリストから `{word}` を削除しました！")

# メッセージ監視
@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # ホワイトリストチェック
    if any(word in message.content for word in whitelist):
        return

    # 辞書ベースの暴言チェック
    if any(word in message.content for word in profanity_list):
        await message.delete()
        await message.channel.send(f"{message.author.mention} 不適切な発言は禁止されています！")

        # ログ記録
        log_entry = {
            "message_id": message.id,
            "content": message.content,
            "author": message.author.name,
            "author_id": message.author.id,
            "channel": message.channel.name,
            "channel_id": message.channel.id
        }
        with open("deleted_messages.json", "a") as log_file:
            json.dump(log_entry, log_file)
            log_file.write("\n")
        return

    # AIによる暴言チェック
    result = classifier(message.content)
    if result[0]["label"] == "LABEL_1" and result[0]["score"] > 0.8:  # 閾値を調整可能
        await message.delete()
        await message.channel.send(f"{message.author.mention} AIにより不適切と判断されました。")

        # ログ記録
        log_entry = {
            "message_id": message.id,
            "content": message.content,
            "author": message.author.name,
            "author_id": message.author.id,
            "channel": message.channel.name,
            "channel_id": message.channel.id
        }
        with open("deleted_messages.json", "a") as log_file:
            json.dump(log_entry, log_file)
            log_file.write("\n")
    await bot.process_commands(message)

# メッセージ復元
@bot.command(name="復元")
async def restore_message(ctx, message_id: int):
    try:
        with open("deleted_messages.json", "r") as log_file:
            for line in log_file:
                log_entry = json.loads(line)
                if log_entry["message_id"] == message_id:
                    await ctx.send(
                        f"復元されたメッセージ:\n"
                        f"内容: {log_entry['content']}\n"
                        f"送信者: {log_entry['author']} (ID: {log_entry['author_id']})\n"
                        f"送信元チャンネル: {log_entry['channel']}"
                    )

                    # ホワイトリストに追加
                    if log_entry["content"] not in whitelist:
                        whitelist.append(log_entry["content"])
                        save_file("whitelist.txt", whitelist)
                        await ctx.send(f"このメッセージ内容をホワイトリストに追加しました！")
                    return
        await ctx.send("指定されたメッセージIDは見つかりませんでした。")
    except Exception as e:
        await ctx.send(f"エラーが発生しました: {e}")

bot.run(DISCORD_BOT_TOKEN)
