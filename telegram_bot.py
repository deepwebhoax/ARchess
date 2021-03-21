import time
import requests
import random
import telebot
from PIL import Image
from config import token
import numpy as np
from telebot import types
from io import BytesIO

from augment import augmentReality


bot_token = token
bot = telebot.TeleBot(bot_token)
filename = 'usr_board.jpg'

@bot.message_handler(commands=['help', 'start'])
def send_info(message):
    if message.text == '/start':
        bot.send_message(message.from_user.id, 'Привіт!\nЯ вмію перемагати в шахи 😋\nНадішли мені фотографію шахової дошки.')

    elif message.text == '/help':
        # bot.send_photo(message.from_user.id, img)
        bot.send_message(message.from_user.id, 'Надішли мені зверхнє фото шахової дошки.')

@bot.message_handler(content_types=["photo"])
def send_ARed_photo(message):
    # get file id
    file_id = message.json['photo'][-1]['file_id']
    im_file = bot.get_file(file_id)
    # download and save file 
    img = requests.get('https://api.telegram.org/file/bot%s/%s' % (bot_token, im_file.file_path))
    with open(filename, 'wb') as f:
        f.write(img.content)

    try:
        modifiedImage = augmentReality(filename)
        
        # getting image bytes
        # buf = BytesIO()
        # img.save(buf, 'jpeg')
        # buf.seek(0)
        # image_bytes = buf.read()
        # buf.close()

        


        bot.send_photo(message.from_user.id, modifiedImage)
    except Exception as e:
        print(e)


@bot.message_handler(content_types=["sticker", "pinned_message", "audio"])
def send_info(message):
    bot.send_message(message.from_user.id, 'Ха-ха😄 Дуже сішно!')


while True:
    try:
        bot.polling()
    except Exception:
        time.sleep(10)