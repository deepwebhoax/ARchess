import time

import requests
import random
import telebot
from PIL import Image
from config import token
import numpy as np
from telebot import types

bot_token = token
bot = telebot.TeleBot(bot_token)


@bot.message_handler(commands=['help', 'start'])
def send_info(message):
    if message.text == '/start':
        bot.send_message(message.from_user.id, 'Привіт!\nЯ вмію перемагати в шахи 😋\nНадішли мені фотографію шахової дошки.')

    elif message.text == '/help':
        img = Image.open('img/help.jpg')
        bot.send_photo(message.from_user.id, img)

@bot.message_handler(content_types=["photo"]):
def send_ARed_photo():
    

@bot.message_handler(content_types=["sticker", "pinned_message", "audio"])
def send_info(message):
    bot.send_message(message.from_user.id, 'Ха-ха😄 Дуже сішно!')

def gen_logo(text):

    # generating logo
    if random.random() > 0.6:
        logo = gen_logo_color()
    else:
        logo = model.generate_logo()
    # applying superresolution and filtering
    logo = superresolute(logo)
    logo = imgfilter(logo)
    # adding company name on the image
    logo = add_text_to_img(text, logo)
    logo = Image.fromarray(logo)

     # create keyboard
    keyboard = types.InlineKeyboardMarkup()
    # create button
    mem_button = types.InlineKeyboardButton(text="Получить мем", callback_data='meme')
    more_button = types.InlineKeyboardButton(text="Еще", callback_data='more')
    keyboard.add(mem_button)
    keyboard.add(more_button)
    return logo, keyboard

@bot.message_handler()
def company_receiving(message):

    logo, keyboard = gen_logo(message.text)
    # sending picture with keyboard to user
    bot.send_photo(message.from_user.id, logo, caption=message.text,
                   reply_markup=keyboard)



    
@bot.callback_query_handler(lambda query: query.data == 'more')
def process_callback(query):
    try:
        
        logo, keyboard = gen_logo(query.message.caption)
        bot.send_photo(query.from_user.id, logo, caption=query.message.caption,
                    reply_markup=keyboard)
    except Exception as e:
        print(e)


@bot.callback_query_handler(lambda query: query.data == 'meme')
def process_callback(query):
    # get link to file
    im_sizes = query.message.json['photo'][0]
    h, w = im_sizes['height'], im_sizes['width']
    file_id = im_sizes['file_id']
    im_file = bot.get_file(file_id)
    # download file
    img = requests.get('https://api.telegram.org/file/bot%s/%s' % (bot_token, im_file.file_path))
    with open('query.img', 'wb') as f:
        f.write(img.content)
    try:
        img = Image.open('query.img')
        encod_img = np.array(img)
        result = get_examples(encod_img)
        result = Image.fromarray(result)
        bot.send_photo(query.from_user.id, result)
    except Exception as e:
        print(e)


while True:
    try:
        bot.polling()
    except Exception:
        time.sleep(1)