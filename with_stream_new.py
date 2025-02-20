#!/usr/bin/env python
# coding: utf-8
#spectrum.app@inxeption.com / Spectrum4ChatGPT!
import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import numpy as np
import sys
import time
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import json
from colorama import Fore, Back, Style
import ast 
################################################################################
### Step 1
################################################################################                                                                                                                                                                                                   


#load_dotenv(Path(r"C:\Users\norin.saiyed\web-crawl-q-and-a\.env"))
load_dotenv()
#api_key = os.environ["API_KEY"]
openai.api_key = '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'
#openai.api_key = api_key
def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    #f'text-search-{size}-doc-001'
    #text-embedding-ada-002

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question_2(
    df,
    model="gpt-3.5-turbo",
    #text-davinci-003
    question="",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=1800,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        
        print("\n\n")

    try:
        prompt= f" \n\nContext: {context}\n\n Question: {question}\nAnswer:"
        #print(prompt)
        
        # instructions = f"""Create an email bot from Inxeption marketing team to generate buyer emails with unique subject line for welcome and product specific.
        # Emails from Inxeption solar panel buyers within the specified word counts, extracted from user input. 
        # Employ conditional statements or variables to ensure content length aligns with user preferences. 
        # Validate word count consistently during generation to adhere to user's specification. 
        # For instance, if the user seeks emails below 150 words, adapt content by removing non-essential elements and must provide following link for more information: [here](https://inxeption.com/search-results/products?s=products&page=1&). 
        # If word count exceeds, truncate while retaining essentials and If it's below, add relevant details.  
        # For specific products provide introduction of that product in one line with product URLs and URLs are accurately extracted from the current context and lead users to the appropriate product pages on Inxeption's marketplace.
        # Optimize content while respecting user-defined word limits.
        # Instruct the Chatbot to dynamically adjust the content to meet the user's specified word count while ensuring the key message is conveyed effectively.
        # Condense some sentences and remove unnecessary words while retaining the main message and tone.
        # Remove any unnecessary or redundant information that is not essential for the update on the shipment.
        # To reduce the length, prioritize the most crucial information while still making the email engaging and informative.
        # Use clear and concise language to convey the message effectively without unnecessary words.
        # Remember to maintain a friendly tone and always base your suggestions on the existing context without creating new product URLs.
        # """

        sender_name = "Alok Kumar Das email:- alok.das@ext.i"

        # instructions = f"""
        #     Create email template bot from Inxeption marketing as per user requirement with unique subject line for welcome,check in and product specific email.
        #     Extract the word count limit specified by the user and must be strictly adhere on it for the email template generation.
        #     Use conditional statements or variables to control the content length based on the user's suggestion. For example, if the user wants an email template below 150 words, you can adjust the content length accordingly by removing non-essential details.
        #     If the word count exceeds the limit, truncate or shorten the content while preserving essential information to meet the desired length.
        #     If the word count is below the limit, consider adding relevant details to achieve the user's desired word count.
        #     Continuously validate the word count during text generation to strictly adhere to the user's specified length.
        #     Emphasize the importance of maintaining conciseness and coherence within the specified word count.
        #     Instruct the Chatbot to dynamically adjust the content to meet the user's specified word count while ensuring the key message is conveyed effectively.
        #     Condense some sentences and remove unnecessary words while retaining the main message and tone.
        #     Remove any unnecessary or redundant information that is not essential for the update on the shipment.
        #     To reduce the length, prioritize the most crucial information while still making the email engaging and informative.
        #     Use clear and concise language to convey the message effectively without unnecessary words.
        #     For 25 words count provide as default with product name and for more information you can check from [here](https://inxeption.com/search-results/products?s=solar+panels).
        #     For welcome template with 25 word count do not include any product details only provide default URL for more information from [here](https://inxeption.com/search-results/products?s=solar+panels).
        #     For specific product introduction, ensure that to provide product URL and all product URLs are directly taken from the existing context and are valid, leading to the correct product pages on Inxeption's marketplace.
        #     Remember to maintain a friendly tone and always base your suggestions on the existing context without creating new product URLs,contact nos.
        #     """
        
        instructions = f"""
            Create email template bot from Inxeption marketing as per user requirement with unique subject line for welcome,check in and product specific email.
            Extract the word count limit specified by the user and must be strictly adhere on it for the email template generation.
            Use conditional statements or variables to control the content length based on the user's suggestion. For example, if the user wants an email template below 150 words, you can adjust the content length accordingly by removing non-essential details.
            If the word count exceeds the limit, truncate or shorten the content while preserving essential information to meet the desired length.
            If the word count is below the limit, consider adding relevant details to achieve the user's desired word count.
            Continuously validate the word count during text generation to strictly adhere to the user's specified length.
            Emphasize the importance of maintaining conciseness and coherence within the specified word count.
            Instruct the Chatbot to dynamically adjust the content to meet the user's specified word count while ensuring the key message is conveyed effectively.
            Condense some sentences and remove unnecessary words while retaining the main message and tone.
            Remove any unnecessary or redundant information that is not essential for the update on the shipment.
            To reduce the length, prioritize the most crucial information while still making the email engaging and informative.
            Use clear and concise language to convey the message effectively without unnecessary words.
            For 25 words count provide as default with product name and for more information you can check from [here](https://inxeption.com/search-results/products?s=solar+panels).
            For welcome template with 25 word count do not include any product details only provide default URL for more information from [here](https://inxeption.com/search-results/products?s=solar+panels).
            For specific product introduction, ensure that to provide product URL and all product URLs are directly taken from the existing context and are valid, leading to the correct product pages on Inxeption's marketplace.
            Remember to maintain a friendly tone and always base your suggestions on the existing context without creating new product URLs,contact nos.
            Show the image url of the product in the body.
            

            Here are the details of **[productname](Product_URL)**
            <div class="table_1">
            | |  |
            | :--- | :--- |
            | <td rowspan="8" class="div_image">![Product Image](IMAGE_URL "a title") </td>  |
            | | **Usage:** USAGE_TYPE
            | | **Manufacturer:** MANUFACTURER |
            | | **Type:** Monocrystalline |
            | | **Number of cells:** NUMBER_OF_CELLS |
            | | **Wattage:** WATTAGE |
            | | **Weight:** WEIGHT |
            | | **Height:** HEIGHT |

            Regards,
            {sender_name}

            Example:
            Subject: Solar4America is the next big thing !!

            Dear [Customer Name],

            Welcome to Inxeption Energy Marketplace! We are excited to have you as a new customer and provide you with the best solar energy solutions. 
            As per your request, here is a brief introduction to one of our popular products:

            - Product Name: Solar4America-S4A410-72MH5BB 
            - Manufacturer: Solar4America
            - Type: Monocrystalline
            - Number of Cells: 72
            - Wattage: 410
            - Weight: 22 kg
            - Height: 40 mm

            For more information about Solar4America-S4A410-72MH5BB, please visit the product page [here](https://inxeptionmarketplace-crate.inxeption.io/purl/inxeptionenergymarketplace-solar4america-s4a410-72mh5bb).

            ![Product Image](https://s3.amazonaws.com/prisma-oce.inxeption.io/ledger207/7909e9aa-e221-4468-b697-1bbac67b2817/Solar4America-S4A410-72MH5BB-1.jpg)

            If you have any questions or need further assistance, please feel free to reach out to us. We are here to help!

            Thank you for choosing Inxeption Energy Marketplace. We look forward to serving you.


            Best regards,
            Alok Kumar Das
            
            alok.das@ext.inxeption.com

            """    

        response = openai.ChatCompletion.create(
            
            model="gpt-3.5-turbo",
            messages = [{ "role": "system", "content": instructions},{"role": "user", "content": prompt}],
            max_tokens = 1024,
            temperature = 0,
            stream = True)
        #message = response.choices[0].message.content
        #print(message)
        collected_chunks = []
        collected_messages = []
        # iterate through the stream of events
        for chunk in response:
            #chunk_time = time.time() - start_time  # calculate the time delay of the chunk
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk['choices'][0]['delta']  # extract the message
            #collected_messages.append(chunk_message)  # save the message
            #print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text
            #print(chunk_message)
            if "content" in chunk_message:
                message_text = chunk_message['content']
                #yield message_text
                print(Fore.RED + Style.BRIGHT +message_text, end='',flush=True)
                #print(Fore.CYAN + Style.BRIGHT + "INX: " + Style.NORMAL +Fore.RED + Style.BRIGHT+ Style.NORMAL +message_text)
                #print(f"{message_text}", end="")
                #collected_messages += message_text
            #time.sleep(1)
                
        # print the time delay and text received
        #print(f"Full response received {chunk_time:.2f} seconds after request")
        full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
        #print(f"Full conversation received: {full_reply_content}")
    except Exception as e:
        print(e)
        return ""

################################################################################
### Step 2
################################################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide an input argument.")
    else:
        try:
            #df = pd.read_parquet('processed/dataset_850prod.parquet.gzip') ##working
            # df = pd.read_parquet('dataset_850prod_updated.parquet.gzip') ##working
            df = pd.read_parquet('Askiris_package_updated.parquet.gzip')
            question_test = sys.argv[1]
            start_time = time.time()
            replay = answer_question_2(df, question=question_test, debug=False)
            #print("##########################")
            #print(replay)
            #end_time = time.time()
            #diff = end_time - start_time
            #print(diff)
        except Exception as e:
            print(e)

        




