# Bring in deps
import os 
import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper 
import matplotlib.pyplot as plt
import re
from langchain_google_genai import ChatGoogleGenerativeAI

api_key = st.sidebar.text_input('Enter your Google AI Key', type="password")
st.sidebar.info("A Google AI Key is required.You can obtain a Google API Key by following the instructions [here](https://aistudio.google.com/app/apikey).")
st.sidebar.markdown("---")  # Horizontal rule


# Main content area
st.markdown("<h1 style='text-align: center; color: white; padding: 10px; background-color: #002147;'>Twitter Assistant</h1>", unsafe_allow_html=True)
st.markdown("---")  # Horizontal rule


# Check if API key is provided
if api_key:
    os.environ['GOOGLE_API_KEY'] = api_key
    # App framework
    prompt = st.text_input('Plug in your prompt here') 

    # Prompt templates
    title_template = PromptTemplate(
        input_variables=['topic'], 
        template='write me a Twitter title about {topic}, enlist all related hashtags on twitter, give acuurate total no of tweets till now on this topic in bold latters in saperate line, (enlist saperately hashtag  with tweets count in simple numbers formate in bullets form (format should be like #PakistanPolitics: 6,589,124 tweets))'
    )

    twitter_script_template = PromptTemplate(
        input_variables=['topic'], 
        template='write me a Twitter script based on this title TITLE: {topic}, give facts and figures and give data in numbers as much possible, give no of tweets count per hashtag, give highly impressions 5 tweets on this topic along author name and impression count'
    )

    script_template = PromptTemplate(
        input_variables=['title', 'wikipedia_research'], 
        template='write me a Twitter script based on this title TITLE: {title}, give information in fact and numeric data in 6 to 7 lines short summary, while leveraging this Wikipedia research:{wikipedia_research} '
    )
    # Memory 
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    twitter_script_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

    # Llms
    llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.9, top_p=0.85, google_api_key= api_key)
    #llm = OpenAI(temperature=0.9, model='gpt-3.5-turbo-instruct')  # Use engine instead of model
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    twitter_script_chain = LLMChain(llm=llm, prompt=twitter_script_template, verbose=True, output_key='script', memory=twitter_script_memory)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

    wiki = WikipediaAPIWrapper()

    # Show stuff to the screen if there's a prompt
    if prompt: 
        title = title_chain.run(prompt)
        twitter_script = twitter_script_chain.run(prompt)
        wiki_research = wiki.run(prompt) 
        script = script_chain.run(title=title, wikipedia_research=wiki_research)

        # Left side content
        st.write(title) 
        st.write(twitter_script) 
        print("Hashtags:", twitter_script)
        with st.expander('Title'): 
            st.info(title_memory.buffer)
        with st.expander('Wikipedia Research'): 
            st.info(wiki_research) 
        


        # Right side content
        st.sidebar.title('Hashtags vs Tweets')

        # Extract hashtags and tweet counts using regular expressions
        hashtag_counts = dict()
        matches = re.findall(r'#(\w+): (\d{1,3}(?:,\d{3})*\s+tweets)', title)

        for match in matches:
            hashtag = match[0]
            count = int(match[1].replace(',', '').split()[0])
            hashtag_counts[hashtag] = count

        hashtags = list(hashtag_counts.keys())
        tweet_counts = list(hashtag_counts.values())

        # Debug prints
        print("Hashtags:", hashtags)
        print("Tweet Counts:", tweet_counts)



        # Show graph
        st.sidebar.subheader('Number of Tweets per Hashtag')
        fig, ax = plt.subplots()
        ax.bar(hashtags, tweet_counts, label='Number of Tweets')
        ax.set_xlabel('Hashtags')
        ax.set_ylabel('Number of Tweets')
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
        ax.legend()  # Add legend
        ax.set_yscale('log')  # Set y-axis scale to logarithmic for better readability (optional)
        st.sidebar.pyplot(fig)


        # Show pie chart
        st.sidebar.subheader('Percentage of Tweets per Hashtag')
        fig, ax = plt.subplots()
        ax.pie(tweet_counts, labels=None, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.legend(hashtags, title="Hashtags", bbox_to_anchor=(1, 0.5), loc="center left", fontsize=8, title_fontsize=10)
        st.sidebar.pyplot(fig)