# Prairie Chatbot
A program which takes questions and anwers them with information from the the "documents" folder. In this project, the documents and example queries asked regard information about the Blackland Prarie of Central Texas, but it can be modified to suit your documents

# How to use
- Generate API keys for [Pinecone](https://www.pinecone.io/) and [Hugging Face](https://huggingface.co/). Rename the ".env-template" file ".env" and paste your keys in.
- Run chatbot.py. Check your Pinecone console to make sure an index was created and the documents were loaded in properly.
- When the terminal reads "Prompt:", you may type your question. You can also use prewritten prompts in the "prompts.py" file by adding an exclamation mark (!) before the name of the variable containing the prompt.
- Exit by typing "q", "exit", "f", "quit". 

# Sources
- Blackland Prarie Ecological Region (https://tpwd.texas.gov/landwater/land/habitats/cross_timbers/ecoregions/blackland.phtml) - Texas Parks & Wildlife
- Native and Adapted Landscape Plants - an earthwise guide for Central Texas (https://services.austintexas.gov/watershed_protection/publications/document.cfm?id=198301) - City of Austin
- Plant Guidance by Ecoregions (https://tpwd.texas.gov/huntwild/wild/wildlife_diversity/wildscapes/ecoregions/ecoregion_4.phtml) - Texas Parks & Wildlife
