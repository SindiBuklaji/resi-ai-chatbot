# RESI AI Conversation Bot

This is the conversation bot for the RESI. It is a multi-lingual bot for helping people how to control RESI.

## Usage

1. create a **.env** file and paste there your *OPENAI_API_KEY*. The content of the **.env** should be identical to:
    ```py
    OPENAI_API_KEY=sk-xxxx
    ```
    which should be 51 character-long in total.


2. Run the app using the command:
    ```py
    streamlit run app.py    #Or, to select a port, you can run the command
    stremalit run app.py --server.port=85XX
    ```

3. According to the settings, open a browser and the app should now be running on: 
    ```py
        http://localhost:85XX/          # or
        http://127.0.0.1:85XX/          
    ```
    Note even without specifying the IP and port in the app, these values will be set by default.
