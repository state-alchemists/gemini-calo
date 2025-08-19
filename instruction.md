# Instruction

Make a rollup-conversation middleware, the idea is as follow:

- First, we have LRU cache with certain value
- Whenever we receive completion request (openai compatible/gemini native), we will try to  turn user + assistant messages into md5 and compare it to our LRU. If LRU is found, then:
    - The system prompt will be injected with the data on LRU (past x conversation + conversation summary context)
    - The user messages matching the LRU is deleted from payload
    - The LRU should check the longer sequences of user message first. For example if the payload contains 5 user message then:
        - We check message 1-5, turn it into md5 and lookup the LRU
        - If not found, we check message 1-4, etc
    - The system prompt should not be part of LRU checking
- If conversation length achieving some threshold, we will send request to gemini to roll up the conversation based on matching LRU and all the conversation length

So the middleware will look like this:


create_rollup_middleware(lru_size, conversation_size_threshold, summarizer_prompt)

The summarizer_prompt will have default value.