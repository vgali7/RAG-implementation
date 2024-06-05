# RAG-implementation

Notes -

RAG - retrieval augmented generation
Used to improve the efficiency and accuracy of generative AI
Main purpose is to provide contextual relevancy to responses
Highly useful for using llm’s in specific use cases
	
A rag will use a prompt to get contextualized information from a database
		It will feel the contextualized information and prompt to an LLM
<img width="661" alt="Screenshot 2024-06-03 at 12 58 24 PM" src="https://github.com/vgali7/RAG-implementation/assets/79680489/f5fb66ea-816d-4170-9425-482a77719d32">

We want to use vectorized embeddings to capture semantic relationships
This allows the llm to capture meanings of questions and associate with the correct contextualized information

	Pros:
	Creating personal rag will allow for privacy of personal information (salesman user)
	Allows for customization (guiding the user to certain dataset endpoints based on prompts)


	Cons:
	Openai updates to chatgpt make uploading files easy and accessible
	Openai’s current Assistant + file api is seamless, although not as much for large volumes of data as it is limited to 20 files

Steps

	Identify key points of user prompt (stemming, embedding, etc.)
	Rank database chunks on relevancy to prompt
	Take most relevant chunks and feed into llm with prompt


Use vector embeddings rather than lexical to ensure semantic meaning

Potential RAG frameworks to be used	

	- langchain
		Multiple LLM interfaces
		Openai, huggingface, PaLM
		Better for end to end applications as more complete framework
		More intuitive interface which can be beneficial for MVP
		Vectorizes databases- more simple, less flexible
	- llamaindex
		More used for advanced rag using llm and data connectors 
		Focused on querying data making it robust for storage/indexing
		More flexible with building memory makes deeper llm to data connection

Langchain provides a set of abstractions for converting the database into a vector set, memory,, etc	

Langchain is written in python, for C++ you can use faiss, annoy, usearch	

Overall, langchain is more simple and intuitive than llamaindex, which is better for more data flexibility focused projects. 	

 
Amazon Elastic Cloud (EC2) -

	Compute instances in the cloud (creating a server)
	Pay for time server is up
	Provides various instance types based on cpu, memory, storage, and gpu
	Best for persistent servers
Amazon Lambda -

	Don’t have to manage servers yourself
	No charge when code isn’t running (pay when code is being used)
	Automatic scaling
	Best for event driven tasks

Amazon lambda provides a better service for our case, charging only when code is being used. Although for MVP, we only need the free version of both options


Facebook AI Similarity Search (FAISS) - Uses vectorized embeddings to approximate nearest neighbors

HuggingFace provides thousands of models for processing NLP tasks including some of the most popular as distilbert, roberta, albert. For models on inference need basis- bert is recommended - intel has provided an optimized version for question/answer called tinybert

Steps:

	Install and import dependencies for langchain 
	Find appropriate database to simulate salesman user information
	Choose a text splitter to split data into chunks - You want to maintain semantic relations between chunks
	Recursive splitter - langchain recommended as it keeps related sections of text together
	character/token - splits text based on specific desired characters/tokens
	Semantic - splits based on embeddings of sentences alone
	Using huggingfaceebeddings and FAISS, search for data chunks relevant to prompt
	Retrieve the data chunks, and feed them into an LLM


Chroma - in memory rag for learning

----------------------------------------------------------------------------------------------------------------
<img width="605" alt="Screenshot 2024-06-03 at 12 59 09 PM" src="https://github.com/vgali7/RAG-implementation/assets/79680489/b8a603e5-7339-4309-8d17-db14d4c22f2c">

**LLM Chaining** - The act of connecting an LLM to an external application (a RAG is an example where the external application is a data source)


There are various LLM chaining frameworks 

		Langchain - good for prototyping due to its breadth and simplicity
  		LLamaindex - excels at data collection, indexing, and querying, best for semantic search and retrieval
    	Haystack - best for simpler search and indexing-based LLM applications
    	AutoGen - best for multi-agent interactions, automation, and conversational prompting 

      
**Semantic router** -A super fast decision making layer for the LLM

	Allows user to configure deterministic responses to specific triggers
	Eg -> one input prompt can trigger a different set of rules for an LLM agent

**LLM agents** - 
	a system with complex reasoning capabilities, memory, and the means to execute tasks more complex than a rag.
		an llm acts as the 'brain' of the agent.
	There are thousands of potential prompts, models, use cases that can be used as agent core available on langchain hubs.
 
<img width="687" alt="Screenshot 2024-06-03 at 1 49 57 PM" src="https://github.com/vgali7/RAG-implementation/assets/79680489/fda1d1c8-f7cb-41ed-b09b-148ed4719204">

		User Request - a user question or request  	
		Agent/Brain - the agent core acting as coordinator	
		Planning - assists the agent in planning future actions		
		Memory - manages the agent's past behaviors
  		Tools - includes databases, knowledge bases, and external models.


Semantic Kernel -


Microsoft created an open source SDK called semantic kernel used for automated AI function chains - github copilot is an example

It acts as an alternative to langchain/llamaindex/etc

An LLM needs more information than the api is was built on. One way is prompt engineering

It is commonly used for creating complex code on the fly, but has a purpose in prompting as well -
reading intent of a prompt, and even configuring it in the following ways:

		Make the prompt more specific
		Add structure to the output with formatting
		Provide examples with few-shot prompting (choices)
		Tell the AI what to do to avoid doing something wrong
		Provide context to the AI (history)
		Using message roles in chat completion prompts
		Give your AI words of encouragement



**Elastic Search** - Acts as a search engine and data analysis tool. Can be used to improve efficacy of RAG
