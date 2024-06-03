# RAG-implementation

Notes -

RAG - retrieval augmented generation
Used to improve the efficiency and accuracy of generative AI
Main purpose is to provide contextual relevancy to responses
Highly useful for using llm’s in specific use cases
	
A rag will use a prompt to get contextualized information from a database
		It will feel the contextualized information and prompt to an LLM

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




Semantic router -
A super fast decision making layer for the LLM 
 	Allows user to configure deterministic responses to specific triggers
	Eg -> one input prompt can trigger a different set of rules for an LLM agent

LLM actions - 

LLM agents - 

Semantic Kernel -

LLM Chaining -
