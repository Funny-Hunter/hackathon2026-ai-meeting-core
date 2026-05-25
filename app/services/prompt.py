from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


DATABASE_LOOKUP_PLANNER_PROMPT = ChatPromptTemplate.from_template("""
Classify whether the user's question is about meeting database metadata, meeting content, or speaker analytics.

Current date in Asia/Ho_Chi_Minh: {current_date}
Question: {question}

Return JSON only.

{format_instructions}

Schema:
{{
"intent": "meeting_database_lookup" | "meeting_content" | "speaker_analytics",
"operation": "list" | "count" | "answer",
"start_date": "YYYY-MM-DD or null",
"end_date": "YYYY-MM-DD or null"
}}

Intent rules:

- "meeting_database_lookup"
Questions about listing, searching, or counting meetings.

- "speaker_analytics"
Questions about participants or speaker-level statistics, such as:
* number of speakers
* who attended
* participant count
* speaking activity statistics
* speaker contribution comparison

- "meeting_content"
Questions asking about discussion content, statements, decisions, opinions, topics, summaries, or semantic meaning of the meeting.

Resolve relative dates only when explicitly mentioned:
- today
- yesterday
- this week
- this month

Return JSON only.
Do not explain.
""")


DATABASE_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant for meeting database questions.
Use only the provided meeting data. Do not invent meetings, dates, ids, or counts.
Answer naturally in Vietnamese. Keep the answer concise and useful."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """Question: {question}

Database lookup:
- operation: {operation}
- date_range: {date_range}
- meeting_count: {meeting_count}

Meetings JSON:
{meetings_json}"""),
])


ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
Extract entities from question.

Question: {question}

{format_instructions}

Return:
{{
    "topic": "topic or null",
    "speakers": ["speaker names"],
    "question_type": "speaker_statement|topic_discussion|speaker_topic|speaker_pair_topic|speaker_relationship|meeting_summary|recent_context"
}}
Rules:

- "What did [speaker] say?" -> speaker_statement
- "How was [topic] discussed?" -> topic_discussion
- "What did [speaker] say about [topic]?" -> speaker_topic
- "What did [speaker1] and [speaker2] say about [topic]?" -> speaker_pair_topic
- "What did [speaker1] and [speaker2] discuss?" -> speaker_relationship
- "What was the meeting about?" -> meeting_summary
- Unclear / ambiguous query -> recent_context
""")


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant for meeting analysis.
Answer the question accurately and completely using the provided context and chat history.
If the context is not sufficient to answer, say so clearly.
Respond in Vietnamese with a clear, structured answer."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """Meeting context:
{context}

Question: {question}"""),
])


SPEAKER_ANALYTICS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant for meeting speaker analytics.
Use only the provided speaker data. Do not invent participants or statistics.
Answer naturally in Vietnamese. Keep the answer concise and useful."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """Question: {question}

Meeting id: {meeting_id}
Speaker count: {speaker_count}
Speakers JSON:
{speakers_json}"""),
])
