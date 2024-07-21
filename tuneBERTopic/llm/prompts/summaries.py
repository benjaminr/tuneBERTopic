prompt = """
Keywords: {keywords_str}
Example documents:
{example_docs_str}

Generate a summary based on the above keywords and documents. The documents are only a sample of the documents, so keep that in mind when generating the summary.
Try to keep the summary to the general topic being discussed and reference only important views and details.
Return the summary in the following JSON format: {{\"summary\": \"this is the summary of the documents.\"}}

Limit your summary to 200 words maximum.
"""